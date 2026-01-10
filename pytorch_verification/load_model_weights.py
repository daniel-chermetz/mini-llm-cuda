"""
Model Weight Loader for PyTorch Verification

Loads binary model weights exported from the JavaScript implementation.
Handles conversion from row-major (JS) to column-major (PyTorch) storage.

Also handles vocabulary and sample loading.

Binary Format:
- 8 bytes: header length (uint64, little-endian)
- N bytes: JSON metadata (UTF-8)
- Padding to 8-byte alignment
- Tensor data (float32/float64, packed sequentially)

Key transformations:
1. All 2D tensors: stored as column-major (transposed view)
2. Embedding weights: physically transposed from [vocabSize, dim] to [dim, vocabSize]
"""

import torch
import numpy as np
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ==========================================
# Vocabulary Loading
# ==========================================

def load_vocab(filepath: str, verbose: bool = True) -> Tuple[List[str], Dict[str, int]]:
    """
    Load vocabulary from JSON file.
    
    Args:
        filepath: Path to vocab.json file
        verbose: Print loading info
    
    Returns:
        Tuple of:
        - vocab_list: List of tokens indexed by token ID
        - token_to_idx: Dict mapping token string to index
    """
    filepath = Path(filepath)
    
    if verbose:
        print(f"--- Loading vocabulary from {filepath} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)
    
    # Create token to index mapping
    token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    
    if verbose:
        print(f"  Vocabulary size: {len(vocab_list)}")
        print(f"  Sample tokens: {vocab_list[:5]}...")
    
    return vocab_list, token_to_idx


def load_sample(
    filepath: str,
    token_to_idx: Dict[str, int],
    max_length: int = 256,
    pad_token: str = "~",
    verbose: bool = True
) -> Tuple[List[str], torch.Tensor]:
    """
    Load sample tokens from JSON file.
    
    Args:
        filepath: Path to sample.json file
        token_to_idx: Token to index mapping from load_vocab
        max_length: Maximum sequence length (L)
        pad_token: Token to use for padding
        verbose: Print loading info
    
    Returns:
        Tuple of:
        - tokens: List of token strings (padded/truncated to max_length)
        - token_indices: Tensor of token indices [max_length]
    """
    filepath = Path(filepath)
    
    if verbose:
        print(f"--- Loading sample from {filepath} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    
    original_length = len(tokens)
    
    # Truncate if longer than max_length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        if verbose:
            print(f"  Truncated from {original_length} to {max_length} tokens")
    
    # Pad if shorter than max_length
    elif len(tokens) < max_length:
        pad_count = max_length - len(tokens)
        tokens = tokens + [pad_token] * pad_count
        if verbose:
            print(f"  Padded from {original_length} to {max_length} tokens (added {pad_count} '{pad_token}')")
    
    # Convert to indices
    token_indices = []
    unknown_tokens = set()
    
    for token in tokens:
        if token in token_to_idx:
            token_indices.append(token_to_idx[token])
        else:
            # Handle unknown tokens - use pad token index as fallback
            unknown_tokens.add(token)
            token_indices.append(token_to_idx.get(pad_token, 0))
    
    if unknown_tokens and verbose:
        print(f"  Warning: {len(unknown_tokens)} unknown tokens found: {list(unknown_tokens)[:5]}...")
    
    if verbose:
        print(f"  Final sequence length: {len(token_indices)}")
        print(f"  First 10 token indices: {token_indices[:10]}")
    
    return tokens, torch.tensor(token_indices, dtype=torch.long)


# ==========================================
# Model Weight Loading
# ==========================================


def load_model_weights(
    filepath: str,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load model weights from binary format.
    
    Args:
        filepath: Path to the binary weights file
        device: Target device ('cpu' or 'cuda')
        dtype: Target dtype for tensors
        verbose: Print loading progress
    
    Returns:
        Dictionary containing:
        - 'token_embeddings': [DIM, VOCAB_SIZE] tensor (physically transposed from original [VOCAB_SIZE, DIM])
        - 'final_rms_gamma': [DIM] tensor
        - 'layers': List of layer weight dicts (shapes preserved, column-major storage)
    """
    filepath = Path(filepath)
    
    if verbose:
        print(f"--- Loading model weights from {filepath} ---")
    
    with open(filepath, 'rb') as f:
        file_data = f.read()
    
    if verbose:
        print(f"  File size: {len(file_data) / 1024 / 1024:.2f} MB")
    
    # Parse header length (8 bytes, little-endian uint64)
    header_length = struct.unpack('<Q', file_data[:8])[0]
    header_start = 8
    header_end = header_start + header_length
    
    if header_end > len(file_data):
        raise ValueError("Header length exceeds file size - file may be corrupted")
    
    # Parse JSON metadata
    header_bytes = file_data[header_start:header_end]
    metadata = json.loads(header_bytes.decode('utf-8'))
    
    if verbose:
        print(f"  Header parsed ({header_length} bytes)")
    
    # Calculate data offset with 8-byte alignment
    ALIGNMENT = 8
    padding_needed = (ALIGNMENT - (header_end % ALIGNMENT)) % ALIGNMENT
    data_offset = header_end + padding_needed
    
    if verbose:
        print(f"  Data starts at offset {data_offset}")
    
    # Helper to read a tensor from the binary data
    def read_tensor(tensor_meta: Dict, current_offset: int) -> Tuple[np.ndarray, int]:
        """Read a tensor from binary data and return (tensor, new_offset)."""
        shape = tensor_meta['shape']
        dtype_str = tensor_meta['dtype']
        
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        if dtype_str == 'float32':
            bytes_per_element = 4
            np_dtype = np.float32
        elif dtype_str == 'float64':
            bytes_per_element = 8
            np_dtype = np.float64
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        
        byte_length = num_elements * bytes_per_element
        
        if current_offset + byte_length > len(file_data):
            raise ValueError(f"Attempting to read past end of file at offset {current_offset}")
        
        # Read raw bytes and convert to numpy array
        raw_bytes = file_data[current_offset:current_offset + byte_length]
        flat_data = np.frombuffer(raw_bytes, dtype=np_dtype)
        
        # Reshape to original shape (row-major / C order, as stored in JS)
        tensor = flat_data.reshape(shape)
        
        return tensor, current_offset + byte_length
    
    def to_column_major_torch(arr: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor with column-major storage.
        
        For 2D arrays:
        - Keeps the same logical shape [rows, cols]
        - Reorders memory so columns are contiguous (Fortran/column-major order)
        - Element [i, j] stays at logical position [i, j], but is stored at flat index j*rows + i
        
        For 1D arrays:
        - Returns as-is (no concept of row/column major for 1D)
        """
        if arr.ndim == 1:
            # 1D tensor - just convert
            return torch.from_numpy(arr.copy()).to(device=device, dtype=dtype)
        
        elif arr.ndim == 2:
            # Convert to Fortran (column-major) order while keeping same logical shape
            # This reorders memory so columns are contiguous: [col0_all_rows, col1_all_rows, ...]
            arr_col_major = np.asfortranarray(arr)
            return torch.from_numpy(arr_col_major.copy()).to(device=device, dtype=dtype)
        
        else:
            raise ValueError(f"Unsupported tensor dimension: {arr.ndim}")
    
    # Track current position in data section
    current_offset = data_offset
    
    # ==========================================
    # Load Token Embeddings
    # ==========================================
    # Original shape: [vocabSize, dim] in row-major
    # Target shape: [dim, vocabSize] with physical transpose
    if verbose:
        print("  Loading token embeddings...")
    
    token_emb_meta = metadata['tokenEmbeddings']
    token_emb_np, current_offset = read_tensor(token_emb_meta, current_offset)
    
    original_emb_shape = token_emb_np.shape
    # Physically transpose: [vocabSize, dim] -> [dim, vocabSize]
    # And store as column-major (which means dim-dimension is contiguous)
    token_embeddings = torch.from_numpy(token_emb_np.T.copy()).to(device=device, dtype=dtype)
    
    if verbose:
        print(f"    Original shape: {original_emb_shape} -> Transposed to: {tuple(token_embeddings.shape)}")
    
    # ==========================================
    # Load Final RMS Norm Gamma
    # ==========================================
    if verbose:
        print("  Loading final RMS norm gamma...")
    
    final_rms_meta = metadata['finalRMSNormGamma']
    final_rms_np, current_offset = read_tensor(final_rms_meta, current_offset)
    final_rms_gamma = torch.from_numpy(final_rms_np.copy()).to(device=device, dtype=dtype)
    
    if verbose:
        print(f"    Shape: {tuple(final_rms_gamma.shape)}")
    
    # ==========================================
    # Load Transformer Blocks
    # ==========================================
    if verbose:
        print(f"  Loading {len(metadata['transformerBlocks'])} transformer blocks...")
    
    layers = []
    
    # Expected tensor names in each block (from binary file)
    # These map to our PyTorch weight names
    tensor_name_mapping = {
        # RMS Norm weights
        'rmsGamma': 'rms_gamma',
        'rmsGamma2': 'rms_gamma2',
        # Attention weights (actual names from binary)
        'queryWeights': 'Wq',
        'keyWeights': 'Wk', 
        'valueWeights': 'Wv',
        'outputProjectionWeights': 'Wo',
        # FFN weights (actual names from binary)
        'feedForwardWeights1A': 'W_ffn1',   # Gate projection (with SiLU)
        'feedForwardWeights1B': 'W_ffn2',   # Up projection
        'feedForwardWeights2': 'W_ffn_left', # Down projection
        # Alternative naming conventions that might appear
        'Wq': 'Wq',
        'Wk': 'Wk',
        'Wv': 'Wv',
        'Wo': 'Wo',
        'W_ffn1': 'W_ffn1',
        'W_ffn2': 'W_ffn2',
        'W_ffn_left': 'W_ffn_left',
    }
    
    for block_idx, block_meta in enumerate(metadata['transformerBlocks']):
        if verbose:
            print(f"    Block {block_idx}...")
        
        layer_weights = {}
        
        for tensor_name, tensor_meta in block_meta.items():
            # Read tensor
            tensor_np, current_offset = read_tensor(tensor_meta, current_offset)
            
            # Determine PyTorch name
            pytorch_name = tensor_name_mapping.get(tensor_name, tensor_name)
            
            # Convert to column-major PyTorch tensor (shape preserved, memory reordered)
            tensor_pt = to_column_major_torch(tensor_np)
            
            layer_weights[pytorch_name] = tensor_pt
            
            if verbose:
                orig_shape = tensor_np.shape
                new_shape = tuple(tensor_pt.shape)
                storage_info = "(col-major)" if tensor_np.ndim == 2 else ""
                print(f"      {tensor_name} -> {pytorch_name}: {orig_shape} -> {new_shape} {storage_info}")
        
        layers.append(layer_weights)
    
    if verbose:
        bytes_read = current_offset - data_offset
        print(f"  Total tensor data read: {bytes_read / 1024 / 1024:.2f} MB")
        print(f"--- Model loading complete ---")
    
    return {
        'token_embeddings': token_embeddings,
        'final_rms_gamma': final_rms_gamma,
        'layers': layers,
    }


def assign_weights_to_model(
    model_weights: Dict[str, Any],
    x: Optional[torch.Tensor] = None,
    rms_gamma: Optional[List[torch.Tensor]] = None,
    rms_gamma2: Optional[List[torch.Tensor]] = None,
    Wq: Optional[List[torch.Tensor]] = None,
    Wk: Optional[List[torch.Tensor]] = None,
    Wv: Optional[List[torch.Tensor]] = None,
    Wo: Optional[List[torch.Tensor]] = None,
    W_ffn1: Optional[List[torch.Tensor]] = None,
    W_ffn2: Optional[List[torch.Tensor]] = None,
    W_ffn_left: Optional[List[torch.Tensor]] = None,
) -> None:
    """
    Assign loaded weights to existing model weight tensors.
    
    This copies data from loaded weights into the provided tensors,
    preserving requires_grad and other tensor properties.
    """
    layers = model_weights['layers']
    num_layers = len(layers)
    
    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        
        if rms_gamma is not None and 'rms_gamma' in layer:
            rms_gamma[layer_idx].data.copy_(layer['rms_gamma'])
        
        if rms_gamma2 is not None and 'rms_gamma2' in layer:
            rms_gamma2[layer_idx].data.copy_(layer['rms_gamma2'])
        
        if Wq is not None and 'Wq' in layer:
            Wq[layer_idx].data.copy_(layer['Wq'])
        
        if Wk is not None and 'Wk' in layer:
            Wk[layer_idx].data.copy_(layer['Wk'])
        
        if Wv is not None and 'Wv' in layer:
            Wv[layer_idx].data.copy_(layer['Wv'])
        
        if Wo is not None and 'Wo' in layer:
            Wo[layer_idx].data.copy_(layer['Wo'])
        
        if W_ffn1 is not None and 'W_ffn1' in layer:
            W_ffn1[layer_idx].data.copy_(layer['W_ffn1'])
        
        if W_ffn2 is not None and 'W_ffn2' in layer:
            W_ffn2[layer_idx].data.copy_(layer['W_ffn2'])
        
        if W_ffn_left is not None and 'W_ffn_left' in layer:
            W_ffn_left[layer_idx].data.copy_(layer['W_ffn_left'])


def verify_shapes(model_weights: Dict[str, Any], config: Dict[str, int]) -> bool:
    """
    Verify that loaded weight shapes match expected configuration.
    
    Args:
        model_weights: Loaded weights dictionary
        config: Expected configuration with keys:
            - DIM: Model dimension
            - FFN_DIM: FFN hidden dimension  
            - VOCAB_SIZE: Vocabulary size
            - LAYERS: Number of transformer layers
    
    Returns:
        True if all shapes match, False otherwise
    """
    DIM = config['DIM']
    FFN_DIM = config['FFN_DIM']
    VOCAB_SIZE = config['VOCAB_SIZE']
    LAYERS = config['LAYERS']
    
    all_ok = True
    
    # Check token embeddings: should be [DIM, VOCAB_SIZE] after transpose
    emb_shape = tuple(model_weights['token_embeddings'].shape)
    expected_emb = (DIM, VOCAB_SIZE)
    if emb_shape != expected_emb:
        print(f"ERROR: token_embeddings shape {emb_shape} != expected {expected_emb}")
        all_ok = False
    
    # Check final RMS gamma: should be [DIM]
    rms3_shape = tuple(model_weights['final_rms_gamma'].shape)
    if rms3_shape != (DIM,):
        print(f"ERROR: final_rms_gamma shape {rms3_shape} != expected ({DIM},)")
        all_ok = False
    
    # Check number of layers
    if len(model_weights['layers']) != LAYERS:
        print(f"ERROR: {len(model_weights['layers'])} layers != expected {LAYERS}")
        all_ok = False
    
    # Check each layer's weights
    # Column-major conversion preserves shapes (only memory layout changes)
    expected_shapes = {
        'rms_gamma': (DIM,),      # 1D, unchanged
        'rms_gamma2': (DIM,),     # 1D, unchanged
        'Wq': (DIM, DIM),         # [DIM, DIM] preserved
        'Wk': (DIM, DIM),
        'Wv': (DIM, DIM),
        'Wo': (DIM, DIM),
        'W_ffn1': (FFN_DIM, DIM), # [FFN_DIM, DIM] preserved
        'W_ffn2': (FFN_DIM, DIM),
        'W_ffn_left': (DIM, FFN_DIM),  # [DIM, FFN_DIM] preserved
    }
    
    for layer_idx, layer in enumerate(model_weights['layers']):
        for name, expected in expected_shapes.items():
            if name in layer:
                actual = tuple(layer[name].shape)
                if actual != expected:
                    print(f"ERROR: Layer {layer_idx} {name} shape {actual} != expected {expected}")
                    all_ok = False
    
    if all_ok:
        print("✓ All weight shapes verified successfully")
    
    return all_ok


# ==========================================
# Standalone test / example usage
# ==========================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and verify model weights')
    parser.add_argument('filepath', type=str, help='Path to binary weights file')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--ffn-dim', type=int, default=2048, help='FFN dimension')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load weights
    weights = load_model_weights(
        args.filepath,
        device=args.device,
        verbose=True
    )
    
    # Verify shapes
    config = {
        'DIM': args.dim,
        'FFN_DIM': args.ffn_dim,
        'VOCAB_SIZE': args.vocab_size,
        'LAYERS': args.layers,
    }
    
    print("\n=== Shape Verification ===")
    verify_shapes(weights, config)
    
    # Print sample values
    print("\n=== Sample Values ===")
    print(f"token_embeddings [0:4, 0]:\n  {weights['token_embeddings'][0:4, 0].tolist()}")
    print(f"final_rms_gamma [0:4]:\n  {weights['final_rms_gamma'][0:4].tolist()}")
    
    if len(weights['layers']) > 0:
        layer0 = weights['layers'][0]
        if 'Wq' in layer0:
            print(f"Layer 0 Wq [0:4, 0]:\n  {layer0['Wq'][0:4, 0].tolist()}")
        if 'W_ffn1' in layer0:
            print(f"Layer 0 W_ffn1 [0:4, 0]:\n  {layer0['W_ffn1'][0:4, 0].tolist()}")
