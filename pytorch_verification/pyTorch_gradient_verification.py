import torch
import math
from load_model_weights import load_model_weights, load_vocab, load_sample

# ==========================================
# 0. Configuration
# ==========================================
DIM = 512
FFN_DIM = 2048
L = 256
EPS = 1e-8
LAYERS = 4
ATTN_HEADS = 8
ROPE_BASE = 10000.0
VOCAB_SIZE = 10097  # From model.bin

# Loss configuration
# rightEndIndex: compute loss from position 0 to rightEndIndex (inclusive)
# Each position i predicts token at position i+1, so rightEndIndex can be 0 to L-2 (254)
RIGHT_END_INDEX = 100  # Can be 0 to 254

# File paths
MODEL_PATH = "./model.bin"
VOCAB_PATH = "./vocab.json"
SAMPLE_PATH = "./sample.json"

HEAD_DIM = DIM // ATTN_HEADS  # 64
SCALE = 1.0 / math.sqrt(HEAD_DIM)

# Device configuration
device = 'mps'  # Change to 'cuda' if you want to verify float precision on GPU

# ==========================================
# 1. Load Vocabulary and Sample
# ==========================================
vocab_list, token_to_idx = load_vocab(VOCAB_PATH)
sample_tokens, sample_indices = load_sample(SAMPLE_PATH, token_to_idx, max_length=L)

# ==========================================
# 2. RoPE Precomputation (Fixed / No Grad)
# ==========================================
# Logic: Successive Pairs (0,1), (2,3)...
# Standard formula: theta_i = base ^ (-2(i)/d) for i = 0, 1, ... d/2-1
with torch.no_grad():
    # 1. Create Theta frequencies for pairs
    theta_indices = torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM
    inv_freq = 1.0 / (ROPE_BASE ** theta_indices)  # Shape: [HEAD_DIM/2]

    # 2. Create Position indices [0, ..., L-1]
    t_pos = torch.arange(L, device=device, dtype=torch.float32)

    # 3. Outer Product -> [HEAD_DIM/2, L]
    freqs = torch.outer(inv_freq, t_pos)

    # 4. Cos/Sin tables -> [HEAD_DIM/2, L]
    cos_vals = freqs.cos()
    sin_vals = freqs.sin()

# ==========================================
# 3. Load Model Weights from Binary File
# ==========================================
print(f"\n--- Loading model weights from {MODEL_PATH} ---")
model_weights = load_model_weights(MODEL_PATH, device=device, verbose=True)

# Extract weights into variables with requires_grad=True
# Token Embeddings: [DIM, VOCAB_SIZE] (already transposed by loader)
token_embeddings = model_weights['token_embeddings'].clone().requires_grad_(True)

# Final RMS Norm Gamma: [DIM] -> reshape to [DIM, 1]
rms_gamma_final = model_weights['final_rms_gamma'].unsqueeze(1).clone().requires_grad_(True)

# Per-layer weights stored in lists
# Note: Model file has layer 0 at top of stack (near output), layer N-1 at bottom (near input)
# We iterate from LAYERS-1 down to 0, so layer order in lists matches file order
rms_gamma = []
rms_gamma2 = []
Wq = []
Wk = []
Wv = []
Wo = []
W_ffn1 = []
W_ffn2 = []
W_ffn_left = []

for layer_idx in range(LAYERS):
    layer_weights = model_weights['layers'][layer_idx]
    
    # RMSNorm Weights: [DIM] -> reshape to [DIM, 1]
    rms_gamma.append(layer_weights['rms_gamma'].unsqueeze(1).clone().requires_grad_(True))
    rms_gamma2.append(layer_weights['rms_gamma2'].unsqueeze(1).clone().requires_grad_(True))
    
    # Attention Weights: [DIM, DIM] (column-major storage, shape preserved)
    Wq.append(layer_weights['Wq'].clone().requires_grad_(True))
    Wk.append(layer_weights['Wk'].clone().requires_grad_(True))
    Wv.append(layer_weights['Wv'].clone().requires_grad_(True))
    Wo.append(layer_weights['Wo'].clone().requires_grad_(True))
    
    # FFN Weights (column-major storage, shapes preserved from bin file)
    # W_ffn1: [FFN_DIM, DIM]
    # W_ffn2: [FFN_DIM, DIM]
    # W_ffn_left: [DIM, FFN_DIM]
    W_ffn1.append(layer_weights['W_ffn1'].clone().requires_grad_(True))
    W_ffn2.append(layer_weights['W_ffn2'].clone().requires_grad_(True))
    W_ffn_left.append(layer_weights['W_ffn_left'].clone().requires_grad_(True))

print(f"  Loaded {LAYERS} transformer layers")

# ==========================================
# 4. Create Input from Sample
# ==========================================
# Look up embeddings for each token: token_embeddings[:, token_idx] for each position
# Result: [DIM, L]
print(f"\n--- Creating input embeddings from sample ---")
x = token_embeddings[:, sample_indices].clone().requires_grad_(True)
x.retain_grad()  # x is non-leaf (derived from token_embeddings), so retain its grad
print(f"  Input shape: {x.shape}")

# ==========================================
# 5. Forward Pass
# ==========================================
print("\n--- Starting Forward Pass ---")

# Current hidden state starts as input x
hidden = x

# Iterate from bottom of stack (LAYERS-1) to top (0)
# Layer LAYERS-1 receives input embeddings, layer 0 outputs to final RMS
for layer in range(LAYERS - 1, -1, -1):
    print(f"  Layer {layer}...")
    
    # --- Step A: RMS Norm (Llama 3 style) ---
    # x_norm = x * w * rsqrt(mean(x^2) + eps)
    hidden_sq_mean = hidden.pow(2).mean(dim=0, keepdim=True)  # Mean across feat dim
    rsqrt = torch.rsqrt(hidden_sq_mean + EPS)
    hidden_norm = hidden * rsqrt * rms_gamma[layer]

    # --- Step B: QKV Projections ---
    # Weights are stored column-major with shape [DIM, DIM]
    # [DIM, DIM] @ [DIM, L] -> [DIM, L]
    q_flat = torch.matmul(Wq[layer], hidden_norm)
    k_flat = torch.matmul(Wk[layer], hidden_norm)
    v_flat = torch.matmul(Wv[layer], hidden_norm)

    # --- Step C: Reshape (Split Heads) ---
    # [DIM, L] -> [Heads, HeadDim, L]
    q_heads = q_flat.view(ATTN_HEADS, HEAD_DIM, L)
    k_heads = k_flat.view(ATTN_HEADS, HEAD_DIM, L)
    v_heads = v_flat.view(ATTN_HEADS, HEAD_DIM, L)

    # --- Step D: RoPE (Successive Pairs) ---
    # 1. Reshape to separate pairs: [Heads, HeadDim/2, 2, L]
    q_pairs = q_heads.view(ATTN_HEADS, HEAD_DIM // 2, 2, L)
    k_pairs = k_heads.view(ATTN_HEADS, HEAD_DIM // 2, 2, L)

    # 2. Split Even/Odd components (Real/Imaginary)
    q_r, q_i = q_pairs[:, :, 0, :], q_pairs[:, :, 1, :]
    k_r, k_i = k_pairs[:, :, 0, :], k_pairs[:, :, 1, :]

    # 3. Broadcast Cos/Sin: [HeadDim/2, L] -> [1, HeadDim/2, L]
    cos_bc = cos_vals.unsqueeze(0)
    sin_bc = sin_vals.unsqueeze(0)

    # 4. Apply Rotation
    # Even_new = Even * cos - Odd * sin
    # Odd_new  = Odd * cos + Even * sin
    q_r_rot = q_r * cos_bc - q_i * sin_bc
    q_i_rot = q_i * cos_bc + q_r * sin_bc

    k_r_rot = k_r * cos_bc - k_i * sin_bc
    k_i_rot = k_i * cos_bc + k_r * sin_bc

    # 5. Stack and Flatten back to [Heads, HeadDim, L]
    q_rot = torch.stack([q_r_rot, q_i_rot], dim=2).view(ATTN_HEADS, HEAD_DIM, L)
    k_rot = torch.stack([k_r_rot, k_i_rot], dim=2).view(ATTN_HEADS, HEAD_DIM, L)

    # --- Step E: Attention Scores ---
    # S = (K^T @ Q) * Scale
    # K: [Heads, D, L] -> Transpose -> [Heads, L, D]
    # Q: [Heads, D, L]
    # Result: [Heads, L, L] where Rows=Keys, Cols=Queries
    scores = torch.matmul(k_rot.transpose(-1, -2), q_rot)
    scores = scores * SCALE

    # --- Step F: Causal Masking ---
    # Mask where Row(Key) > Col(Query).
    # This corresponds to the strict lower triangle.
    mask = torch.tril(torch.ones(L, L, device=device), diagonal=-1).bool()
    scores.masked_fill_(mask, float('-inf'))

    # --- Step G: Softmax ---
    # Normalize over Rows (Keys) -> dim=-2
    attn_probs = torch.softmax(scores, dim=-2)

    # --- Step H: Weighted Sum ---
    # O = V @ A
    # V: [Heads, D, L]
    # A: [Heads, L, L]
    # Out: [Heads, D, L]
    attn_out = torch.matmul(v_heads, attn_probs)

    # --- Step I: Reshape Output (Optional, usually concat heads) ---
    # Flatten back to [DIM, L] for residual connection
    attn_out_flat = attn_out.reshape(DIM, L)

    # --- Step J: Output Projection ---
    # Wo is stored column-major with shape [DIM, DIM]
    # [DIM, DIM] @ [DIM, L] -> [DIM, L]
    output_proj = torch.matmul(Wo[layer], attn_out_flat)

    # --- Step K: Residual Connection (post-attention) ---
    # Add input hidden to projected output
    post_attn_residual = output_proj + hidden

    # --- Step L: RMS Norm 2 (post-attention) ---
    post_attn_sq_mean = post_attn_residual.pow(2).mean(dim=0, keepdim=True)
    rsqrt2 = torch.rsqrt(post_attn_sq_mean + EPS)
    post_attn_norm = post_attn_residual * rsqrt2 * rms_gamma2[layer]

    # --- Step M: FFN Projections ---
    # W_ffn1/W_ffn2 are stored column-major with shape [FFN_DIM, DIM]
    # [FFN_DIM, DIM] @ [DIM, L] -> [FFN_DIM, L]
    ffn_right_1_pre_silu = torch.matmul(W_ffn1[layer], post_attn_norm)
    ffn_right_2 = torch.matmul(W_ffn2[layer], post_attn_norm)

    # --- Step N: SiLU Activation ---
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    ffn_right_1_post_silu = ffn_right_1_pre_silu * torch.sigmoid(ffn_right_1_pre_silu)

    # --- Step O: Hadamard (Element-wise) Product ---
    ffn_hadamard = ffn_right_1_post_silu * ffn_right_2

    # --- Step P: FFN Left (Down Projection) ---
    # W_ffn_left is stored column-major with shape [DIM, FFN_DIM]
    # [DIM, FFN_DIM] @ [FFN_DIM, L] -> [DIM, L]
    ffn_out = torch.matmul(W_ffn_left[layer], ffn_hadamard)

    # --- Step Q: Residual Connection (post-FFN) ---
    # Add post-attention residual to FFN output
    hidden = ffn_out + post_attn_residual

# Final output after all layers
final_output = hidden

# --- Step R: Final RMS Norm (post all transformer layers) ---
final_sq_mean = final_output.pow(2).mean(dim=0, keepdim=True)
rsqrt_final = torch.rsqrt(final_sq_mean + EPS)
output_post_rms_final = final_output * rsqrt_final * rms_gamma_final

# --- Step S: Calculate Logits (Embedding Scores) ---
# embeddings.T [VOCAB_SIZE, DIM] @ output [DIM, L] -> [VOCAB_SIZE, L]
logits = torch.matmul(token_embeddings.T, output_post_rms_final)

# --- Step T: Softmax over vocabulary ---
# Apply softmax with numerical stability
# Normalize over vocab dimension (dim=0) for each position
# logits: [VOCAB_SIZE, L], probs: [VOCAB_SIZE, L]
# For each column (position), probs sum to 1 over all vocab tokens
probs = torch.softmax(logits, dim=0)

# ==========================================
# 6. Calculate Loss (Cross-Entropy for Language Modeling)
# ==========================================
print(f"\n--- Calculating Loss (positions 0 to {RIGHT_END_INDEX}) ---")

# For each position i, the target is the token at position i+1
# target_indices[i] = sample_indices[i+1] for i in 0..RIGHT_END_INDEX
target_indices = sample_indices[1:RIGHT_END_INDEX + 2]  # Positions 1 to RIGHT_END_INDEX+1

# Get the probabilities of the correct next tokens
# probs[target_token, position] for each position
# We need positions 0 to RIGHT_END_INDEX
positions = torch.arange(RIGHT_END_INDEX + 1)
correct_probs = probs[target_indices, positions]  # Shape: [RIGHT_END_INDEX + 1]

# Cross-entropy loss: -log(prob of correct token), summed over positions
# Add small epsilon for numerical stability in log
log_probs = torch.log(correct_probs + 1e-10)
loss = -log_probs.sum()

print(f"  Target tokens (first 10): {target_indices[:10].tolist()}")
print(f"  Correct probs (first 10): {correct_probs[:10].tolist()}")
print(f"  Loss: {loss.item():.6f}")

# ==========================================
# 7. Backward Pass
# ==========================================
print("\n--- Starting Backward Pass ---")

# Trigger Backprop
loss.backward()

# ==========================================
# 8. Verification Output
# ==========================================
print(f"\n=== Sample Info ===")
print(f"First 10 tokens: {sample_tokens[:10]}")
print(f"First 10 indices: {sample_indices[:10].tolist()}")

print(f"\n=== Shapes Verification ===")
print(f"Input x:             {x.shape}")
print(f"Final Output:        {final_output.shape} (Expect: [{DIM}, {L}])")
print(f"Output (post final RMS): {output_post_rms_final.shape} (Expect: [{DIM}, {L}])")
print(f"Logits:              {logits.shape} (Expect: [{VOCAB_SIZE}, {L}])")
print(f"Probs (softmax):     {probs.shape} (Expect: [{VOCAB_SIZE}, {L}])")

# Note: Layer 0 is at TOP of stack (outputs to final RMS)
#       Layer LAYERS-1 is at BOTTOM (receives input embeddings)
print(f"\n=== Gradient Shapes Verification (Layer 0 - top of stack) ===")
print(f"dL/dx:               {x.grad.shape}")
print(f"dL/dWq[0]:           {Wq[0].grad.shape}")
print(f"dL/dWk[0]:           {Wk[0].grad.shape}")
print(f"dL/dWv[0]:           {Wv[0].grad.shape}")
print(f"dL/dWo[0]:           {Wo[0].grad.shape}")
print(f"dL/dRmsGamma[0]:     {rms_gamma[0].grad.shape}")
print(f"dL/dRmsGamma2[0]:    {rms_gamma2[0].grad.shape}")
print(f"dL/dW_ffn1[0]:       {W_ffn1[0].grad.shape}")
print(f"dL/dW_ffn2[0]:       {W_ffn2[0].grad.shape}")
print(f"dL/dW_ffn_left[0]:   {W_ffn_left[0].grad.shape}")

print(f"\n=== Final Layer Gradient Shapes ===")
print(f"dL/dRmsGammaFinal:   {rms_gamma_final.grad.shape}")
print(f"dL/dTokenEmbeddings: {token_embeddings.grad.shape}")

# ==========================================
# 9. Embedding Gradient Verification
# ==========================================
# Verify PyTorch correctly scatter-adds x.grad back to token_embeddings.grad
# x.grad has shape [DIM, L] - gradient for each position
# token_embeddings.grad has shape [DIM, VOCAB_SIZE] - accumulated gradient per token type
#
# IMPORTANT: Only positions 0 to RIGHT_END_INDEX contribute to the loss!
# Positions beyond RIGHT_END_INDEX (padding) have zero gradient.
#
# If token 9999 appears at positions 5 and 10 (both <= RIGHT_END_INDEX), then:
#   token_embeddings.grad[:, 9999] = x.grad[:, 5] + x.grad[:, 10]

print(f"\n=== Embedding Gradient Verification ===")
print(f"x.grad shape: {x.grad.shape} (gradient per position)")
print(f"token_embeddings.grad shape: {token_embeddings.grad.shape} (accumulated per token type)")

# Only positions 0 to RIGHT_END_INDEX contribute to loss (they predict tokens 1 to RIGHT_END_INDEX+1)
# Positions beyond RIGHT_END_INDEX have no gradient flow
loss_positions = RIGHT_END_INDEX + 1  # positions 0..RIGHT_END_INDEX inclusive

# Manually compute what we expect token_embeddings.grad to be
# by scatter-adding x.grad ONLY for positions that contribute to loss
manual_emb_grad = torch.zeros_like(token_embeddings)
for pos in range(loss_positions):  # Only 0 to RIGHT_END_INDEX
    token_idx = sample_indices[pos]
    manual_emb_grad[:, token_idx] += x.grad[:, pos]

# Check if PyTorch's automatic gradient matches our manual computation
emb_grad_diff = (token_embeddings.grad - manual_emb_grad).abs().max().item()
print(f"Loss computed for positions 0 to {RIGHT_END_INDEX} ({loss_positions} positions)")
print(f"Max diff between PyTorch emb grad and manual scatter-add: {emb_grad_diff:.2e}")
if emb_grad_diff < 1e-6:
    print("  ✓ PyTorch correctly accumulates embedding gradients!")
else:
    print("  ✗ WARNING: Gradients don't match - there may be an issue")

# Verify that x.grad is zero for positions beyond RIGHT_END_INDEX (padding)
if RIGHT_END_INDEX + 1 < L:
    padding_grad_max = x.grad[:, RIGHT_END_INDEX + 1:].abs().max().item()
    print(f"Max x.grad beyond position {RIGHT_END_INDEX}: {padding_grad_max:.2e}")
    if padding_grad_max < 1e-6:
        print("  ✓ Padding positions have zero gradient (as expected)")

# Show token frequency and gradient accumulation example
# Only count tokens in positions that contribute to loss
sample_indices_in_loss = sample_indices[:loss_positions]
unique_tokens, counts = torch.unique(sample_indices_in_loss, return_counts=True)
multi_occur = [(t.item(), c.item()) for t, c in zip(unique_tokens, counts) if c > 1]
print(f"\nTokens appearing multiple times in loss positions: {len(multi_occur)}")
if multi_occur:
    # Show an example of gradient accumulation
    example_token, example_count = multi_occur[0]
    # Only consider positions within loss range
    positions_of_token = [p for p in (sample_indices == example_token).nonzero(as_tuple=True)[0].tolist() 
                          if p < loss_positions]
    print(f"  Example: Token {example_token} appears {len(positions_of_token)}x at positions {positions_of_token}")
    
    # Sum of x.grad at those positions (only within loss range)
    x_grad_sum = sum(x.grad[:, pos] for pos in positions_of_token)
    emb_grad_at_token = token_embeddings.grad[:, example_token]
    print(f"  Sum of x.grad at those positions [0:4]: {x_grad_sum[0:4].tolist()}")
    print(f"  token_embeddings.grad[:, {example_token}] [0:4]: {emb_grad_at_token[0:4].tolist()}")

print(f"\n=== Sample Gradient Values (Layer 0 - top of stack, near output) ===")
print("You can compare these floats with your CUDA implementation.")
print(f"dL/dx [0:4, 0]:\n {x.grad[0:4, 0].tolist()}")
print(f"dL/dWq[0] [0:4, 0]:\n {Wq[0].grad[0:4, 0].tolist()}")
print(f"dL/dWo[0] [0:4, 0]:\n {Wo[0].grad[0:4, 0].tolist()}")
print(f"dL/dW_ffn1[0] [0:4, 0]:\n {W_ffn1[0].grad[0:4, 0].tolist()}")
print(f"dL/dW_ffn2[0] [0:4, 0]:\n {W_ffn2[0].grad[0:4, 0].tolist()}")
print(f"dL/dW_ffn_left[0] [0:4, 0]:\n {W_ffn_left[0].grad[0:4, 0].tolist()}")

print(f"\n=== Sample Gradient Values (Layer {LAYERS-1} - bottom of stack, near input) ===")
print(f"dL/dWq[{LAYERS-1}] [0:4, 0]:\n {Wq[LAYERS-1].grad[0:4, 0].tolist()}")
print(f"dL/dWo[{LAYERS-1}] [0:4, 0]:\n {Wo[LAYERS-1].grad[0:4, 0].tolist()}")
print(f"dL/dW_ffn_left[{LAYERS-1}] [0:4, 0]:\n {W_ffn_left[LAYERS-1].grad[0:4, 0].tolist()}")

print(f"\n=== Sample Final Layer Gradients ===")
print(f"dL/dRmsGammaFinal [0:4]: {rms_gamma_final.grad[0:4, 0].tolist()}")
print(f"dL/dTokenEmbeddings [0:4, token 0]: {token_embeddings.grad[0:4, 0].tolist()}")
print(f"dL/dx [0:4, position 0]: {x.grad[0:4, 0].tolist()}")

# Optional: Export for C++ comparison
# torch.save({'x_grad': x.grad, 'wq_grad': Wq[0].grad}, 'ref_grads.pt')