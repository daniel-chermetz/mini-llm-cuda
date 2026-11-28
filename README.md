### LLM training with CUDA (Nvidia GPU required)

### the following files were 100% manually human written line by line:
inference.cu,
inference_ple.cu
linear_attention.cu,
training.cu
training_ple.cu,
training_linear_attention.cu,
optimizer.cu

###  The rest of the project is largely or wholly AI written.

Defaults to Llama-3.2-1B architecutre (except for a smaller default model size, and MHA instead of GQA), but now with optional Q/K RMSNorm, Qwen-style query gating, Mamba-2 like linear attention, and Gemma 4 inspired Per Layer Embeddings (though with my own unique formula). The project can be extended to support multiple GPUs with less than 100 lines of code change.

Training optimizer: AdEMAMix with weight decay, except for vocab (can be easily changed to other Adam variants).

Model size (hidden dimension, FFN dimension, number of layers, vocab size etc.) and training hyperparameters are fully configurable.

###  New: Mamba-2 style linear attention

Each transformer in the stack can be configred to either use ordinary multi-head self-attention, or linear attention per Mamba-2's formula, but with an additional write gating as per Gated DeltaNet (yet without Gated DeltaNet's erase term). 

Linear layers make post-training inference much faster and more memory efficient, as well as improve model capabilities, but the training might be slightly slower than training a comparably sized model with only ordinary self-attention across all transformers.

(All this project's capabilities & indeed many more are available on PyTorch, TensorFlow, JAX - so I didn't come up with anything new (except perhaps the PLE formula), but this project does offer a minimalistic implementation that's of educational value.)
