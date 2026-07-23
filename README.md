### LLM training with CUDA (Nvidia GPU required)

### 100% of training.cu, 100% of inference.cu, 100% of optimizer.cu, 100% of inference_ple.cu, and 100% of training_ple.cu were manually human written line-by-line. The rest of the project is largely or wholly AI written.

Defaults to Llama-3.2-1B architecutre (except for MHA instead of GQA), but now with optional Q/K RMSNorm, Qwen-style query gating, and Gemma 4 inspired Per Layer Embeddings (though with my own unique formula). Can be extended to support multiple GPUs with less than 100 lines of code change.

Training optimizer: AdEMAMix with weight decay, except for vocab (can be easily changed to other Adam variants).

Model size (hidden dimension, FFN dimension, number of layers, vocab size etc.) and training hyperparameters are fully configurable.

(All this project's capabilities & indeed many more are available on PyTorch, TensorFlow, JAX - so I didn't come up with anything new (except perhaps the PLE formula), but this project does offer a minimalistic implementation that's of educational value.)
