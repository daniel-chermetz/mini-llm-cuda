__global__ void apply_1D_conv(
	float* x_post_conv_post_silu, 
	float* x_post_conv, 
	const float* x, 
	const float* conv_weights, 
	const int dim, 
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= dim * L) {
		return;
	}

	int col = index / dim;
	int channel = index - col * dim;
	int channelOffset = channel * 4;

	float w0 = conv_weights[channelOffset];
	float w1 = conv_weights[channelOffset + 1];
	float w2 = conv_weights[channelOffset + 2];
	float w3 = conv_weights[channelOffset + 3];

	int x2_index = index - dim;
	int x1_index = x2_index - dim;
	int x0_index = x1_index - dim;

	float x0 = x0_index >= 0 ? x[x0_index] : 0;
	float x1 = x1_index >= 0 ? x[x1_index] : 0;
	float x2 = x2_index >= 0 ? x[x2_index] : 0;
	float x3 = x[index];

	float val = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
	x_post_conv[index] = val;
	x_post_conv_post_silu[index] = val / (1.0f + expf(-val));
}

__global__ void apply_l2_norm(
	float* x_post_l2_norm, 
	const float* x,
	const int headDim, 
	const int dim,
	const int L
) {
    int globalColIndex = blockIdx.x;
    int headIndex = globalColIndex / L;
    int headRelativeColIndex = globalColIndex - headIndex * L;
    int tIndex = threadIdx.x;

    int colOffset = headRelativeColIndex * dim + headIndex * headDim;

    extern __shared__ float sData[];

    float threadSum = 0.0f;
    for (int rowIndex = tIndex; rowIndex < headDim; rowIndex += blockDim.x) {
    	threadSum += (x[colOffset + rowIndex] * x[colOffset + rowIndex]);
    }
    sData[tIndex] = threadSum;
    __syncthreads();

    for (int reductionSize = blockDim.x / 2; reductionSize > 0; reductionSize /= 2) {
    	if (tIndex < reductionSize) {
    		sData[tIndex] = sData[tIndex] + sData[tIndex + reductionSize];
    	}
    	__syncthreads();
    }

    if (tIndex == 0) {
    	sData[0] = sqrtf(sData[0] + 1e-6f);
    }
    __syncthreads();

 	for (int rowIndex = tIndex; rowIndex < headDim; rowIndex += blockDim.x) {
 		int index = colOffset + rowIndex;
 		x_post_l2_norm[index] = x[index] / sData[0];
 	}
}

__global__ void set_alpha_beta_gates(
	float* alpha_gate,
	float* beta_gate,
	const float* alpha,
	const float* beta,
	const float* A_by_head_weights,
	const float* bias_by_head_weights,
	const int numAttnHeads,
	const int L
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numAttnHeads * L) {
		return;
	}

	int headRelativeColIndex = index / numAttnHeads;
    int headIndex = index - headRelativeColIndex * numAttnHeads;

    float x = alpha[index] + bias_by_head_weights[headIndex];
    float softplus_x = x > 20.0f ? x : log1pf(expf(x)); // ln(1 + e^(a + bias))
    alpha_gate[index] = expf(-expf(A_by_head_weights[headIndex]) * softplus_x);

    beta_gate[index] = 1.0f / (1.0f + expf(-beta[index]));
}

/*
S is not weights!
St ~ v_1 * k_1_transpose + v_2 * k_2_transpose + ... v_t * k_t_transpose
What's the good of that?

output_t = St * q_t = 

(v_1 * k_1_transpose + v_2 * k_2_transpose + ... + v_t * k_t_transpose) * q_t =

(v_1 * k_1_transpose * q_t + v_2 * k_2_transpose * q_t + ... + v_t * k_t_transpose * q_t)

And this is really the regular attention (except wobbly because except q, the v * k_transpose are not exact)!

Let's verify the shapes:

v_t * k_t_transpose = [headDim, 1] * [1, headDim] = [headDim, headDim]

q_t = [headDim, 1]

v_t * k_t_transpose * q_t = [headDim, 1]

Goal at every token:

Step 1: have the kernel erase from S any k_t_transpose data, so that all rows would become orthogonal to k_t_transpose and multiply by k_t_transpose results in zero
Step 2: have the kernel add to S: v_t * k_t_transpose, so that S * k_t_transpose = v_t (well approximately given that the k directions are not really orthogonal)

Step 1: St-1 - St-1 * k_t * k_t_transpose

But there's a way to calculate it more easily, and we can combine step 2 as well:

St = St-1 - St-1 * k_t * k_t_transpose + v_t * k_t_transpose =

St-1 + (-St-1 * k_t + v_t) * k_t_transpose

delta = v_t - St-1 * k_t

Is just matrix * headDim vector easily reducable to headDim threads with a result of [headDim, 1]

Then perhaps each thread can multiply one element of k_t_transpose by the whole column and add it to St-1

Or by rows is even better with each row element in k_t_transpose multiplying one feature of the column on the left side, and saving each result in a row of St, with the row index corresponding to the feature index in the column on the left side of the multiplication.

With gates this becomes:

alpha * St-1 + (-alpha * St-1 * k_t + v_t) * beta * k_t_transpose
*/

__global__ void run_state_update_loop(
	float* output,
	// float* state,
	const float* values,
	const float* keys,
	const float* queries,
	const float* alpha_gate,
	const float* beta_gate,
	const int L
) {
	int headIndex = blockIdx.x;
	int tIndex = threadIdx.x;
	int values_keys_queries_offset = headIndex * headDim;

	float tIndex_delta;
	float tIndex_state_row[headDim] = {0.0f};

	__shared__ float sKeys[headDim];
	__shared__ float sQueries[headDim];	

	for (int token_index = 0; token_index < L; token_index++) {
		float tIndex_feature_output = 0.0f;

		sKeys[tIndex] = keys[values_keys_queries_offset + tIndex];
		sQueries[tIndex] = queries[values_keys_queries_offset + tIndex];		
		__syncthreads(); // to ensure all key features for the current token_index have been written

		int gateIndex = headIndex + token_index * attnHeads;
		float alpha_gate_val = alpha_gate[gateIndex];		

		float v_feature_tIndex_prediction_on_prior_state = 0.0f; // negative of v_prediction
		for (int state_col_index = 0; state_col_index < headDim; state_col_index++) {
			tIndex_state_row[state_col_index] *= alpha_gate_val;
			v_feature_tIndex_prediction_on_prior_state -= (tIndex_state_row[state_col_index] * sKeys[state_col_index]);
		}
		tIndex_delta = (v_feature_tIndex_prediction_on_prior_state + values[values_keys_queries_offset + tIndex]) * beta_gate[gateIndex];

		for (int keys_row_index = 0; keys_row_index < headDim; keys_row_index++) {
			tIndex_state_row[keys_row_index] += tIndex_delta * sKeys[keys_row_index];
			tIndex_feature_output += tIndex_state_row[keys_row_index] * sQueries[keys_row_index];
		}
		output[values_keys_queries_offset + tIndex] = tIndex_feature_output;

		values_keys_queries_offset += dim;
		__syncthreads(); // to ensure all keys and queries can be safely overwritten
	}

	// we don't need it for forward
	/*
		int stateRowOffset = headIndex * headDim + tIndex;
		for (int state_col_index = 0; state_col_index < headDim; state_col_index++) {
			state[stateRowOffset + state_col_index * dim] = tIndex_state_row[state_col_index];
		}
	*/
}

St = St-1 + (-St-1 * Kt + Vt) * Kt_transposed

delta_t = -St-1 * Kt + Vt

St = St-1 + delta_t * Kt_transposed

delta_1 = -S0 * K1 + V1
S1 = 
S0 + delta_1 * K1_transposed =
S0 + (-S0 * K1 + V1) * K1_transposed

S2 = 
S1 + delta_2 * K2_transposed =
S1 + (-S1 * K2 + V2) * K2_transposed =
S1 + (-     (S0 + (-S0 * K1 + V1) * K1_transposed)     * K2 + V2) * K2_transposed =
S1 + (-     (S0 + delta_1 * K1_transposed)     * K2 + V2) * K2_transposed =

S1 - S0 * K2*K2_transposed - delta_1 * K1_transposed * K2 * K2_transposed + V2*K2_transposed =

S1 + (-S0 * K2 - delta_1 * K1_transposed * K2 + V2) * K2_transposed =

S1 + delta_2 * K2_transposed

delta_2 = (V2 - S0 * K2 - delta_1 * K1_transposed * K2)

S2 = S0 + delta_1 * K1_transposed + delta_2 * K2_transposed

S3 =
S2 + delta_3 * K3_transposed = 
(S0 + delta_1 * K1_transposed + delta_2 * K2_transposed) + (-S2 * K3 + V3) * K3_transposed = 

delta_3 = -S2 * K3 + V3 = 

-(S0 + delta_1 * K1_transposed + delta_2 * K2_transposed) * K3 + V3 =

S3 = 

(S0 + delta_1 * K1_transposed + delta_2 * K2_transposed) // S2
+
delta_3 * K3_transposed

// pattern:

S4 = 

(S0 + delta_1 * K1_transposed + delta_2 * K2_transposed + delta_3 * K3_transposed) // S3
+
delta_4 * K4_transposed

delta_4 = -S3 * K4 + V4 = 

-(S0 + delta_1 * K1_transposed + delta_2 * K2_transposed + delta_3 * K3_transposed) * K4 + V4


All the deltas have:

V1 - S0 * K1, V2 -S0 * K2, V3 - S0 * K3, V4 - S0 * K4

These can be called W:

W1 = V1 - S0 * K1
W2 = V2 - S0 * K2
W3 = V3 - S0 * K3
W4 = V4 - S0 * K4


Notice another common thing:

(K1_transposed + ... + K64_transposed) * K_token_index 

[64, headDim] * [headDim, 64] = [64, 64]

[1, headDim] * [headDim, 1] = [1, 1]

__global__ void run_state_update_loop_v2(
	float* output,
	float * keys_matmul,
	// float* state,
	const float* values,
	const float* keys,
	const float* queries,
	const float* alpha_gate,
	const float* beta_gate,
	const int headDim,
	const int dim,
	const int chunk_L,
	const int L
) {
	int tIndex = threadIdx.x;
	int numBlockThreads = blockDim.x;
	int numChunks = (L + chunk_L - 1) / chunk_L;	
	int headIndex = blockIdx.x;
	int headStartRowOffset = headIndex * headDim;

	for (int chunkIndex = 0; chunkIndex < numChunks; chunkIndex++) {
	 	int keys_chunk_col_offset = chunkIndex * chunk_L;
	 	int keys_chunk_offset = chunkIndex * chunk_L * dim;

	 	int local_chunk_L = keys_chunk_col_offset + chunk_L > L ? L - keys_chunk_col_offset : chunk_L;
	 	if (local_chunk_L == 0) {
	 		continue;
	 	}

		if (true) {
			// [chunk_L, headDim] * [headDim, chunk_L] = [chunk_L, chunk_L]
			int maxChunkSize = chunk_L * chunk_L;
		 	int headMatmulOutOffset = headIndex * maxChunkSize;

	 		int maxLocalChunkSize = local_chunk_L * local_chunk_L;
	 		int indices_per_thread = (maxLocalChunkSize + numBlockThreads - 1) / numBlockThreads;
			int keys_matmul_out_index = indices_per_thread * tIndex;

			int col_index = keys_matmul_out_index / local_chunk_L;
			int row_index = keys_matmul_out_index - col_index * local_chunk_L;

			int proj_offset = col_index * chunk_L + row_index;

			int keys_offset = keys_chunk_offset + col_index * dim + headStartRowOffset;
			int keys_transposed_offset = keys_chunk_offset + row_index * dim + headStartRowOffset;

			for (int i = 0; i < indices_per_thread; i++) {
				if (keys_matmul_out_index >= maxLocalChunkSize) {
					break; // break because the kernel has more work after this mathmul
				}

				float keys_dot_product = 0.0f;
				for (int j = 0; j < headDim; j++) {
					keys_dot_product += keys[keys_transposed_offset + j] * keys[keys_offset + j];
				}
				keys_matmul[headMatmulOutOffset + proj_offset] = keys_dot_product;

				keys_matmul_out_index++;
				
				col_index = keys_matmul_out_index / local_chunk_L;
				row_index = keys_matmul_out_index - col_index * local_chunk_L;
				proj_offset = col_index * chunk_L + row_index;

				keys_offset = keys_chunk_offset + col_index * dim + headStartRowOffset;
				keys_transposed_offset = keys_chunk_offset + row_index * dim + headStartRowOffset;
			}
		}

		if (chunkIndex > 0) {
			// state [headDim, headDim]
			// state [headDim, headDim] * keys [headDim, local_chunk_L] = [headDim, local_chunk_L]		
			int maxChunkSize = headDim * chunk_L;
			int maxLocalChunkSize = headDim * local_chunk_L;
			int indices_per_thread = (maxLocalChunkSize + numBlockThreads - 1) / numBlockThreads;
			int headMatmulOutOffset = headIndex * maxChunkSize; 			
			int state_keys_matmul_out_index = indices_per_thread * tIndex;

			for (int i = 0; i < indices_per_thread; i++) {
				if (state_keys_matmul_out_index >= maxLocalChunkSize) {
					break;
				}

				int col_index = state_keys_matmul_out_index / headDim;
				int row_index = state_keys_matmul_out_index - col_index * headDim;
				int proj_offset = col_index * headDim + row_index;

				int state_row_offset = headStartRowOffset + row_index;
				int keys_offset = keys_chunk_offset + col_index * dim + headStartRowOffset;

				float state_keys_dot_product = 0.0f;
				for (int j = 0; j < headDim; j++) {
					state_keys_dot_product += state[state_row_offset + j * dim] * keys[keys_offset + j];
				}
				state_keys_matmul[headMatmulOutOffset + proj_offset] = state_keys_dot_product;

				state_keys_matmul_out_index++;
			}
		}

		__syncthreads();
	}
}
