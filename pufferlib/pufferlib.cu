#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace pufferlib {

/*
__global__ void p3o_kernel(
    float* reward_block,    // [num_steps, horizon]
    float* reward_mask,     // [num_steps, horizon]
    float* values_mean,     // [num_steps, horizon]
    float* values_std,      // [num_steps, horizon]
    float* buf,            // [num_steps, horizon]
    float* dones,          // [num_steps]
    float* rewards,        // [num_steps]
    float* advantages,     // [num_steps]
    int* bounds,          // [num_steps]
    int num_steps,
    float r_std,
    float puf,
    int horizon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_steps) return;

    int k = 0;
    for (int j = 0; j < horizon-1; j++) {
        int t = i + j;
        if (t >= num_steps - 1) {
            break;
        }
        if (dones[t+1]) {
            k++;
            break;
        }
        k++;
    }

    float gamma_max = 0.0f;
    float n = 0.0f;
    for (int j = k-1; j >= 0; j--) {
        int idx = i * horizon + j;
        n++;

        float vstd = values_std[idx];
        if (vstd == 0.0f) {
            buf[idx] = 0.0f;
            continue;
        }

        float gamma = 1.0f / (vstd*vstd);
        if (r_std != 0.0f) {
            gamma -= puf/(r_std*r_std);
        }

        if (gamma < 0.0f) {
            gamma = 0.0f;
        }

        if (gamma > gamma_max) {
            gamma_max = gamma;
        }
        buf[idx] = gamma;
        reward_mask[idx] = 1.0f;
    }

    //float bootstrap = 0.0f;
    //if (k == horizon-1) {
    //    bootstrap = buf[i*horizon + horizon - 1]*values_mean[i*horizon + horizon - 1];
    //}

    float R = 0.0f;
    for (int j = 0; j <= k-1; j++) {
        int t = i + j;
        int idx = i * horizon + j;
        float r = rewards[t+1];

        float gamma = buf[idx];
        if (gamma_max > 0) {
            gamma /= gamma_max;
        }

        if (j >= 16 && values_std[idx] > 0.95*r_std) {
            break;
        }

        R += gamma * (r - values_mean[idx]);
        reward_block[idx] = r;
        buf[idx] = gamma;
    }

    advantages[i] = R;
    bounds[i] = k;
}


void compute_p3o(torch::Tensor reward_block, torch::Tensor reward_mask,
        torch::Tensor values_mean, torch::Tensor values_std, torch::Tensor buf,
        torch::Tensor dones, torch::Tensor rewards, torch::Tensor advantages,
        torch::Tensor bounds, int num_steps, float vstd_max, float puf,
        int horizon) {

    // TODO: Port from python
    assert all(t.is_cuda for t in [reward_block, reward_mask, values_mean, values_std, 
                                  buf, dones, rewards, advantages, bounds]), "All tensors must be on GPU"
    
    # Ensure contiguous memory
    tensors = [reward_block, reward_mask, values_mean, values_std, buf, dones, rewards, advantages, bounds]
    for t in tensors:
        t.contiguous()
        assert t.is_cuda

    num_steps = rewards.shape[0]
    
    # Precompute vstd_min and vstd_max
    #vstd_max = values_std.max().item()
    #vstd_min = values_std.min().item()

    # Launch kernel
    threads_per_block = 256
    assert num_steps % threads_per_block == 0
    blocks = (num_steps + threads_per_block - 1) // threads_per_block
 
    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    p3o_kernel<<<blocks, threads_per_block>>>(
        reward_block.data_ptr<float>(),
        reward_mask.data_ptr<float>(),
        values_mean.data_ptr<float>(),
        values_std.data_ptr<float>(),
        buf.data_ptr<float>(),
        dones.data_ptr<float>(),
        rewards.data_ptr<float>(),
        advantages.data_ptr<float>(),
        bounds.data_ptr<int>(),
        num_steps,
        vstd_max, 
        puf,
        horizon
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return;
}

// [num_steps, horizon]
__global__ void gae_kernel(float* values, float* rewards, float* dones,
        float* advantages, float gamma, float gae_lambda, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int offset = row*horizon;
    gae_row(values + offset, rewards + offset, dones + offset,
        advantages + offset, gamma, gae_lambda, horizon);
}

torch::Tensor compute_gae(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, float gamma, float gae_lambda) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    torch::Tensor advantages = gae_check(values, rewards, dones, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;
    assert(num_steps % threads_per_block == 0);

    gae_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma,
        gae_lambda,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return advantages;
}

 // [num_steps, horizon]
__global__ void vtrace_kernel(float* values, float* rewards, float* dones, float* importance,
        float* vs, float* advantages, float gamma, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int offset = row*horizon;
    vtrace_row(values + offset, rewards + offset, dones + offset,
        importance + offset, vs + offset, advantages + offset, gamma, rho_clip, c_clip, horizon);
}

void compute_vtrace(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor vs, torch::Tensor advantages,
        float gamma, float rho_clip, float c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check(values, rewards, dones, importance, vs, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");
    assert(horizon <= max_horizon);

    int threads_per_block = 128;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;
    assert(num_steps % threads_per_block == 0);

    vtrace_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        vs.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma,
        rho_clip,
        c_clip,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
*/

static const int max_horizon = 256;
__host__ __device__ void puff_advantage_row_cuda(float* values, float* rewards, float* dones,
        float* importance, float* vs, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    vs[horizon-1] = values[horizon-1];
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        // TODO: t_next works and t doesn't. Check original formula
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        
        //float delta = rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t];
        //lastpufferlam = delta + gamma*lambda*lastpufferlam*nextnonterminal;


        advantages[t] = lastpufferlam;
        vs[t] = advantages[t] + values[t];
        //advantages[t] = rho_t*(rewards[t] + gamma*vs[t_next]*nextnonterminal - values[t]);
        //vs[t] = lastpufferlam + values[t];
    }
}

void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor vs, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, vs, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        assert(horizon <= max_horizon);
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}


 // [num_steps, horizon]
__global__ void puff_advantage_kernel(float* values, float* rewards, float* dones, float* importance,
        float* vs, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int offset = row*horizon;
    puff_advantage_row_cuda(values + offset, rewards + offset, dones + offset,
        importance + offset, vs + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor vs, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, vs, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");
    assert(horizon <= max_horizon);

    int threads_per_block = 256;
    if (threads_per_block > num_steps) {
        threads_per_block = 2*(num_steps/2);
    }
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;
    assert(num_steps % threads_per_block == 0);

    puff_advantage_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        vs.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma,
        lambda,
        rho_clip,
        c_clip,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

}
