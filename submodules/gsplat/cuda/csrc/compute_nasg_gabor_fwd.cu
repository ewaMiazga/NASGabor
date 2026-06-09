#include "bindings.h"
#include "spherical_nasg_gabor.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

    template <typename T>
    __global__ void compute_nasg_gabor_fwd_kernel(
        const uint32_t N,
        const uint32_t num_primitives,
        const uint32_t active_primitives,
        const vec3<T> *__restrict__ dirs, // [N, 3]
        const T *__restrict__ c0,         // [3] base color
        const T *__restrict__ coeffs,     // [N, max_num_primitives * 9] [cosθ, cosϕ, cosτ, λ, a, k, r, g, b]
        const bool *__restrict__ masks,   // [N]
        T *__restrict__ colors            // [N, 3]
    ) {
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }
        if (masks != nullptr && !masks[idx]) {
            return;
        }
        nasg_gabor_fwd(
            active_primitives,
            3,
            c0 + idx * 3,
            coeffs + idx * num_primitives * 9,
            dirs[idx],
            colors + idx * 3
        );
    }

    template <typename T>
    __global__ void compute_nasg_gabor_eval_fwd_kernel(
        const uint32_t N,
        const uint32_t num_primitives,
        const uint32_t active_primitives,
        const vec3<T> *__restrict__ dirs, // [N, 3]
        const T *__restrict__ c0,         // [3] base color
        const T *__restrict__ coeffs,     // [N, max_num_primitives * 12] [x(3), z(3), λ, a, k, r, g, b]
        const bool *__restrict__ masks,   // [N]
        T *__restrict__ colors            // [N, 3]
    ) {
        // parallelize over N
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }
        if (masks != nullptr && !masks[idx]) {
            return;
        }
        nasg_gabor_eval_fwd(
            active_primitives,
            3,
            c0 + idx * 3,
            coeffs + idx * num_primitives * 12,
            dirs[idx],
            colors + idx * 3
        );
    }

    torch::Tensor compute_nasg_gabor_fwd_tensor(
            const uint32_t active_primitives,
            const torch::Tensor &dirs,              // [..., 3]
            const torch::Tensor &c0,                // [3] base color
            const torch::Tensor &coeffs,            // [..., max_num_primitives * 9]
            const at::optional<torch::Tensor> masks // [...]
        ) {
            GSPLAT_DEVICE_GUARD(dirs);
            GSPLAT_CHECK_INPUT(dirs);
            GSPLAT_CHECK_INPUT(c0);
            GSPLAT_CHECK_INPUT(coeffs);
            if (masks.has_value()) {
                GSPLAT_CHECK_INPUT(masks.value());
            }
            TORCH_CHECK(coeffs.size(-1) % 9 == 0, "coeffs must have last dimension 9 multiple");
            TORCH_CHECK(coeffs.size(-1) / 9 >= active_primitives, "max_primitives must be more than or equal to active_primitives");
            TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
            TORCH_CHECK(c0.size(-1) == 3, "c0 must have last dimension 3");
            const uint32_t N = dirs.numel() / 3;
            const uint32_t max_num_primitives = coeffs.size(-1) / 9;
            torch::Tensor colors = torch::empty_like(dirs);

            if (N) {
                compute_nasg_gabor_fwd_kernel<float>
                <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS>>>(
                    N,
                    max_num_primitives,
                    active_primitives,
                    reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
                    c0.data_ptr<float>(),
                    coeffs.data_ptr<float>(),
                    masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                    colors.data_ptr<float>()
                );
            }

            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error in compute_nasg_gabor_fwd_tensor: " << cudaGetErrorString(error) << std::endl;
                throw std::runtime_error("CUDA error in compute_nasg_gabor_fwd_tensor " + std::string(cudaGetErrorString(error)));
            }

            return colors;
    }

    torch::Tensor compute_nasg_gabor_eval_fwd_tensor(
            const uint32_t active_primitives,
            const torch::Tensor &dirs,              // [..., 3]
            const torch::Tensor &c0,                // [3] base color
            const torch::Tensor &coeffs,            // [..., max_num_primitives * 12]
            const at::optional<torch::Tensor> masks // [...]
        ) {
            GSPLAT_DEVICE_GUARD(dirs);  // ensures that tensor is on correct device
            GSPLAT_CHECK_INPUT(dirs); // ensures that tensor is on GPU, contiguous, and of correct type (usually float32)
            GSPLAT_CHECK_INPUT(c0);
            GSPLAT_CHECK_INPUT(coeffs);
            if (masks.has_value()) {
                GSPLAT_CHECK_INPUT(masks.value());
            }
            TORCH_CHECK(coeffs.size(-1) % 12 == 0, "coeffs must have last dimension 12 multiple");
            TORCH_CHECK(coeffs.size(-1) / 12 >= active_primitives, "max_primitives must be more than or equal to active_primitives");
            TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
            TORCH_CHECK(c0.size(-1) == 3, "c0 must have last dimension 3");
            const uint32_t N = dirs.numel() / 3;
            const uint32_t max_num_primitives = coeffs.size(-1) / 12;
            torch::Tensor colors = torch::empty_like(dirs); // [..., 3]

        // parallelize over N
        if (N) {
            compute_nasg_gabor_eval_fwd_kernel<float>
            <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS>>>(
                N,
                max_num_primitives,
                active_primitives,
                reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
                c0.data_ptr<float>(),
                coeffs.data_ptr<float>(),
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                colors.data_ptr<float>()
            );

        }
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in compute_nasg_gabor_eval_fwd_tensor: " << cudaGetErrorString(error) << std::endl;
            throw std::runtime_error("CUDA error in compute_nasg_gabor_eval_fwd_tensor " + std::string(cudaGetErrorString(error)));
        }

        return colors; // [..., 3]
    }

} // namespace gsplat
