#ifndef GSPLAT_SPHERICAL_NASG_GABOR_CUH
#define GSPLAT_SPHERICAL_NASG_GABOR_CUH

#include "bindings.h"
#include "types.cuh"
#include "utils.cuh"

namespace gsplat {
    /** 
     * @brief The formulation of normalized anisotropic spherical gaussians with Gabor (NASG-Gabor):
     * 
     * C = c0 + ∑ᵢ cᵢ 
     * 
     * TOTAL PARAMETERS: 3 + 9N, where N is the number of primitives.
     * Each primitive has: cosθ, cosϕ, cosτ, λ, a, k, r, g, b
     * The Gabor term adds: (1 + cos(k * <v, x>)) / 2
     * 
     * @param num_primitives
     * @param num_colors
     * @param dir
     * @param coeffs
     * @param colors
     * 
     */
    template <typename T>
    __forceinline__ __device__ void nasg_gabor_fwd(
        const uint32_t num_primitives, // number of primitives to be evaluated
        const uint32_t num_colors,      // color channel
        const T* c0,           // [3] base color
        const T* primitives,   // [num_primitives, 9]
                               // Each primitive has: cosθ, cosϕ, cosτ, λ, a, k, r, g, b, normalization
        const vec3<T> &dir,    // [3]
        // output
        T *colors // [3]
    ){
        colors[0] = c0[0];
        colors[1] = c0[1];
        colors[2] = c0[2];

        // Normalize the direction vector
        T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        vec3<T> dir_norm = {
            dir.x * inorm,
            dir.y * inorm,
            dir.z * inorm
        };

        for (uint32_t i = 0; i < num_primitives; i++){  

            // Position
            T cosθ = primitives[i * 9 + 0];
            T cosϕ = primitives[i * 9 + 1];
            T cosτ = primitives[i * 9 + 2];

            // Shape
            T lambda = primitives[i * 9 + 3];
            T a   = primitives[i * 9 + 4];
            
            // Gabor frequency
            T k = primitives[i * 9 + 5];

            // Weight
            T r = primitives[i * 9 + 6];
            T g = primitives[i * 9 + 7];
            T b = primitives[i * 9 + 8];

            // parametrize basis direction
            cosθ = fminf(fmaxf(cosθ, -0.999999f), 0.999999f);
            cosϕ = fminf(fmaxf(cosϕ, -0.999999f), 0.999999f);
            cosτ = fminf(fmaxf(cosτ, -0.999999f), 0.999999f);

            T sinθ = sqrtf(1.0f - cosθ * cosθ);
            T sinϕ = sqrtf(1.0f - cosϕ * cosϕ);
            T sinτ = sqrtf(1.0f - cosτ * cosτ);

            vec3<T> x = {
                cosθ * cosϕ * cosτ - sinθ * sinτ, 
                sinθ * cosϕ * cosτ + cosθ * sinτ, 
                - sinϕ * cosτ
            };

            vec3<T> z = {
                cosθ * sinϕ,
                sinθ * sinϕ, 
                cosϕ
            };

            // nasg_gabor implementation
            const T eps = 5e-6f;
            const T eps_norm = 1e-8f;

            // Apply exponential activation and clamp
            a = __expf(a);
            lambda = __expf(lambda);
            a = fminf(a, 1e4f);
            lambda = fminf(lambda, 1e4f);
            
            // Activate k: k = (tanh(k_raw) + 1.0) * 20.0
            k = (tanhf(k) + 1.0f) * 20.0f;

            // Compute dot products
            T vz = dir_norm.x * z.x + dir_norm.y * z.y + dir_norm.z * z.z;
            T vx = dir_norm.x * x.x + dir_norm.y * x.y + dir_norm.z * x.z;

            // Handle edge cases
            bool mask_one = vz >= 1.0f - 1e-7f;
            bool mask_zero = vz <= -1.0f + 1e-7f;
            bool valid = !(mask_one || mask_zero);

            T pdf = 0.0f;

            if (valid) {
                // K_base = (vz + 1.0) * 0.5
                T K_base = (vz + 1.0f) * 0.5f;
                
                // K_exp = eps + (a * vx^2) / (1.0 - vz^2)
                T numerator = a * vx * vx;
                T denominator = 1.0f - vz * vz;
                T K_exp = eps + numerator / denominator;
                
                // exp = K_base^K_exp
                T exp_val = __powf(K_base, K_exp);
                
                // Compute normalization factor: inv_nasg_norm(λ, a)
                // inv_nasg_norm = (λ * sqrt(1.0 + a)) / (2π * (1.0 + eps - exp(-2λ)))
                T norm_num = lambda * sqrtf(1.0f + a);
                T norm_denom = TWO_PI * (1.0f + eps_norm - __expf(-2.0f * lambda));
                T inv_norm = norm_num / norm_denom;
                
                // Gabor term: (1 + cos(k * vx)) / 2
                T cosine_term = __cosf(k * vx);
                T gabor_term = (1.0f + cosine_term) * 0.5f;
                
                // pdf = exp(2λ * (exp*K_base - 1)) * exp * gabor_term * inv_norm
                T exponent = 2.0f * lambda * (exp_val * K_base - 1.0f);
                pdf = __expf(exponent) * exp_val * gabor_term * inv_norm;
            } else if (mask_one) {
                pdf = 1.0f;
            }
            // if mask_zero, pdf remains 0.0f

            // Accumulate color contribution
            colors[0] += pdf * r;
            colors[1] += pdf * g;
            colors[2] += pdf * b;
        }
    }

    template <typename T>
    __forceinline__ __device__ void nasg_gabor_eval_fwd(
        const uint32_t num_primitives,
        const uint32_t num_colors,
        const T* c0,
        const T* primitives,   // [num_primitives, 12] [x(3), z(3), λ, a, k, r, g, b]
        const vec3<T> &dir,
        T *colors
    ){
        colors[0] = c0[0];
        colors[1] = c0[1];
        colors[2] = c0[2];

        T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        vec3<T> dir_norm = {
            dir.x * inorm,
            dir.y * inorm,
            dir.z * inorm
        };

        for (uint32_t i = 0; i < num_primitives; i++){
            vec3<T> x = {
                primitives[i * 12 + 0],
                primitives[i * 12 + 1],
                primitives[i * 12 + 2]
            };

            vec3<T> z = {
                primitives[i * 12 + 3],
                primitives[i * 12 + 4],
                primitives[i * 12 + 5]
            };

            T lambda = primitives[i * 12 + 6];
            T a = primitives[i * 12 + 7];
            T k = primitives[i * 12 + 8];

            T r = primitives[i * 12 + 9];
            T g = primitives[i * 12 + 10];
            T b = primitives[i * 12 + 11];

            const T eps = 5e-6f;
            const T eps_norm = 1e-8f;

            T vz = dir_norm.x * z.x + dir_norm.y * z.y + dir_norm.z * z.z;
            T vx = dir_norm.x * x.x + dir_norm.y * x.y + dir_norm.z * x.z;

            bool mask_one = vz >= 1.0f - 1e-7f;
            bool mask_zero = vz <= -1.0f + 1e-7f;
            bool valid = !(mask_one || mask_zero);

            T pdf = 0.0f;

            if (valid) {
                T K_base = (vz + 1.0f) * 0.5f;
                T numerator = a * vx * vx;
                T denominator = 1.0f - vz * vz;
                T K_exp = eps + numerator / denominator;
                T exp_val = __powf(K_base, K_exp);

                T norm_num = lambda * sqrtf(1.0f + a);
                T norm_denom = TWO_PI * (1.0f + eps_norm - __expf(-2.0f * lambda));
                T inv_norm = norm_num / norm_denom;

                T cosine_term = __cosf(k * vx);
                T gabor_term = (1.0f + cosine_term) * 0.5f;

                T exponent = 2.0f * lambda * (exp_val * K_base - 1.0f);
                pdf = __expf(exponent) * exp_val * gabor_term * inv_norm;
            } else if (mask_one) {
                pdf = 1.0f;
            }

            colors[0] += pdf * r;
            colors[1] += pdf * g;
            colors[2] += pdf * b;
        }
    }

    template <typename T>
    __forceinline__ __device__ void nasg_gabor_bwd(
        const uint32_t num_primitives, // number of primitives to be evaluated
        const uint32_t num_colors,      // color channel

        const T* c0,           // [3] base color
        const T* primitives,   // [num_primitives, 9]
                               // Each primitive has: cosθ, cosϕ, cosτ, λ, a, k, r, g, b
        const vec3<T> &dir,    // [3]

        // Gradient input (w.r.t. output colors)
        const T* v_colors,     // [3] Gradients: [dL/dr, dL/dg, dL/db]

        // Gradient outputs (w.r.t. input parameters)
        T* v_c0,               // [3] Gradients for base color
        T* v_primitives        // [num_primitives, 9] Gradients for primitives
    )

    // Initialize gradients for base color c0
    {
        v_c0[0] = v_colors[0];
        v_c0[1] = v_colors[1];
        v_c0[2] = v_colors[2];

        // Normalize the direction vector
        T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        vec3<T> dir_norm = {
            dir.x * inorm,
            dir.y * inorm,
            dir.z * inorm
        };

        for (uint32_t i = 0; i < num_primitives; i++){  

            // Position
            T cosθ = primitives[i * 9 + 0];
            T cosϕ = primitives[i * 9 + 1];
            T cosτ = primitives[i * 9 + 2];

            // Shape
            T lambda = primitives[i * 9 + 3];
            T a   = primitives[i * 9 + 4];
            
            // Gabor frequency
            T k_raw = primitives[i * 9 + 5];
            
            // Weight
            T r = primitives[i * 9 + 6];
            T g = primitives[i * 9 + 7];
            T b = primitives[i * 9 + 8];

            // parametrize basis direction
            cosθ = fminf(fmaxf(cosθ, -0.999999f), 0.999999f);
            cosϕ = fminf(fmaxf(cosϕ, -0.999999f), 0.999999f);
            cosτ = fminf(fmaxf(cosτ, -0.999999f), 0.999999f);

            T sinθ = sqrtf(1.0f - cosθ * cosθ);
            T sinϕ = sqrtf(1.0f - cosϕ * cosϕ);
            T sinτ = sqrtf(1.0f - cosτ * cosτ);

            vec3<T> x = {
                cosθ * cosϕ * cosτ - sinθ * sinτ, 
                sinθ * cosϕ * cosτ + cosθ * sinτ, 
                - sinϕ * cosτ
            };

            vec3<T> z = {
                cosθ * sinϕ,
                sinθ * sinϕ, 
                cosϕ
            };

            // nasg_gabor backward implementation
            const T eps = 5e-6f;
            const T eps_norm = 1e-8f;

            // Apply exponential activation and clamp
            a = __expf(a);
            lambda = __expf(lambda);
            
            // Store whether values were clamped (before clamping)
            bool a_clamped = (a >= 1e4f);
            bool lambda_clamped = (lambda >= 1e4f);
            
            a = fminf(a, 1e4f);
            lambda = fminf(lambda, 1e4f);
            
            // Activate k: k = (tanh(k_raw) + 1.0) * 20.0
            T tanh_k = tanhf(k_raw);
            T k = (tanh_k + 1.0f) * 20.0f;

            // Compute dot products
            T vz = dir_norm.x * z.x + dir_norm.y * z.y + dir_norm.z * z.z;
            T vx = dir_norm.x * x.x + dir_norm.y * x.y + dir_norm.z * x.z;

            // Handle edge cases
            bool mask_one = vz >= 1.0f - 1e-7f;
            bool mask_zero = vz <= -1.0f + 1e-7f;
            bool valid = !(mask_one || mask_zero);

            T pdf = 0.0f;
            T K_base = 0.0f;
            T K_exp = 0.0f;
            T E = 0.0f;
            T p = 0.0f;
            T C = 0.0f;
            T gabor_term = 0.0f;
            T cosine_term = 0.0f;

            if (valid) {
                // Recompute forward pass values
                K_base = (vz + 1.0f) * 0.5f;
                T numerator = a * vx * vx;
                T denominator = 1.0f - vz * vz;
                K_exp = eps + numerator / denominator;
                E = __powf(K_base, K_exp);
                
                // Normalization C = λ√(1+a) / [2π(1 + eps - e^(-2λ))]
                T sqrt_1_plus_a = sqrtf(1.0f + a);
                T exp_neg_2lam = __expf(-2.0f * lambda);
                T norm_denom = TWO_PI * (1.0f + eps_norm - exp_neg_2lam);
                C = (lambda * sqrt_1_plus_a) / norm_denom;
                
                // Gabor term
                cosine_term = __cosf(k * vx);
                gabor_term = (1.0f + cosine_term) * 0.5f;
                
                // p = exp(2λ(E*K_base - 1))
                T exponent = 2.0f * lambda * (E * K_base - 1.0f);
                p = __expf(exponent);
                
                pdf = p * E * gabor_term * C;
            } else if (mask_one) {
                pdf = 1.0f;
            }

            // Compute gradients w.r.t. color contributions (r, g, b)
            v_primitives[i * 9 + 6] = v_colors[0] * pdf; // grad_r
            v_primitives[i * 9 + 7] = v_colors[1] * pdf; // grad_g
            v_primitives[i * 9 + 8] = v_colors[2] * pdf; // grad_b

            T grad_a = 0.0f;
            T grad_lambda = 0.0f;
            T grad_k = 0.0f;
            
            vec3<T> grad_x = {0.0f, 0.0f, 0.0f};
            vec3<T> grad_z = {0.0f, 0.0f, 0.0f};
            
            T grad_cosθ = 0.0f;
            T grad_cosϕ = 0.0f;
            T grad_cosτ = 0.0f;

            if (valid) {
                // Gradient of loss w.r.t pdf
                T grad_pdf = v_colors[0] * r + v_colors[1] * g + v_colors[2] * b;
                
                // pdf = p * E * gabor_term * C
                // Using product rule for four terms
                
                // ========================================
                // Derivative w.r.t. k (Gabor frequency)
                // ========================================
                // ∂gabor_term/∂k = -0.5 * sin(k*vx) * vx
                T dgabor_dk = -0.5f * __sinf(k * vx) * vx;
                
                // ∂pdf/∂k = p * E * C * ∂gabor_term/∂k
                T dpdf_dk = p * E * C * dgabor_dk;
                
                // Chain rule for k activation: k = (tanh(k_raw) + 1) * 20
                // dk/dk_raw = 20 * (1 - tanh²(k_raw))
                T dk_dkraw = 20.0f * (1.0f - tanh_k * tanh_k);
                grad_k = grad_pdf * dpdf_dk * dk_dkraw;
                
                // ========================================
                // Derivative w.r.t. lambda (with chain rule for exp activation)
                // ========================================
                // ∂p/∂λ = p * 2(E*K_base - 1)
                T dp_dlam = p * 2.0f * (E * K_base - 1.0f);
                
                // ∂C/∂λ using quotient rule:
                T sqrt_1_plus_a = sqrtf(1.0f + a);
                T exp_neg_2lam = __expf(-2.0f * lambda);
                T u = lambda * sqrt_1_plus_a;
                T v = TWO_PI * (1.0f + eps_norm - exp_neg_2lam);
                T u_prime = sqrt_1_plus_a;
                T v_prime = 2.0f * TWO_PI * exp_neg_2lam;
                T dC_dlam = (u_prime * v - u * v_prime) / (v * v);
                
                // Product rule: pdf = p * E * gabor_term * C
                // ∂pdf/∂λ = (∂p/∂λ)*E*gabor_term*C + p*E*gabor_term*(∂C/∂λ)
                T dpdf_dlam_activated = E * gabor_term * dp_dlam * C + p * E * gabor_term * dC_dlam;
                
                // Chain rule for exp(lambda_raw)
                grad_lambda = grad_pdf * dpdf_dlam_activated * lambda;
                
                // Zero out gradient if value was clamped
                if (lambda_clamped) grad_lambda = 0.0f;
                
                // ========================================
                // Derivative w.r.t. a (with chain rule for exp activation)
                // ========================================
                // ∂K_exp/∂a = vx² / (1 - vz²)
                T dK_exp_da = (vx * vx) / (1.0f - vz * vz);
                
                // ∂E/∂a = E * ln(K_base) * ∂K_exp/∂a
                T dE_da = E * __logf(K_base) * dK_exp_da;
                
                // ∂p/∂a = p * 2λ * K_base * ∂E/∂a
                T dp_da = p * 2.0f * lambda * K_base * dE_da;
                
                // ∂C/∂a
                T dC_da = (lambda / (2.0f * sqrt_1_plus_a)) / v;
                
                // Product rule
                T dpdf_da_activated = dp_da * gabor_term * E * C + p * gabor_term * dE_da * C + p * E * gabor_term * dC_da;
                
                // Chain rule for exp(a_raw)
                grad_a = grad_pdf * dpdf_da_activated * a;
                
                // Zero out gradient if value was clamped
                if (a_clamped) grad_a = 0.0f;
                
                // ========================================
                // Derivative w.r.t. x (basis vector)
                // ========================================
                // x appears in both K_exp and gabor_term
                
                // From K_exp:
                T K_exp_dx_coeff = (2.0f * a * vx) / (1.0f - vz * vz);
                vec3<T> K_exp_dx = {
                    K_exp_dx_coeff * dir_norm.x,
                    K_exp_dx_coeff * dir_norm.y,
                    K_exp_dx_coeff * dir_norm.z
                };
                
                T dE_dx_coeff = E * __logf(K_base);
                vec3<T> dE_dx = {
                    dE_dx_coeff * K_exp_dx.x,
                    dE_dx_coeff * K_exp_dx.y,
                    dE_dx_coeff * K_exp_dx.z
                };
                
                T dp_dx_coeff = p * 2.0f * lambda * K_base;
                vec3<T> dp_dx = {
                    dp_dx_coeff * dE_dx.x,
                    dp_dx_coeff * dE_dx.y,
                    dp_dx_coeff * dE_dx.z
                };
                
                // From gabor_term: ∂gabor_term/∂x = -0.5 * sin(k*vx) * k * v
                T dgabor_dx_coeff = -0.5f * __sinf(k * vx) * k;
                vec3<T> dgabor_dx = {
                    dgabor_dx_coeff * dir_norm.x,
                    dgabor_dx_coeff * dir_norm.y,
                    dgabor_dx_coeff * dir_norm.z
                };
                
                // Product rule: pdf = p * E * gabor_term * C
                // ∂pdf/∂x = (∂p/∂x)*E*gabor_term*C + p*(∂E/∂x)*gabor_term*C + p*E*(∂gabor_term/∂x)*C
                grad_x.x = grad_pdf * (dp_dx.x * E * gabor_term * C + p * dE_dx.x * gabor_term * C + p * E * dgabor_dx.x * C);
                grad_x.y = grad_pdf * (dp_dx.y * E * gabor_term * C + p * dE_dx.y * gabor_term * C + p * E * dgabor_dx.y * C);
                grad_x.z = grad_pdf * (dp_dx.z * E * gabor_term * C + p * dE_dx.z * gabor_term * C + p * E * dgabor_dx.z * C);
                
                // ========================================
                // Derivative w.r.t. z (basis vector)
                // ========================================
                vec3<T> K_base_dz = {
                    0.5f * dir_norm.x,
                    0.5f * dir_norm.y,
                    0.5f * dir_norm.z
                };
                
                T K_exp_dz_coeff = a * vx * vx * (2.0f * vz) / ((1.0f - vz * vz) * (1.0f - vz * vz));
                vec3<T> K_exp_dz = {
                    K_exp_dz_coeff * dir_norm.x,
                    K_exp_dz_coeff * dir_norm.y,
                    K_exp_dz_coeff * dir_norm.z
                };
                
                T dE_dz_coeff1 = E * (K_exp / K_base);
                T dE_dz_coeff2 = E * __logf(K_base);
                vec3<T> dE_dz = {
                    dE_dz_coeff1 * K_base_dz.x + dE_dz_coeff2 * K_exp_dz.x,
                    dE_dz_coeff1 * K_base_dz.y + dE_dz_coeff2 * K_exp_dz.y,
                    dE_dz_coeff1 * K_base_dz.z + dE_dz_coeff2 * K_exp_dz.z
                };
                
                T dp_dz_coeff = p * 2.0f * lambda;
                vec3<T> dp_dz = {
                    dp_dz_coeff * (K_base * dE_dz.x + E * K_base_dz.x),
                    dp_dz_coeff * (K_base * dE_dz.y + E * K_base_dz.y),
                    dp_dz_coeff * (K_base * dE_dz.z + E * K_base_dz.z)
                };
                
                // Product rule
                grad_z.x = grad_pdf * (dp_dz.x * E * gabor_term * C + p * dE_dz.x * gabor_term * C);
                grad_z.y = grad_pdf * (dp_dz.y * E * gabor_term * C + p * dE_dz.y * gabor_term * C);
                grad_z.z = grad_pdf * (dp_dz.z * E * gabor_term * C + p * dE_dz.z * gabor_term * C);
                
                // ========================================
                // Transform grad_x and grad_z to gradients w.r.t. cosθ, cosϕ, cosτ
                // ========================================
                T dsinθ_dcosθ = -cosθ / sinθ;
                T dsinϕ_dcosϕ = -cosϕ / sinϕ;
                T dsinτ_dcosτ = -cosτ / sinτ;
                
                // ∂x/∂cosθ
                T dx0_dcosθ = cosϕ * cosτ - dsinθ_dcosθ * sinτ;
                T dx1_dcosθ = dsinθ_dcosθ * cosϕ * cosτ + sinτ;
                T dx2_dcosθ = 0.0f;
                
                // ∂z/∂cosθ
                T dz0_dcosθ = sinϕ;
                T dz1_dcosθ = dsinθ_dcosθ * sinϕ;
                T dz2_dcosθ = 0.0f;
                
                grad_cosθ = grad_x.x * dx0_dcosθ + grad_x.y * dx1_dcosθ + grad_x.z * dx2_dcosθ +
                            grad_z.x * dz0_dcosθ + grad_z.y * dz1_dcosθ + grad_z.z * dz2_dcosθ;
                
                // ∂x/∂cosϕ
                T dx0_dcosϕ = cosθ * cosτ;
                T dx1_dcosϕ = sinθ * cosτ;
                T dx2_dcosϕ = -dsinϕ_dcosϕ * cosτ;
                
                // ∂z/∂cosϕ
                T dz0_dcosϕ = cosθ * dsinϕ_dcosϕ;
                T dz1_dcosϕ = sinθ * dsinϕ_dcosϕ;
                T dz2_dcosϕ = 1.0f;
                
                grad_cosϕ = grad_x.x * dx0_dcosϕ + grad_x.y * dx1_dcosϕ + grad_x.z * dx2_dcosϕ +
                            grad_z.x * dz0_dcosϕ + grad_z.y * dz1_dcosϕ + grad_z.z * dz2_dcosϕ;
                
                // ∂x/∂cosτ
                T dx0_dcosτ = cosθ * cosϕ - sinθ * dsinτ_dcosτ;
                T dx1_dcosτ = sinθ * cosϕ + cosθ * dsinτ_dcosτ;
                T dx2_dcosτ = -sinϕ;
                
                // ∂z/∂cosτ (z doesn't depend on τ)
                T dz0_dcosτ = 0.0f;
                T dz1_dcosτ = 0.0f;
                T dz2_dcosτ = 0.0f;
                
                grad_cosτ = grad_x.x * dx0_dcosτ + grad_x.y * dx1_dcosτ + grad_x.z * dx2_dcosτ +
                            grad_z.x * dz0_dcosτ + grad_z.y * dz1_dcosτ + grad_z.z * dz2_dcosτ;
            }

            // Store gradients
            v_primitives[i * 9 + 0] = grad_cosθ;
            v_primitives[i * 9 + 1] = grad_cosϕ;
            v_primitives[i * 9 + 2] = grad_cosτ;
            v_primitives[i * 9 + 3] = grad_lambda;
            v_primitives[i * 9 + 4] = grad_a;
            v_primitives[i * 9 + 5] = grad_k;
        }
    }

} // namespace gsplat
#endif // GSPLAT_SPHERICAL_NASG_CUH