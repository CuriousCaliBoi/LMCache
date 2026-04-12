// SPDX-License-Identifier: Apache-2.0
//
// TokenDance — Fused Sparse Restore CUDA Kernels
//
// Two kernels for applying block-sparse KV diffs during the layerwise
// transfer pipeline:
//
//   1. apply_kv_diff_single  — corrects a single KV plane (K or V).
//   2. apply_kv_diff_paired  — corrects K and V together when they
//      share the same block-index list.
//
// Reference: TokenDance paper, Section 4.4 — Fused Diff Restore.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tokendance {

// ─── Single-plane diff kernel ──────────────────────────────────────────

template <typename scalar_t>
__global__ void apply_kv_diff_single_kernel(
    scalar_t* __restrict__ buffer,       // (num_tokens, hidden_dim)
    const int* __restrict__ block_indices,  // (num_diff_blocks,)
    const scalar_t* __restrict__ corrections, // (num_diff_blocks, block_size, hidden_dim)
    int num_tokens,
    int hidden_dim,
    int block_size,
    int num_diff_blocks) {
  // Each thread handles one element within one diff block.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_diff_blocks * block_size * hidden_dim;

  if (tid >= total_elements) return;

  int blk_local = tid / (block_size * hidden_dim);
  int remainder = tid % (block_size * hidden_dim);
  int tok_in_blk = remainder / hidden_dim;
  int h = remainder % hidden_dim;

  int blk_idx = block_indices[blk_local];
  int tok_global = blk_idx * block_size + tok_in_blk;

  if (tok_global >= num_tokens) return;

  int buf_offset = tok_global * hidden_dim + h;
  int cor_offset = blk_local * block_size * hidden_dim + tok_in_blk * hidden_dim + h;

  buffer[buf_offset] += corrections[cor_offset];
}

void apply_kv_diff_single(
    torch::Tensor buffer,
    torch::Tensor block_indices,
    torch::Tensor corrections,
    int block_size) {
  int num_tokens = buffer.size(0);
  int hidden_dim = buffer.size(1);
  int num_diff_blocks = block_indices.size(0);

  int total = num_diff_blocks * block_size * hidden_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      buffer.scalar_type(), "apply_kv_diff_single", [&] {
        apply_kv_diff_single_kernel<scalar_t><<<blocks, threads>>>(
            buffer.data_ptr<scalar_t>(),
            block_indices.data_ptr<int>(),
            corrections.data_ptr<scalar_t>(),
            num_tokens, hidden_dim, block_size, num_diff_blocks);
      });
}

// ─── Paired K+V diff kernel ───────────────────────────────────────────

template <typename scalar_t>
__global__ void apply_kv_diff_paired_kernel(
    scalar_t* __restrict__ k_buffer,
    scalar_t* __restrict__ v_buffer,
    const int* __restrict__ block_indices,
    const scalar_t* __restrict__ k_corrections,
    const scalar_t* __restrict__ v_corrections,
    int num_tokens,
    int hidden_dim,
    int block_size,
    int num_diff_blocks) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_diff_blocks * block_size * hidden_dim;

  if (tid >= total_elements) return;

  int blk_local = tid / (block_size * hidden_dim);
  int remainder = tid % (block_size * hidden_dim);
  int tok_in_blk = remainder / hidden_dim;
  int h = remainder % hidden_dim;

  int blk_idx = block_indices[blk_local];
  int tok_global = blk_idx * block_size + tok_in_blk;

  if (tok_global >= num_tokens) return;

  int buf_offset = tok_global * hidden_dim + h;
  int cor_offset = blk_local * block_size * hidden_dim + tok_in_blk * hidden_dim + h;

  k_buffer[buf_offset] += k_corrections[cor_offset];
  v_buffer[buf_offset] += v_corrections[cor_offset];
}

void apply_kv_diff_paired(
    torch::Tensor k_buffer,
    torch::Tensor v_buffer,
    torch::Tensor block_indices,
    torch::Tensor k_corrections,
    torch::Tensor v_corrections,
    int block_size) {
  int num_tokens = k_buffer.size(0);
  int hidden_dim = k_buffer.size(1);
  int num_diff_blocks = block_indices.size(0);

  int total = num_diff_blocks * block_size * hidden_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      k_buffer.scalar_type(), "apply_kv_diff_paired", [&] {
        apply_kv_diff_paired_kernel<scalar_t><<<blocks, threads>>>(
            k_buffer.data_ptr<scalar_t>(),
            v_buffer.data_ptr<scalar_t>(),
            block_indices.data_ptr<int>(),
            k_corrections.data_ptr<scalar_t>(),
            v_corrections.data_ptr<scalar_t>(),
            num_tokens, hidden_dim, block_size, num_diff_blocks);
      });
}

}  // namespace tokendance

// ─── PyBind ───────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "TokenDance fused sparse restore CUDA kernels";
  m.def("apply_kv_diff_single", &tokendance::apply_kv_diff_single,
        "Apply block-sparse corrections to a single KV plane (CUDA)");
  m.def("apply_kv_diff_paired", &tokendance::apply_kv_diff_paired,
        "Apply block-sparse corrections to K and V simultaneously (CUDA)");
}
