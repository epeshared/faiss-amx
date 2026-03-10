#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace scalar_quantizer {

// Computes inner products between BF16-coded vectors.
//
// - A is a matrix with shape (a_rows, d) in row-major layout.
// - b is a single BF16 vector of length d.
// - out receives a_rows FP32 dot products.
//
// Returns 0 on success (AMX path executed). Returns non-zero to indicate
// AMX is unavailable for the current build or thread and the caller should
// fall back to another implementation.
int amx_bf16_ip_a_rows(
        const uint16_t* a,
        const uint16_t* b,
        size_t d,
        size_t a_rows,
        float* out);

} // namespace scalar_quantizer
} // namespace faiss
