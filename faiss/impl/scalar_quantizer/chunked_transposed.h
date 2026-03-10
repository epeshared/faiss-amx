/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <faiss/Index.h>
#include <vector>
#include <cstring>

#include <faiss/utils/bf16.h>

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#endif

namespace faiss {
namespace scalar_quantizer {

/**
 * Chunked Transposed layout for BF16 Scalar Quantizer.
 *
 * Memory layout:
 * - Vectors are grouped into chunks of size `chunk_size` (e.g., 64 vectors)
 * - Within each chunk, data is stored in dimension-major order:
 *   [chunk0_dim0_all_vecs, chunk0_dim1_all_vecs, ..., chunk0_dimD_all_vecs,
 *    chunk1_dim0_all_vecs, ...]
 *
 * This layout enables efficient AMX DPBF16PS operations for batch inner product
 * computation, as a full chunk can be loaded into tiles and multiplied with
 * a query vector in one operation.
 */
struct ChunkedTransposedBF16 {
    // Configuration
    size_t d;              // Dimensionality
    size_t chunk_size;     // Number of vectors per chunk (e.g., 64)
    size_t nchunks;        // Number of chunks
    size_t ntotal;         // Total number of vectors

    // Data storage: [nchunks][d][chunk_size] of uint16_t (BF16)
    // Flattened as: uint16_t[nchunks * d * chunk_size]
    std::vector<uint16_t> data;

    // Last chunk may have fewer vectors
    size_t last_chunk_size;  // Actual vector count in the last chunk

    ChunkedTransposedBF16()
        : d(0), chunk_size(64), nchunks(0), ntotal(0), last_chunk_size(0) {}

    ChunkedTransposedBF16(size_t d, size_t chunk_size = 64)
        : d(d), chunk_size(chunk_size), nchunks(0), ntotal(0), last_chunk_size(0) {
        // data will be resized as vectors are added
    }

    // Reserve space for ntotal_max vectors
    void reserve(size_t ntotal_max) {
        nchunks = (ntotal_max + chunk_size - 1) / chunk_size;
        data.resize(nchunks * d * chunk_size);
    }

    // Encode and add n vectors to the storage
    void add_vectors(const float* x, size_t n) {
        if (n == 0) return;

        size_t old_ntotal = ntotal;
        ntotal += n;

        // Recalculate chunks if needed
        size_t new_nchunks = (ntotal + chunk_size - 1) / chunk_size;
        if (new_nchunks > nchunks) {
            data.resize(new_nchunks * d * chunk_size);
            nchunks = new_nchunks;
        }

        // Encode vectors to BF16 and store in transposed layout
        for (size_t i = 0; i < n; i++) {
            size_t vec_id = old_ntotal + i;
            size_t chunk_id = vec_id / chunk_size;
            size_t chunk_offset = vec_id % chunk_size;

            for (size_t dim = 0; dim < d; dim++) {
                size_t data_idx = (chunk_id * d + dim) * chunk_size + chunk_offset;
                data[data_idx] = encode_bf16(x[i * d + dim]);
            }
        }

        last_chunk_size = (ntotal - 1) % chunk_size + 1;
    }

    // Get a pointer to the BF16 codes for a specific vector (for compatibility)
    // Note: This is inefficient with transposed layout - avoid in hot paths
    const uint16_t* get_vector_codes(size_t vec_id) const {
        // Returns a thread-local buffer with reconstructed row-major codes
        // This is a fallback for compatibility
        static thread_local std::vector<uint16_t> tmp;
        if (tmp.size() < d) tmp.resize(d);

        size_t chunk_id = vec_id / chunk_size;
        size_t chunk_offset = vec_id % chunk_size;
        size_t actual_chunk_size = (chunk_id == nchunks - 1) ? last_chunk_size : chunk_size;

        if (chunk_offset < actual_chunk_size) {
            for (size_t dim = 0; dim < d; dim++) {
                size_t data_idx = (chunk_id * d + dim) * chunk_size + chunk_offset;
                tmp[dim] = data[data_idx];
            }
        }
        return tmp.data();
    }

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    // Compute inner products between query (BF16) and a batch of vectors in a chunk
    // This is the main AMX-optimized routine
    //
    // Parameters:
    //   chunk_id: which chunk to process
    //   qbf16: query vector in BF16 format (size d)
    //   vec_offsets: array of vector offsets within the chunk (size n)
    //   n: number of vectors to compute
    //   out: output buffer for results (size n)
    //
    // Returns: true if AMX was used, false if fell back to AVX512
    bool compute_batch_ip_amx(
            size_t chunk_id,
            const uint16_t* qbf16,
            const size_t* vec_offsets,
            size_t n,
            float* out) const {

        if (n == 0 || n > 16) {
            // AMX tile size limits batch to 16 vectors at a time
            return false;
        }

        if (chunk_id >= nchunks) {
            return false;
        }

        size_t actual_chunk_size = (chunk_id == nchunks - 1) ? last_chunk_size : chunk_size;

        // Verify all offsets are valid
        for (size_t i = 0; i < n; i++) {
            if (vec_offsets[i] >= actual_chunk_size) {
                return false;
            }
        }

        // Enable AMX for this thread
        if (!amx_enable_this_thread()) {
            return false;
        }

        // Pack the selected vectors into a contiguous n x d matrix
        // This is still needed because vec_offsets may not be contiguous
        // Future optimization: if offsets are contiguous, skip packing
        thread_local std::vector<uint16_t> packed;
        packed.resize(n * d);

        for (size_t vec_i = 0; vec_i < n; vec_i++) {
            size_t offset = vec_offsets[vec_i];
            for (size_t dim = 0; dim < d; dim++) {
                size_t src_idx = (chunk_id * d + dim) * chunk_size + offset;
                packed[vec_i * d + dim] = data[src_idx];
            }
        }

        // Configure AMX tiles for n x d matrix times d x 1 vector
        if (!configure_amx_tiles(n, d)) {
            return false;
        }

        // Zero the accumulator tile (tile 2)
        _tile_zero(2);

        // Process dimensions in blocks of 32 (BF16 elements per DPBF16PS)
        size_t d_blocks = d / 32;
        size_t d_remain = d % 32;

        int a_stride = d * sizeof(uint16_t);  // Row stride of packed matrix
        int b_stride = 4;  // 1 column of BF16 query

        for (size_t blk = 0; blk < d_blocks; blk++) {
            const uint16_t* a_ptr = packed.data() + blk * 32;
            const uint16_t* b_ptr = qbf16 + blk * 32;

            _tile_loadd(0, (const void*)a_ptr, a_stride);
            _tile_loadd(1, (const void*)b_ptr, b_stride);
            _tile_dpbf16ps(2, 0, 1);
        }

        // Handle remainder with scalar
        float results[16] = {0};
        if (d_remain > 0) {
            for (size_t vec_i = 0; vec_i < n; vec_i++) {
                for (size_t dim = d_blocks * 32; dim < d; dim++) {
                    size_t src_idx = (chunk_id * d + dim) * chunk_size + vec_offsets[vec_i];
                    results[vec_i] += decode_bf16(data[src_idx]) * decode_bf16(qbf16[dim]);
                }
            }
        }

        // Store results from tile 2
        _tile_stored(2, (void*)out, 4);

        // Add remainder to results
        for (size_t vec_i = 0; vec_i < n; vec_i++) {
            out[vec_i] += results[vec_i];
        }

        return true;
    }
#endif // __AMX_TILE__ && __AMX_BF16__

    // Compute inner products between query and n vectors at specific global IDs
    // This is a higher-level interface that handles chunk boundaries
    //
    // Key optimization: vectors within the same chunk can be processed together
    // using AMX, avoiding the need for memcpy packing.
    void compute_ip_batch(
            const uint16_t* qbf16,
            const idx_t* vec_ids,
            size_t n,
            float* out) const {

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
        // Group vectors by chunk for batch processing
        // Vectors in the same chunk can use AMX without packing
        constexpr size_t MAX_BATCH = 16;  // AMX tile limit

        size_t i = 0;
        while (i < n) {
            // Find vectors in the same chunk starting from i
            size_t chunk_id = vec_ids[i] / chunk_size;

            // Collect up to MAX_BATCH vectors from the same chunk
            std::vector<size_t> batch_offsets;
            std::vector<size_t> batch_indices;

            for (size_t j = i; j < n && batch_offsets.size() < MAX_BATCH; j++) {
                size_t vec_id = vec_ids[j];
                size_t cid = vec_id / chunk_size;
                size_t offset = vec_id % chunk_size;

                if (cid == chunk_id && offset < chunk_size) {
                    batch_offsets.push_back(offset);
                    batch_indices.push_back(j);
                }
            }

            // Process batch with AMX if we have multiple vectors
            if (batch_offsets.size() >= 2) {
                bool amx_ok = compute_batch_ip_amx(
                    chunk_id, qbf16, batch_offsets.data(),
                    batch_offsets.size(), out);

                if (!amx_ok) {
                    // Fallback to AVX512
                    for (size_t bi = 0; bi < batch_offsets.size(); bi++) {
                        out[batch_indices[bi]] = compute_single_ip_amx_chunked(
                            chunk_id, batch_offsets[bi], qbf16);
                    }
                }
                i += batch_indices.size();
            } else {
                // Single vector: use direct AVX512
                size_t chunk_offset = vec_ids[i] % chunk_size;
                out[i] = compute_single_ip_amx_chunked(chunk_id, chunk_offset, qbf16);
                i++;
            }
        }
#else
        // Non-AMX fallback
        for (size_t i = 0; i < n; i++) {
            const uint16_t* codes = get_vector_codes(vec_ids[i]);
            out[i] = 0.0f;
            for (size_t dim = 0; dim < d; dim++) {
                out[i] += decode_bf16(codes[dim]) * decode_bf16(qbf16[dim]);
            }
        }
#endif
    }

private:
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    // Compute single vector IP within a chunk using AVX512-BF16
    float compute_single_ip_amx_chunked(
            size_t chunk_id,
            size_t chunk_offset,
            const uint16_t* qbf16) const {

        if (chunk_id >= nchunks) return 0.0f;
        size_t actual_chunk_size = (chunk_id == nchunks - 1) ? last_chunk_size : chunk_size;
        if (chunk_offset >= actual_chunk_size) return 0.0f;

        float result = 0.0f;

        if (d >= 32 && d % 32 == 0) {
            __m512 acc = _mm512_setzero_ps();
            for (size_t dim = 0; dim < d; dim += 32) {
                size_t data_idx = (chunk_id * d + dim) * chunk_size + chunk_offset;
                const __m512i va = _mm512_loadu_si512((const void*)(data.data() + data_idx));
                const __m512i vb = _mm512_loadu_si512((const void*)(qbf16 + dim));
                acc = _mm512_dpbf16_ps(acc, (__m512bh)va, (__m512bh)vb);
            }
            result = _mm512_reduce_add_ps(acc);
        } else {
            for (size_t dim = 0; dim < d; dim++) {
                size_t data_idx = (chunk_id * d + dim) * chunk_size + chunk_offset;
                result += decode_bf16(data[data_idx]) * decode_bf16(qbf16[dim]);
            }
        }
        return result;
    }

    static inline thread_local int amx_state = 0;

    static bool amx_enable_this_thread() {
        if (amx_state != 0) {
            return amx_state > 0;
        }

#if defined(__linux__)
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILECFG
#define XFEATURE_XTILECFG 17
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
        long st_cfg = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILECFG);
        long st_data = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
        amx_state = (st_cfg == 0 && st_data == 0) ? 1 : -1;
#else
        amx_state = -1;
#endif
        return amx_state > 0;
    }

    static bool configure_amx_tiles(size_t m, size_t k) {
        // Configure AMX tiles for M x K matrix multiplication
        // Tile 0: A matrix (M x K) in BF16
        // Tile 1: B matrix (K x 1) in BF16
        // Tile 2: C matrix (M x 1) in FP32 (accumulator)

        alignas(64) static thread_local unsigned char cfg[64];
        static thread_local size_t prev_m = 0, prev_k = 0;

        if (m == prev_m && k == prev_k) {
            return true;
        }

        constexpr int kBlock = 32;  // DPBF16PS processes 32 BF16 elements

        const int A_rows = (int)m;
        const int N = 1;
        const int A_colsb = kBlock * 2;  // 32 BF16 = 64 bytes
        const int B_colsb = N * 4;
        const int B_rows = kBlock / 2;  // DPBF16PS expects K/2 rows
        const int C_colsb = N * 4;
        const int C_rows = A_rows;

        std::memset(cfg, 0, sizeof(cfg));
        cfg[0] = 1;  // palette

        // Tile 0: A
        cfg[16 + 2 * 0] = (unsigned char)A_colsb;
        cfg[48 + 0] = (unsigned char)A_rows;

        // Tile 1: B
        cfg[16 + 2 * 1] = (unsigned char)B_colsb;
        cfg[48 + 1] = (unsigned char)B_rows;

        // Tile 2: C
        cfg[16 + 2 * 2] = (unsigned char)C_colsb;
        cfg[48 + 2] = (unsigned char)C_rows;

        _tile_loadconfig(cfg);
        prev_m = m;
        prev_k = k;
        return true;
    }

#endif // __AMX_TILE__ && __AMX_BF16__
};

} // namespace scalar_quantizer
} // namespace faiss
