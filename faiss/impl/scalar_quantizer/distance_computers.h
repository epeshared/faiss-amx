/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#if defined(FAISS_OPT_AMX)
#include <faiss/impl/scalar_quantizer/amx_bf16_ip.h>
#endif

namespace faiss {

namespace scalar_quantizer {

using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, SIMDLevel SL>
struct DCTemplate : SQDistanceComputer {};

#if defined(__AVX512BF16__)

// Fast path for QT_bf16 + IP on CPUs with AVX512_BF16.
//
// Key idea: quantize query to BF16 once in set_query(), then compute inner
// products using VDPBF16PS against BF16-coded vectors.
//
// Notes:
// - Only enabled when __AVX512BF16__ is available (e.g., -march=sapphirerapids).
// - Requires d % 32 == 0 to use dpbf16 cleanly (32 bf16 elements per op).
template <SIMDLevel SL>
struct DCBF16IPDpbf16 : SQDistanceComputer {
    using Sim = SimilarityIP<SL>;

    QuantizerBF16<SL> quant;
    std::vector<uint16_t> qbf16;

    DCBF16IPDpbf16(size_t d, const std::vector<float>& trained)
            : quant(d, trained), qbf16(d) {}

    void set_query(const float* x) final {
        q = x;
        // Match QuantizerBF16::encode_vector semantics (encode_bf16()).
        for (size_t i = 0; i < quant.d; i++) {
            qbf16[i] = encode_bf16(x[i]);
        }
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16(
            const uint16_t* a,
            const uint16_t* b) const {
        // d is expected to be multiple of 32 for this fast path.
        __m512 acc = _mm512_setzero_ps();
        for (size_t i = 0; i < quant.d; i += 32) {
            const __m512i va = _mm512_loadu_si512((const void*)(a + i));
            const __m512i vb = _mm512_loadu_si512((const void*)(b + i));
            const __m512bh bha = (__m512bh)va;
            const __m512bh bhb = (__m512bh)vb;
            acc = _mm512_dpbf16_ps(acc, bha, bhb);
        }
        return _mm512_reduce_add_ps(acc);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const auto* code1 = (const uint16_t*)(codes + i * code_size);
        const auto* code2 = (const uint16_t*)(codes + j * code_size);
        return compute_code_ip_bf16(code1, code2);
    }

    float query_to_code(const uint8_t* code) const final {
        const auto* c = (const uint16_t*)code;
        return compute_code_ip_bf16(qbf16.data(), c);
    }

    float partial_dot_product(
            idx_t i,
            uint32_t offset,
            uint32_t num_components) override {
        const auto* code = (const uint16_t*)(codes + i * code_size);
        float sum = 0.0f;
        const size_t end = std::min<size_t>(quant.d, offset + num_components);
        for (size_t j = offset; j < end; ++j) {
            sum += q[j] * decode_bf16(code[j]);
        }
        return sum;
    }
};

#if defined(FAISS_OPT_AMX) && defined(__AMX_TILE__) && defined(__AMX_BF16__)

// AMX path for QT_bf16 + IP.
// Uses AMX TILE + DPBF16PS to compute inner products. If AMX enable fails for
// the current thread, falls back to the AVX512-BF16 dpbf16 implementation.
template <SIMDLevel SL>
struct DCBF16IPAmx : SQDistanceComputer {
    using Sim = SimilarityIP<SL>;

    QuantizerBF16<SL> quant;
    std::vector<uint16_t> qbf16;
    bool level0_batch_enabled = false;
    size_t level0_chunk_span = 0;
    float query_l2_norm = 0.0f;

    DCBF16IPAmx(size_t d, const std::vector<float>& trained)
            : quant(d, trained), qbf16(d) {}

    void set_query(const float* x) final {
        q = x;
        query_l2_norm = 0.0f;
        for (size_t i = 0; i < quant.d; i++) {
            qbf16[i] = encode_bf16(x[i]);
            query_l2_norm += x[i] * x[i];
        }
        query_l2_norm = std::sqrt(query_l2_norm);
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16_fallback(
            const uint16_t* a,
            const uint16_t* b) const {
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 32 <= quant.d; i += 32) {
            const __m512i va = _mm512_loadu_si512((const void*)(a + i));
            const __m512i vb = _mm512_loadu_si512((const void*)(b + i));
            const __m512bh bha = (__m512bh)va;
            const __m512bh bhb = (__m512bh)vb;
            acc = _mm512_dpbf16_ps(acc, bha, bhb);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; i < quant.d; ++i) {
            sum += decode_bf16(a[i]) * decode_bf16(b[i]);
        }
        return sum;
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16(
            const uint16_t* a,
            const uint16_t* b) const {
        float out = 0.0f;
        if (amx_bf16_ip_a_rows(a, b, quant.d, 1, &out) == 0) {
            return out;
        }
        return compute_code_ip_bf16_fallback(a, b);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const auto* code1 = (const uint16_t*)(codes + i * code_size);
        const auto* code2 = (const uint16_t*)(codes + j * code_size);
        return compute_code_ip_bf16(code1, code2);
    }

    float query_to_code(const uint8_t* code) const final {
        const auto* c = (const uint16_t*)code;
        return compute_code_ip_bf16(qbf16.data(), c);
    }

    float partial_dot_product(
            idx_t i,
            uint32_t offset,
            uint32_t num_components) override {
        const auto* code = (const uint16_t*)(codes + i * code_size);
        float sum = 0.0f;
        const size_t end = std::min<size_t>(quant.d, offset + num_components);
        for (size_t j = offset; j < end; ++j) {
            sum += q[j] * decode_bf16(code[j]);
        }
        return sum;
    }

    void set_level0_batch_hint(bool enabled, size_t chunk_size) override {
        level0_batch_enabled = enabled;
        level0_chunk_span = chunk_size;
    }

    void set_level0_chunk_bounds(
            const float* chunk_centroids,
            const float* chunk_radii,
            size_t n_chunks) override {
        level0_chunk_centroids = chunk_centroids;
        level0_chunk_radii = chunk_radii;
        level0_num_chunks = n_chunks;
    }

    bool supports_level0_batch() const override {
        return level0_batch_enabled && level0_chunk_span > 0;
    }

    size_t level0_chunk_size() const override {
        return level0_chunk_span;
    }

    bool supports_level0_batch_chunk() const override {
        return supports_level0_batch();
    }

    bool supports_level0_chunk_similarity_upper_bound() const override {
        return supports_level0_batch() && level0_chunk_centroids &&
                level0_chunk_radii && level0_num_chunks > 0;
    }

    float level0_chunk_similarity_upper_bound(idx_t chunk_id) const override {
        if (!supports_level0_chunk_similarity_upper_bound() || chunk_id < 0 ||
            static_cast<size_t>(chunk_id) >= level0_num_chunks) {
            return std::numeric_limits<float>::infinity();
        }

        const float* centroid =
                level0_chunk_centroids + chunk_id * quant.d;
        float centroid_ip = 0.0f;
        for (size_t dim = 0; dim < quant.d; ++dim) {
            centroid_ip += q[dim] * centroid[dim];
        }
        return centroid_ip + query_l2_norm * level0_chunk_radii[chunk_id];
    }

    void distances_batch_chunk(
            idx_t chunk_id,
            size_t n,
            const uint32_t* offsets,
            float* dis) override {
        if (n == 0) {
            return;
        }

        auto scalar_fallback = [&]() {
            const idx_t chunk_base = chunk_id * level0_chunk_span;
            for (size_t i = 0; i < n; ++i) {
                dis[i] = this->operator()(chunk_base + offsets[i]);
            }
        };

        if (!supports_level0_batch_chunk()) {
            scalar_fallback();
            return;
        }

        const size_t code_stride = code_size / sizeof(uint16_t);
        const idx_t chunk_base = chunk_id * level0_chunk_span;
        const auto* chunk_codes = reinterpret_cast<const uint16_t*>(
                codes + chunk_base * code_size);

        thread_local std::vector<uint16_t> packed;
        for (size_t i = 0; i < n;) {
            size_t run_len = 1;
            while (i + run_len < n && run_len < 16 &&
                   offsets[i + run_len] == offsets[i] + run_len) {
                ++run_len;
            }

            if (run_len >= 4) {
                float tmp[16];
                const auto* start_code =
                        chunk_codes + offsets[i] * code_stride;
                if (amx_bf16_ip_a_rows(
                            start_code,
                            qbf16.data(),
                            quant.d,
                            run_len,
                            tmp) == 0) {
                    for (size_t r = 0; r < run_len; ++r) {
                        dis[i + r] = tmp[r];
                    }
                    i += run_len;
                    continue;
                }
            }

            const size_t batch_size = std::min<size_t>(16, n - i);
            packed.resize(batch_size * code_stride);
            for (size_t bi = 0; bi < batch_size; ++bi) {
                const auto* code =
                        chunk_codes + offsets[i + bi] * code_stride;
                std::memcpy(
                        packed.data() + bi * code_stride,
                        code,
                        code_stride * sizeof(uint16_t));
            }

            float tmp[16];
            if (amx_bf16_ip_a_rows(
                        packed.data(),
                        qbf16.data(),
                        quant.d,
                        batch_size,
                        tmp) == 0) {
                for (size_t bi = 0; bi < batch_size; ++bi) {
                    dis[i + bi] = tmp[bi];
                }
            } else {
                for (size_t bi = 0; bi < batch_size; ++bi) {
                    const auto* code =
                            chunk_codes + offsets[i + bi] * code_stride;
                    dis[i + bi] =
                            compute_code_ip_bf16_fallback(qbf16.data(), code);
                }
            }
            i += batch_size;
        }
    }

    void distances_batch(size_t n, const idx_t* idx, float* dis) override {
        if (n == 0) {
            return;
        }

        auto scalar_fallback = [&]() {
            for (size_t i = 0; i < n; i++) {
                dis[i] = this->operator()(idx[i]);
            }
        };

        // For tiny batches, packing overhead can dominate.
        if (n < 8) {
            scalar_fallback();
            return;
        }

        if (!supports_level0_batch()) {
            thread_local std::vector<uint16_t> packed;
            packed.resize(n * quant.d);
            for (size_t i = 0; i < n; i++) {
                const auto* c = (const uint16_t*)(codes + idx[i] * code_size);
                std::memcpy(
                        packed.data() + i * quant.d,
                        c,
                        quant.d * sizeof(uint16_t));
            }

            if (amx_bf16_ip_a_rows(
                        packed.data(), qbf16.data(), quant.d, n, dis) == 0) {
                return;
            }

            scalar_fallback();
            return;
        }

        thread_local std::vector<size_t> order;
        thread_local std::vector<uint16_t> packed;

        order.resize(n);
        std::iota(order.begin(), order.end(), size_t(0));
        std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
            const idx_t il = idx[lhs];
            const idx_t ir = idx[rhs];
            const size_t cl = static_cast<size_t>(il) / level0_chunk_span;
            const size_t cr = static_cast<size_t>(ir) / level0_chunk_span;
            if (cl != cr) {
                return cl < cr;
            }
            return il < ir;
        });

        for (size_t i = 0; i < n;) {
            const size_t chunk_id =
                static_cast<size_t>(idx[order[i]]) / level0_chunk_span;
            size_t chunk_end = i + 1;
            while (chunk_end < n &&
                   static_cast<size_t>(idx[order[chunk_end]]) /
                       level0_chunk_span ==
                           chunk_id) {
                ++chunk_end;
            }

            for (size_t j = i; j < chunk_end;) {
                size_t run_len = 1;
                while (j + run_len < chunk_end && run_len < 16 &&
                       idx[order[j + run_len]] ==
                               idx[order[j]] + static_cast<idx_t>(run_len)) {
                    ++run_len;
                }

                if (run_len >= 4) {
                    float tmp[16];
                    const auto* start_code =
                            (const uint16_t*)(codes + idx[order[j]] * code_size);
                    if (amx_bf16_ip_a_rows(
                                start_code,
                                qbf16.data(),
                                quant.d,
                                run_len,
                                tmp) == 0) {
                        for (size_t r = 0; r < run_len; ++r) {
                            dis[order[j + r]] = tmp[r];
                        }
                        j += run_len;
                        continue;
                    }
                }

                const size_t batch_size = std::min<size_t>(16, chunk_end - j);
                packed.resize(batch_size * quant.d);
                for (size_t bi = 0; bi < batch_size; ++bi) {
                    const auto* c =
                            (const uint16_t*)(codes + idx[order[j + bi]] * code_size);
                    std::memcpy(
                            packed.data() + bi * quant.d,
                            c,
                            quant.d * sizeof(uint16_t));
                }

                float tmp[16];
                if (amx_bf16_ip_a_rows(
                            packed.data(),
                            qbf16.data(),
                            quant.d,
                            batch_size,
                            tmp) == 0) {
                    for (size_t bi = 0; bi < batch_size; ++bi) {
                        dis[order[j + bi]] = tmp[bi];
                    }
                } else {
                    for (size_t bi = 0; bi < batch_size; ++bi) {
                        const auto* c =
                                (const uint16_t*)(codes + idx[order[j + bi]] * code_size);
                        dis[order[j + bi]] =
                                compute_code_ip_bf16_fallback(qbf16.data(), c);
                    }
                }
                j += batch_size;
            }

            i = chunk_end;
        }
    }
};

#endif // FAISS_OPT_AMX && __AMX_TILE__ && __AMX_BF16__

#endif

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }

    float partial_dot_product(
            idx_t i,
            uint32_t offset,
            uint32_t num_components) override {
        if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
            const uint8_t* code = codes + i * code_size;
            float sum = 0.0f;
            const size_t end = std::min<size_t>(quant.d, offset + num_components);
            for (size_t j = offset; j < end; ++j) {
                sum += q[j] * quant.reconstruct_component(code, j);
            }
            return sum;
        }
        return SQDistanceComputer::partial_dot_product(i, offset, num_components);
    }
};

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, SIMDLevel SL>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

/*******************************************************************
 * Selection function
 *******************************************************************/

template <SIMDLevel SL>
SQDistanceComputer* sq_select_distance_computer(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer
} // namespace faiss
