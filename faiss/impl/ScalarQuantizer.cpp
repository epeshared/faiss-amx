/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <cmath>
#include <cstring>
#include <memory>

#include <type_traits>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>

#include <faiss/impl/scalar_quantizer/training.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>

#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/chunked_transposed.h>

#define THE_LEVEL_TO_DISPATCH SIMDLevel::NONE
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

namespace faiss {

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d, QuantizerType qtype)
        : Quantizer(d), qtype(qtype) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer() {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QT_8bit:
        case QT_8bit_uniform:
        case QT_8bit_direct:
        case QT_8bit_direct_signed:
            code_size = d;
            bits = 8;
            break;
        case QT_4bit:
        case QT_4bit_uniform:
            code_size = (d + 1) / 2;
            bits = 4;
            break;
        case QT_6bit:
            code_size = (d * 6 + 7) / 8;
            bits = 6;
            break;
        case QT_fp16:
            code_size = d * 2;
            bits = 16;
            break;
        case QT_bf16:
            code_size = d * 2;
            bits = 16;
            break;
        default:
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    using scalar_quantizer::train_NonUniform;
    using scalar_quantizer::train_Uniform;

    int bit_per_dim = qtype == QT_4bit_uniform ? 4
            : qtype == QT_4bit                 ? 4
            : qtype == QT_6bit                 ? 6
            : qtype == QT_8bit_uniform         ? 8
            : qtype == QT_8bit                 ? 8
                                               : -1;

    switch (qtype) {
        case QT_4bit_uniform:
        case QT_8bit_uniform:
            train_Uniform(
                    rangestat,
                    rangestat_arg,
                    n * d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_4bit:
        case QT_8bit:
        case QT_6bit:
            train_NonUniform(
                    rangestat,
                    rangestat_arg,
                    n,
                    int(d),
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_fp16:
        case QT_8bit_direct:
        case QT_bf16:
        case QT_8bit_direct_signed:
            // no training necessary
            break;
        default:
            break;
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
    return with_simd_level([&]<SIMDLevel SL>() -> SQuantizer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* q = scalar_quantizer::sq_select_quantizer<SL>(
                    qtype, d, trained);
            if (q) {
                return q;
            }
        }
        return scalar_quantizer::sq_select_quantizer<SIMDLevel::NONE>(
                qtype, d, trained);
    });
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        squant->encode_vector(x + i * d, codes + i * code_size);
    }
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        squant->decode_vector(codes + i * code_size, x + i * d);
    }
}

ScalarQuantizer::SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    return with_simd_level([&]<SIMDLevel SL>() -> SQDistanceComputer* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* dc = scalar_quantizer::sq_select_distance_computer<SL>(
                    metric, qtype, d, trained);
            if (dc) {
                return dc;
            }
        }
        return scalar_quantizer::sq_select_distance_computer<SIMDLevel::NONE>(
                metric, qtype, d, trained);
    });
}

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
    return with_simd_level([&]<SIMDLevel SL>() -> InvertedListScanner* {
        if constexpr (SL != SIMDLevel::NONE) {
            auto* s = scalar_quantizer::sq_select_InvertedListScanner<SL>(
                    qtype,
                    mt,
                    d,
                    code_size,
                    trained,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
            if (s) {
                return s;
            }
        }
        return scalar_quantizer::sq_select_InvertedListScanner<SIMDLevel::NONE>(
                qtype,
                mt,
                d,
                code_size,
                trained,
                quantizer,
                store_pairs,
                sel,
                by_residual);
    });
}

/*******************************************************************
 * Transposed layout methods for AMX-friendly BF16 storage
 ********************************************************************/

void ScalarQuantizer::add_vectors_transposed(const float* x, size_t n) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == QT_bf16,
            "transposed layout only supported for BF16 quantization");
    FAISS_THROW_IF_NOT_MSG(
            use_transposed_layout,
            "transposed layout not enabled");

    // Initialize transposed storage if not already done
    if (!transposed_storage) {
        transposed_storage = std::make_unique<scalar_quantizer::ChunkedTransposedBF16>(
                d, transposed_chunk_size);
    }

    // Add vectors to transposed storage
    transposed_storage->add_vectors(x, n);
}

void ScalarQuantizer::enable_amx_hnsw_layout(size_t chunk_size) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == QT_bf16,
            "AMX HNSW layout only supported for BF16 quantization");
    use_amx_hnsw_layout = true;
    amx_hnsw_chunk_size = std::max<size_t>(1, chunk_size);
    amx_hnsw_layout_ready = false;
}

void ScalarQuantizer::disable_amx_hnsw_layout() {
    use_amx_hnsw_layout = false;
    amx_hnsw_layout_ready = false;
    amx_hnsw_chunk_centroids.clear();
    amx_hnsw_chunk_radii.clear();
}

void ScalarQuantizer::invalidate_amx_hnsw_layout() {
    amx_hnsw_layout_ready = false;
    amx_hnsw_chunk_centroids.clear();
    amx_hnsw_chunk_radii.clear();
}

void ScalarQuantizer::finalize_amx_hnsw_layout(
        size_t ntotal,
        const uint8_t* codes) {
    if (!use_amx_hnsw_layout) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            qtype == QT_bf16,
            "AMX HNSW layout only supported for BF16 quantization");

    amx_hnsw_chunk_centroids.clear();
    amx_hnsw_chunk_radii.clear();

    if (ntotal == 0) {
        amx_hnsw_layout_ready = false;
        return;
    }

    FAISS_THROW_IF_NOT_MSG(codes, "AMX HNSW layout finalization requires codes");

    const size_t n_chunks = (ntotal + amx_hnsw_chunk_size - 1) /
            amx_hnsw_chunk_size;
    amx_hnsw_chunk_centroids.resize(n_chunks * d, 0.0f);
    amx_hnsw_chunk_radii.resize(n_chunks, 0.0f);

    for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
        const size_t chunk_begin = chunk * amx_hnsw_chunk_size;
        const size_t chunk_end = std::min(ntotal, chunk_begin + amx_hnsw_chunk_size);
        const size_t chunk_count = chunk_end - chunk_begin;
        float* centroid = amx_hnsw_chunk_centroids.data() + chunk * d;

        for (size_t idx = chunk_begin; idx < chunk_end; ++idx) {
            const auto* code = reinterpret_cast<const uint16_t*>(
                    codes + idx * code_size);
            for (size_t dim = 0; dim < d; ++dim) {
                centroid[dim] += decode_bf16(code[dim]);
            }
        }

        const float inv_count = 1.0f / static_cast<float>(chunk_count);
        for (size_t dim = 0; dim < d; ++dim) {
            centroid[dim] *= inv_count;
        }

        float radius_sq = 0.0f;
        for (size_t idx = chunk_begin; idx < chunk_end; ++idx) {
            const auto* code = reinterpret_cast<const uint16_t*>(
                    codes + idx * code_size);
            float dist_sq = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                const float diff = decode_bf16(code[dim]) - centroid[dim];
                dist_sq += diff * diff;
            }
            radius_sq = std::max(radius_sq, dist_sq);
        }
        amx_hnsw_chunk_radii[chunk] = std::sqrt(radius_sq);
    }

    amx_hnsw_layout_ready = true;
}

bool ScalarQuantizer::using_amx_hnsw_layout() const {
    return use_amx_hnsw_layout && amx_hnsw_layout_ready;
}

bool ScalarQuantizer::compute_batch_ip_transposed(
        const uint16_t* xq,
        const idx_t* ids,
        size_t nq,
        size_t k,
        float* dist) const {
    FAISS_THROW_IF_NOT_MSG(
            qtype == QT_bf16,
            "transposed layout only supported for BF16 quantization");
    FAISS_THROW_IF_NOT_MSG(
            use_transposed_layout && transposed_storage,
            "transposed layout not enabled or not initialized");

    // Process each query
    for (size_t i = 0; i < nq; i++) {
        const uint16_t* query_bf16 = xq + i * d;
        const idx_t* query_ids = ids + i * k;
        float* query_dists = dist + i * k;

        // Use the chunked transposed storage for batch IP computation
        transposed_storage->compute_ip_batch(
                query_bf16, query_ids, k, query_dists);
    }

    return true;  // Indicates transposed path was used
}

// Explicit destructor definition for unique_ptr with incomplete type
ScalarQuantizer::~ScalarQuantizer() = default;

// Move constructor and assignment
ScalarQuantizer::ScalarQuantizer(ScalarQuantizer&&) noexcept = default;
ScalarQuantizer& ScalarQuantizer::operator=(ScalarQuantizer&&) noexcept = default;

} // namespace faiss
