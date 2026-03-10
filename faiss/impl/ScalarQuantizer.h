/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/Quantizer.h>
#include <memory>

namespace faiss {

struct InvertedListScanner;

namespace scalar_quantizer {
struct ChunkedTransposedBF16;
}


/**
 * The uniform quantizer has a range [vmin, vmax]. The range can be
 * the same for all dimensions (uniform) or specific per dimension
 * (default).
 */

struct ScalarQuantizer : Quantizer {
    enum QuantizerType {
        QT_8bit,         ///< 8 bits per component
        QT_4bit,         ///< 4 bits per component
        QT_8bit_uniform, ///< same, shared range for all dimensions
        QT_4bit_uniform,
        QT_fp16,
        QT_8bit_direct, ///< fast indexing of uint8s
        QT_6bit,        ///< 6 bits per component
        QT_bf16,
        QT_8bit_direct_signed, ///< fast indexing of signed int8s ranging from
                               ///< [-128 to 127]
    };

    QuantizerType qtype = QT_8bit;

    /** The uniform encoder can estimate the range of representable
     * values of the uniform encoder using different statistics. Here
     * rs = rangestat_arg */

    // rangestat_arg.
    enum RangeStat {
        RS_minmax,    ///< [min - rs*(max-min), max + rs*(max-min)]
        RS_meanstd,   ///< [mean - std * rs, mean + std * rs]
        RS_quantiles, ///< [Q(rs), Q(1-rs)]
        RS_optim,     ///< alternate optimization of reconstruction error
    };

    RangeStat rangestat = RS_minmax;
    float rangestat_arg = 0;

    /// bits per scalar code
    size_t bits = 0;

    /// trained values (including the range)
    std::vector<float> trained;

    /// Use AMX-friendly transposed layout for BF16 quantization
    /// Only applicable when qtype == QT_bf16
    bool use_transposed_layout = false;

    /// Chunk size for transposed layout (number of vectors per chunk)
    /// Default 64 is a good balance for AMX tile usage
    size_t transposed_chunk_size = 64;

    /// Transposed BF16 storage (only used when use_transposed_layout == true)
    std::unique_ptr<scalar_quantizer::ChunkedTransposedBF16> transposed_storage;

    /// Enable AMX-specialized level-0 batching for HNSW on BF16/IP storage.
    bool use_amx_hnsw_layout = false;

    /// Whether the current code order matches the finalized HNSW locality order.
    bool amx_hnsw_layout_ready = false;

    /// Consecutive ids within this chunk are processed as a single slab.
    size_t amx_hnsw_chunk_size = 64;

    /// Per-chunk centroids over the finalized HNSW level-0 order.
    std::vector<float> amx_hnsw_chunk_centroids;

    /// Per-chunk L2 radii used to prune whole chunks during HNSW search.
    std::vector<float> amx_hnsw_chunk_radii;

    ScalarQuantizer(size_t d, QuantizerType qtype);
    ScalarQuantizer();
    ~ScalarQuantizer();
    ScalarQuantizer(const ScalarQuantizer&) = delete;
    ScalarQuantizer& operator=(const ScalarQuantizer&) = delete;
    ScalarQuantizer(ScalarQuantizer&&) noexcept;
    ScalarQuantizer& operator=(ScalarQuantizer&&) noexcept;

    /// updates internal values based on qtype and d
    void set_derived_sizes();

    void train(size_t n, const float* x) override;

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* code, float* x, size_t n) const override;

    /*****************************************************
     * Objects that provide methods for encoding/decoding, distance
     * computation and inverted list scanning
     *****************************************************/

    struct SQuantizer {
        // encodes one vector. Assumes code is filled with 0s on input!
        virtual void encode_vector(const float* x, uint8_t* code) const = 0;
        virtual void decode_vector(const uint8_t* code, float* x) const = 0;

        virtual ~SQuantizer() {}
    };

    SQuantizer* select_quantizer() const;

    struct SQDistanceComputer : FlatCodesDistanceComputer {
        const float* level0_chunk_centroids = nullptr;
        const float* level0_chunk_radii = nullptr;
        size_t level0_num_chunks = 0;

        SQDistanceComputer() : FlatCodesDistanceComputer(nullptr) {}

        virtual float query_to_code(const uint8_t* code) const = 0;

        virtual void set_level0_batch_hint(bool enabled, size_t chunk_size) {
            (void)enabled;
            (void)chunk_size;
        }

        virtual void set_level0_chunk_bounds(
                const float* chunk_centroids,
                const float* chunk_radii,
                size_t n_chunks) {
            (void)chunk_centroids;
            (void)chunk_radii;
            (void)n_chunks;
        }

        float distance_to_code(const uint8_t* code) final {
            return query_to_code(code);
        }
    };

    SQDistanceComputer* get_distance_computer(
            MetricType metric = METRIC_L2) const;

    InvertedListScanner* select_InvertedListScanner(
            MetricType mt,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual = false) const;

    /**
     * Compute batch inner products using transposed BF16 layout.
     * Only available when use_transposed_layout == true and qtype == QT_bf16.
     *
     * @param xq     Query vectors in BF16 format, size nq * d (as uint16_t)
     * @param ids    Vector IDs to compute distances for, size nq * k
     * @param nq     Number of queries
     * @param k      Number of vectors per query
     * @param dist   Output distances, size nq * k
     * @return true if AMX was used, false if fell back to standard method
     */
    bool compute_batch_ip_transposed(
            const uint16_t* xq,
            const idx_t* ids,
            size_t nq,
            size_t k,
            float* dist) const;

    /**
     * Add vectors to transposed storage.
     * Only available when use_transposed_layout == true and qtype == QT_bf16.
     */
    void add_vectors_transposed(const float* x, size_t n);

    /**
     * Get pointer to transposed storage (for advanced use).
     */
    const scalar_quantizer::ChunkedTransposedBF16* get_transposed_storage() const {
        return transposed_storage.get();
    }

    void enable_amx_hnsw_layout(size_t chunk_size = 64);

    void disable_amx_hnsw_layout();

    void invalidate_amx_hnsw_layout();

    void finalize_amx_hnsw_layout(size_t ntotal, const uint8_t* codes);

    bool using_amx_hnsw_layout() const;

    const float* get_amx_hnsw_chunk_centroids() const {
        return amx_hnsw_chunk_centroids.data();
    }

    const float* get_amx_hnsw_chunk_radii() const {
        return amx_hnsw_chunk_radii.data();
    }

    size_t get_amx_hnsw_num_chunks() const {
        return amx_hnsw_chunk_radii.size();
    }

    bool amx_hnsw_layout_enabled() const {
        return use_amx_hnsw_layout;
    }
};

} // namespace faiss
