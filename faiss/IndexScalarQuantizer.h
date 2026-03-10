/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_H
#define FAISS_INDEX_SCALAR_QUANTIZER_H

#include <stdint.h>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/**
 * Flat index built on a scalar quantizer.
 */
struct IndexScalarQuantizer : IndexFlatCodes {
    /// Used to encode the vectors
    ScalarQuantizer sq;

    // Delete copy constructor/assignment since ScalarQuantizer has unique_ptr
    IndexScalarQuantizer(const IndexScalarQuantizer&) = delete;
    IndexScalarQuantizer& operator=(const IndexScalarQuantizer&) = delete;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param qtype  type of scalar quantizer (e.g., QT_4bit)
     * @param metric distance metric used for search (default: METRIC_L2)
     */
    IndexScalarQuantizer(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);

    IndexScalarQuantizer();

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /**
     * Enable transposed layout for BF16 quantization (AMX optimization).
     * Must be called before adding vectors.
     * Only applicable when sq.qtype == QT_bf16.
     *
     * @param chunk_size Number of vectors per chunk (default 64)
     */
    void set_transposed_layout(size_t chunk_size = 64);

    /**
     * Check if transposed layout is enabled.
     */
    bool using_transposed_layout() const;

    /**
     * Override add to support transposed layout for BF16.
     */
    void add(idx_t n, const float* x) override;

        void reset() override;

        void enable_amx_hnsw_layout(size_t chunk_size = 64);

        void disable_amx_hnsw_layout();

        void finalize_amx_hnsw_layout();

        void invalidate_amx_hnsw_layout();

        bool using_amx_hnsw_layout() const;

        bool amx_hnsw_layout_enabled() const;
};

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

struct IndexIVFScalarQuantizer : IndexIVF {
    ScalarQuantizer sq;

    // Delete copy constructor/assignment since ScalarQuantizer has unique_ptr
    IndexIVFScalarQuantizer(const IndexIVFScalarQuantizer&) = delete;
    IndexIVFScalarQuantizer& operator=(const IndexIVFScalarQuantizer&) = delete;

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true,
            bool own_invlists = true);

    IndexIVFScalarQuantizer();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /* standalone codec interface */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
