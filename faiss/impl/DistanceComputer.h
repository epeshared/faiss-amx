/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <faiss/Index.h>

namespace faiss {

/***********************************************************
 * The distance computer maintains a current query and computes
 * distances to elements in an index that supports random access.
 *
 * The DistanceComputer is not intended to be thread-safe (eg. because
 * it maintains counters) so the distance functions are not const,
 * instantiate one from each thread if needed.
 *
 * Note that the equivalent for IVF indexes is the InvertedListScanner,
 * that has additional methods to handle the inverted list context.
 ***********************************************************/
struct DistanceComputer {
    /// called before computing distances. Pointer x should remain valid
    /// while operator () is called
    virtual void set_query(const float* x) = 0;

    /// compute distance of vector i to current query
    virtual float operator()(idx_t i) = 0;

    /// compute distances of current query to 4 stored vectors.
    /// certain DistanceComputer implementations may benefit
    /// heavily from this.
    virtual void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) {
        // compute first, assign next
        const float d0 = this->operator()(idx0);
        const float d1 = this->operator()(idx1);
        const float d2 = this->operator()(idx2);
        const float d3 = this->operator()(idx3);
        dis0 = d0;
        dis1 = d1;
        dis2 = d2;
        dis3 = d3;
    }

    /// compute distances of current query to a batch of stored vectors.
    /// Default implementation falls back to per-vector computation.
    virtual void distances_batch(size_t n, const idx_t* idx, float* dis) {
        for (size_t i = 0; i < n; i++) {
            dis[i] = this->operator()(idx[i]);
        }
    }

    virtual bool supports_level0_batch() const {
        return false;
    }

    virtual size_t level0_chunk_size() const {
        return 0;
    }

    virtual bool supports_level0_batch_chunk() const {
        return false;
    }

    virtual void distances_batch_chunk(
            idx_t chunk_id,
            size_t n,
            const uint32_t* offsets,
            float* dis) {
        const size_t chunk_size = level0_chunk_size();
        FAISS_THROW_IF_NOT_MSG(
                chunk_size > 0,
                "level-0 chunk batching requires a positive chunk size");
        thread_local std::vector<idx_t> ids;
        ids.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ids[i] = chunk_id * chunk_size + offsets[i];
        }
        distances_batch(n, ids.data(), dis);
    }

    virtual bool supports_level0_chunk_pruning() const {
        return false;
    }

    virtual float level0_chunk_distance_lower_bound(idx_t) const {
        return -std::numeric_limits<float>::infinity();
    }

    virtual bool supports_level0_chunk_similarity_upper_bound() const {
        return false;
    }

    virtual float level0_chunk_similarity_upper_bound(idx_t) const {
        return std::numeric_limits<float>::infinity();
    }

    /// compute distance between two stored vectors
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;

    virtual ~DistanceComputer() {}
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCT search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        basedis->distances_batch_4(
                idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
        dis3 = -dis3;
    }

    void distances_batch(size_t n, const idx_t* idx, float* dis) override {
        basedis->distances_batch(n, idx, dis);
        for (size_t i = 0; i < n; i++) {
            dis[i] = -dis[i];
        }
    }

    bool supports_level0_batch() const override {
        return basedis->supports_level0_batch();
    }

    size_t level0_chunk_size() const override {
        return basedis->level0_chunk_size();
    }

    bool supports_level0_batch_chunk() const override {
        return basedis->supports_level0_batch_chunk();
    }

    void distances_batch_chunk(
            idx_t chunk_id,
            size_t n,
            const uint32_t* offsets,
            float* dis) override {
        basedis->distances_batch_chunk(chunk_id, n, offsets, dis);
        for (size_t i = 0; i < n; ++i) {
            dis[i] = -dis[i];
        }
    }

    bool supports_level0_chunk_pruning() const override {
        return basedis->supports_level0_chunk_similarity_upper_bound();
    }

    float level0_chunk_distance_lower_bound(idx_t chunk_id) const override {
        return -basedis->level0_chunk_similarity_upper_bound(chunk_id);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() override {
        delete basedis;
    }
};

/*************************************************************
 * Specialized version of the DistanceComputer when we know that codes are
 * laid out in a flat index.
 */
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;
    size_t code_size;

    const float* q = nullptr; // not used in all distance computers

    FlatCodesDistanceComputer(
            const uint8_t* codes,
            size_t code_size,
            const float* q = nullptr)
            : codes(codes), code_size(code_size), q(q) {}

    explicit FlatCodesDistanceComputer(const float* q)
            : codes(nullptr), code_size(0), q(q) {}

    FlatCodesDistanceComputer() : codes(nullptr), code_size(0), q(nullptr) {}

    float operator()(idx_t i) override {
        return distance_to_code(codes + i * code_size);
    }

    /// Computes a partial dot product over a slice of the query vector.
    /// The slice is defined by the following parameters:
    ///   — `offset`: the starting index of the first component to include
    ///   — `num_components`: the number of consecutive components to include
    ///
    /// Components refer to raw dimensions of the flat (uncompressed) query
    /// vector.
    ///
    /// By default, this method throws an error, as it is only implemented
    /// in specific subclasses such as `FlatL2Dis`. Other flat distance
    /// computers may override this when partial dot product support is needed.
    ///
    /// Over time, this method might be changed to a pure virtual function (`=
    /// 0`) to enforce implementation in subclasses that require this
    /// functionality.
    ///
    /// This method is not part of the generic `DistanceComputer` interface
    /// because for compressed representations (e.g., product quantization),
    /// calling `partial_dot_product` repeatedly is often less efficient than
    /// computing the full distance at once.
    ///
    /// Supporting efficient partial scans generally requires a different memory
    /// layout, such as interleaved blocks that keep SIMD lanes full. This is a
    /// non-trivial change and not supported in the current flat layout.
    ///
    /// For more details on partial (or chunked) dot product computations and
    /// the performance trade-offs involved, refer to the Panorama paper:
    /// https://arxiv.org/pdf/2510.00566
    virtual float partial_dot_product(
            const idx_t /* i */,
            const uint32_t /* offset */,
            const uint32_t /* num_components */) {
        FAISS_THROW_MSG("partial_dot_product not implemented");
    }

    /// compute distance of current query to an encoded vector
    virtual float distance_to_code(const uint8_t* code) = 0;

    /// Compute partial dot products of current query to 4 stored vectors.
    /// See `partial_dot_product` for more details.
    virtual void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) {
        // default implementation for correctness
        const float d0 =
                this->partial_dot_product(idx0, offset, num_components);
        const float d1 =
                this->partial_dot_product(idx1, offset, num_components);
        const float d2 =
                this->partial_dot_product(idx2, offset, num_components);
        const float d3 =
                this->partial_dot_product(idx3, offset, num_components);
        dp0 = d0;
        dp1 = d1;
        dp2 = d2;
        dp3 = d3;
    }

    virtual ~FlatCodesDistanceComputer() override {}
};

} // namespace faiss
