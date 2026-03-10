/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/random.h>

int reference_pop_min(faiss::HNSW::MinimaxHeap& heap, float* vmin_out) {
    assert(heap.k > 0);
    // returns min. This is an O(n) operation
    int i = heap.k - 1;
    while (i >= 0) {
        if (heap.ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = heap.dis[i];
    i--;
    while (i >= 0) {
        if (heap.ids[i] != -1 && heap.dis[i] < vmin) {
            vmin = heap.dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = heap.ids[imin];
    heap.ids[imin] = -1;
    --heap.nvalid;

    return ret;
}

void test_popmin(int heap_size, int amount_to_put) {
    // create a heap
    faiss::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);
    std::uniform_real_distribution<float> uf(0, 1);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        float distance = uf(rng);
        if (distance >= 0.7f) {
            // add infinity values from time to time
            distance = std::numeric_limits<float>::infinity();
        }
        mm_heap.push(index, distance);
    }

    // clone the heap
    faiss::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference pop_min for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

void test_popmin_identical_distances(
        int heap_size,
        int amount_to_put,
        const float distance) {
    // create a heap
    faiss::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        mm_heap.push(index, distance);
    }

    // clone the heap
    faiss::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference pop_min for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

TEST(HNSW, Test_popmin) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32, 64, 128};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin(size, amount);
        }
    }
}

TEST(HNSW, Test_popmin_identical_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin_identical_distances(size, amount, 1.0f);
        }
    }
}

TEST(HNSW, Test_popmin_infinite_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin_identical_distances(
                    size, amount, std::numeric_limits<float>::infinity());
        }
    }
}

TEST(HNSW, Test_IndexHNSW_METRIC_Lp) {
    // Create an HNSW index with METRIC_Lp and metric_arg = 3
    faiss::IndexFlat storage_index(1, faiss::METRIC_Lp);
    storage_index.metric_arg = 3;
    faiss::IndexHNSW index(&storage_index, 32);

    // Add a single data point
    float data[1] = {0.0};
    index.add(1, data);

    // Prepare a query
    float query[1] = {2.0};
    float distance;
    faiss::idx_t label;

    index.search(1, query, 1, &distance, &label);

    EXPECT_NEAR(distance, 8.0, 1e-5); // Distance should be 8.0 (2^3)
    EXPECT_EQ(label, 0);              // Label should be 0
}

class HNSWTest : public testing::Test {
   protected:
    HNSWTest() {
        xb = std::make_unique<std::vector<float>>(d * nb);
        xb->reserve(d * nb);
        faiss::float_rand(xb->data(), d * nb, 12345);
        index = std::make_unique<faiss::IndexHNSWFlat>(d, M);
        index->add(nb, xb->data());
        xq = std::unique_ptr<std::vector<float>>(
                new std::vector<float>(d * nq));
        xq->reserve(d * nq);
        faiss::float_rand(xq->data(), d * nq, 12345);
        dis = std::unique_ptr<faiss::DistanceComputer>(
                index->storage->get_distance_computer());
        dis->set_query(xq->data() + 0 * index->d);
    }

    const int d = 64;
    const int nb = 2000;
    const int M = 4;
    const int nq = 10;
    const int k = 10;
    std::unique_ptr<std::vector<float>> xb;
    std::unique_ptr<std::vector<float>> xq;
    std::unique_ptr<faiss::DistanceComputer> dis;
    std::unique_ptr<faiss::IndexHNSWFlat> index;
};

/** Do a BFS on the candidates list */
int reference_search_from_candidates(
        const faiss::HNSW& hnsw,
        faiss::DistanceComputer& qdis,
        faiss::ResultHandler& res,
        faiss::HNSW::MinimaxHeap& candidates,
        faiss::VisitedTable& vt,
        faiss::HNSWStats& stats,
        int level,
        int nres_in,
        const faiss::SearchParametersHNSW* params) {
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const faiss::IDSelector* sel = params ? params->sel : nullptr;

    faiss::HNSW::C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        faiss::idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // a reference version
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (d < threshold) {
                    if (res.add_result(d, v1)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }

            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

faiss::HNSWStats reference_greedy_update_nearest(
        const faiss::HNSW& hnsw,
        faiss::DistanceComputer& qdis,
        int level,
        faiss::HNSW::storage_idx_t& nearest,
        float& d_nearest) {
    faiss::HNSWStats stats;

    for (;;) {
        faiss::HNSW::storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        size_t ndis = 0;

        for (size_t i = begin; i < end; i++) {
            faiss::HNSW::storage_idx_t v = hnsw.neighbors[i];
            if (v < 0) {
                break;
            }
            ndis += 1;
            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

std::priority_queue<faiss::HNSW::Node> reference_search_from_candidate_unbounded(
        const faiss::HNSW& hnsw,
        const faiss::HNSW::Node& node,
        faiss::DistanceComputer& qdis,
        int ef,
        faiss::VisitedTable* vt,
        faiss::HNSWStats& stats) {
    int ndis = 0;
    std::priority_queue<faiss::HNSW::Node> top_candidates;
    std::priority_queue<
            faiss::HNSW::Node,
            std::vector<faiss::HNSW::Node>,
            std::greater<faiss::HNSW::Node>>
            candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        faiss::HNSW::storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        for (size_t j = begin; j < end; ++j) {
            int v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

TEST_F(HNSWTest, TEST_search_from_candidate_unbounded) {
    omp_set_num_threads(1);
    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto node = faiss::HNSW::Node(d_nearest, nearest);
    faiss::VisitedTable vt(index->ntotal);
    faiss::HNSWStats stats;

    // actual version
    auto top_candidates = faiss::search_from_candidate_unbounded(
            index->hnsw, node, *dis, k, &vt, stats);

    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(nearest);
    auto reference_node =
            faiss::HNSW::Node(reference_d_nearest, reference_nearest);
    faiss::VisitedTable reference_vt(index->ntotal);
    faiss::HNSWStats reference_stats;

    // reference version
    auto reference_top_candidates = reference_search_from_candidate_unbounded(
            index->hnsw,
            reference_node,
            *dis,
            k,
            &reference_vt,
            reference_stats);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_EQ(top_candidates.size(), reference_top_candidates.size());
}

TEST_F(HNSWTest, TEST_greedy_update_nearest) {
    omp_set_num_threads(1);

    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(reference_nearest);

    // actual version
    auto stats = faiss::greedy_update_nearest(
            index->hnsw, *dis, 0, nearest, d_nearest);

    // reference version
    auto reference_stats = reference_greedy_update_nearest(
            index->hnsw, *dis, 0, reference_nearest, reference_d_nearest);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_NEAR(d_nearest, reference_d_nearest, 0.01);
    EXPECT_EQ(nearest, reference_nearest);
}

TEST_F(HNSWTest, TEST_search_from_candidates) {
    omp_set_num_threads(1);

    std::vector<faiss::idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> reference_I(k * nq);
    std::vector<float> reference_D(k * nq);
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;

    faiss::VisitedTable vt(index->ntotal);
    faiss::VisitedTable reference_vt(index->ntotal);
    int num_candidates = 10;
    faiss::HNSW::MinimaxHeap candidates(num_candidates);
    faiss::HNSW::MinimaxHeap reference_candidates(num_candidates);

    for (int i = 0; i < num_candidates; i++) {
        vt.set(i);
        reference_vt.set(i);
        candidates.push(i, (*dis)(i));
        reference_candidates.push(i, (*dis)(i));
    }

    faiss::HNSWStats stats;
    RH bres(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res(
            bres);

    res.begin(0);
    faiss::search_from_candidates(
            index->hnsw, *dis, res, candidates, vt, stats, 0, 0, nullptr);
    res.end();

    faiss::HNSWStats reference_stats;
    RH reference_bres(nq, reference_D.data(), reference_I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler
            reference_res(reference_bres);
    reference_res.begin(0);
    reference_search_from_candidates(
            index->hnsw,
            *dis,
            reference_res,
            reference_candidates,
            reference_vt,
            reference_stats,
            0,
            0,
            nullptr);
    reference_res.end();
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(I[i * k + j], reference_I[i * k + j], 0.1);
            EXPECT_NEAR(D[i * k + j], reference_D[i * k + j], 0.1);
        }
    }
    EXPECT_EQ(reference_stats.ndis, stats.ndis);
    EXPECT_EQ(reference_stats.nhops, stats.nhops);
    EXPECT_EQ(reference_stats.n1, stats.n1);
    EXPECT_EQ(reference_stats.n2, stats.n2);
}

TEST_F(HNSWTest, TEST_search_neighbors_to_add) {
    omp_set_num_threads(1);

    faiss::VisitedTable vt(index->ntotal);
    faiss::VisitedTable reference_vt(index->ntotal);

    std::priority_queue<faiss::HNSW::NodeDistCloser> link_targets;
    std::priority_queue<faiss::HNSW::NodeDistCloser> reference_link_targets;

    faiss::search_neighbors_to_add(
            index->hnsw,
            *dis,
            link_targets,
            index->hnsw.entry_point,
            (*dis)(index->hnsw.entry_point),
            index->hnsw.max_level,
            vt,
            false);

    faiss::search_neighbors_to_add(
            index->hnsw,
            *dis,
            reference_link_targets,
            index->hnsw.entry_point,
            (*dis)(index->hnsw.entry_point),
            index->hnsw.max_level,
            reference_vt,
            true);

    EXPECT_EQ(link_targets.size(), reference_link_targets.size());
    while (!link_targets.empty()) {
        auto val = link_targets.top();
        auto reference_val = reference_link_targets.top();
        EXPECT_EQ(val.d, reference_val.d);
        EXPECT_EQ(val.id, reference_val.id);
        link_targets.pop();
        reference_link_targets.pop();
    }
}

TEST_F(HNSWTest, TEST_nb_neighbors_bound) {
    omp_set_num_threads(1);
    EXPECT_EQ(index->hnsw.nb_neighbors(0), 8);
    EXPECT_EQ(index->hnsw.nb_neighbors(1), 4);
    EXPECT_EQ(index->hnsw.nb_neighbors(2), 4);
    EXPECT_EQ(index->hnsw.nb_neighbors(3), 4);
    // picking a large number to trigger an exception based on checking bounds
    EXPECT_THROW(index->hnsw.nb_neighbors(100), faiss::FaissException);
}

TEST_F(HNSWTest, TEST_search_level_0) {
    omp_set_num_threads(1);
    std::vector<faiss::idx_t> I(k * nq);
    std::vector<float> D(k * nq);

    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    RH bres1(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res1(
            bres1);
    RH bres2(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res2(
            bres2);

    faiss::HNSWStats stats1, stats2;
    faiss::VisitedTable vt1(index->ntotal);
    faiss::VisitedTable vt2(index->ntotal);
    auto nprobe = 5;
    const faiss::HNSW::storage_idx_t values[] = {1, 2, 3, 4, 5};
    const faiss::HNSW::storage_idx_t* nearest_i = values;
    const float distances[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    const float* nearest_d = distances;

    // search_type == 1
    res1.begin(0);
    index->hnsw.search_level_0(
            *dis, res1, nprobe, nearest_i, nearest_d, 1, stats1, vt1, nullptr);
    res1.end();

    // search_type == 2
    res2.begin(0);
    index->hnsw.search_level_0(
            *dis, res2, nprobe, nearest_i, nearest_d, 2, stats2, vt2, nullptr);
    res2.end();

    // search_type 1 calls search_from_candidates in a loop nprobe times.
    // search_type 2 pushes the candidates and just calls search_from_candidates
    // once, so those stats will be much less.
    EXPECT_GT(stats1.ndis, stats2.ndis);
    EXPECT_GT(stats1.nhops, stats2.nhops);
    EXPECT_GT(stats1.n1, stats2.n1);
    EXPECT_GT(stats1.n2, stats2.n2);
}

#if defined(FAISS_OPT_AMX)
TEST(HNSW, Test_IndexHNSWSQ_BF16IP_AMXLevel0Lifecycle) {
    omp_set_num_threads(1);

    constexpr int d = 256;
    constexpr int nb = 256;
    constexpr int nq = 4;
    constexpr int k = 8;
    constexpr int M = 16;

    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);
    faiss::float_rand(xb.data(), xb.size(), 1234);
    std::copy(xb.begin(), xb.begin() + xq.size(), xq.begin());

    faiss::IndexHNSWSQ index(
            d,
            faiss::ScalarQuantizer::QT_bf16,
            M,
            faiss::METRIC_INNER_PRODUCT);
    index.hnsw.efConstruction = 80;
    index.hnsw.efSearch = 64;

    index.enable_amx_level0_optimization(32);
    EXPECT_FALSE(index.using_amx_level0_optimization());

    index.add(nb, xb.data());
    EXPECT_FALSE(index.using_amx_level0_optimization());

    index.finalize_amx_level0_optimization();
    EXPECT_TRUE(index.using_amx_level0_optimization());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k, -1);

    faiss::hnsw_stats.reset();
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    EXPECT_GT(faiss::hnsw_stats.level0_batch_calls, 0);
    EXPECT_GT(faiss::hnsw_stats.level0_batch_candidates, 0);
    EXPECT_GE(
            faiss::hnsw_stats.ndis,
            faiss::hnsw_stats.level0_batch_candidates);
    for (faiss::idx_t label : labels) {
        EXPECT_GE(label, 0);
    }

    index.disable_amx_level0_optimization();
    EXPECT_FALSE(index.using_amx_level0_optimization());

    faiss::hnsw_stats.reset();
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    EXPECT_EQ(faiss::hnsw_stats.level0_batch_calls, 0);
    EXPECT_EQ(faiss::hnsw_stats.level0_batch_candidates, 0);
}

TEST(HNSW, Test_IndexHNSWSQ_BF16IP_AMXChunkPruning) {
    omp_set_num_threads(1);

    constexpr int d = 64;
    constexpr int chunk_size = 16;
    constexpr int n_chunks = 8;
    constexpr int nb = chunk_size * n_chunks;
    constexpr int k = 4;
    constexpr int M = 16;

    std::vector<float> xb(d * nb, 0.0f);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int primary_dim = chunk % d;
        const int secondary_dim = (primary_dim + 1) % d;
        for (int i = 0; i < chunk_size; ++i) {
            float* vec = xb.data() + (chunk * chunk_size + i) * d;
            vec[primary_dim] = 1.0f;
            vec[secondary_dim] = 0.01f * static_cast<float>(i + 1);
        }
    }

    std::vector<float> xq(d, 0.0f);
    xq[0] = 1.0f;

    faiss::IndexHNSWSQ index(
            d,
            faiss::ScalarQuantizer::QT_bf16,
            M,
            faiss::METRIC_INNER_PRODUCT);
    index.hnsw.efConstruction = 80;
    index.hnsw.efSearch = 64;

    index.enable_amx_level0_optimization(chunk_size);
    index.add(nb, xb.data());

    index.hnsw.entry_point = 0;
    for (faiss::idx_t i = 0; i < nb; ++i) {
        size_t begin, end;
        index.hnsw.neighbor_range(i, 0, &begin, &end);
        size_t cursor = begin;
        const faiss::idx_t chunk_begin = (i / chunk_size) * chunk_size;
        const faiss::idx_t chunk_end = chunk_begin + chunk_size;

        if (i + 1 < chunk_end && cursor < end) {
            index.hnsw.neighbors[cursor++] = i + 1;
        }
        if (i > chunk_begin && cursor < end) {
            index.hnsw.neighbors[cursor++] = i - 1;
        }
        if (i == chunk_begin && chunk_end < nb && cursor < end) {
            index.hnsw.neighbors[cursor++] = chunk_end;
        }
        while (cursor < end) {
            index.hnsw.neighbors[cursor++] = -1;
        }
    }

    index.finalize_amx_level0_optimization();

    std::unique_ptr<faiss::DistanceComputer> scoring_qdis(
            new faiss::NegativeDistanceComputer(
                    index.storage->get_distance_computer()));
    scoring_qdis->set_query(xq.data());

    std::vector<std::pair<float, faiss::idx_t>> scored;
    scored.reserve(nb);
    for (faiss::idx_t i = 0; i < nb; ++i) {
        scored.emplace_back((*scoring_qdis)(i), i);
    }
    std::sort(scored.begin(), scored.end());

    const faiss::idx_t good_chunk = scored.front().second / chunk_size;

    auto* sq_storage = dynamic_cast<faiss::IndexScalarQuantizer*>(index.storage);
    ASSERT_NE(sq_storage, nullptr);
    sq_storage->sq.amx_hnsw_chunk_centroids.assign(n_chunks * d, 0.0f);
    sq_storage->sq.amx_hnsw_chunk_radii.assign(n_chunks, 0.05f);
    for (faiss::idx_t chunk = 0; chunk < n_chunks; ++chunk) {
        float* centroid =
                sq_storage->sq.amx_hnsw_chunk_centroids.data() + chunk * d;
        centroid[chunk == good_chunk ? 0 : ((chunk + 1) % d)] = 1.0f;
    }

    std::unique_ptr<faiss::DistanceComputer> qdis(
            new faiss::NegativeDistanceComputer(
                    index.storage->get_distance_computer()));
    qdis->set_query(xq.data());
    EXPECT_TRUE(qdis->supports_level0_batch_chunk());
    EXPECT_TRUE(qdis->supports_level0_chunk_pruning());

    const faiss::idx_t anchor = scored.front().second;
    const float threshold = scored[k - 1].first;

    std::vector<faiss::idx_t> prunable_chunks;
    for (faiss::idx_t chunk = 0; chunk < n_chunks; ++chunk) {
        const float lower_bound = qdis->level0_chunk_distance_lower_bound(chunk);
        if (chunk == good_chunk) {
            continue;
        }
        if (lower_bound > threshold) {
            prunable_chunks.push_back(chunk);
        }
    }
    ASSERT_FALSE(prunable_chunks.empty());

    size_t begin, end;
    index.hnsw.neighbor_range(anchor, 0, &begin, &end);
    size_t cursor = begin;

    for (const auto& [distance, id] : scored) {
        (void)distance;
        if (id / chunk_size == good_chunk && cursor < end) {
            index.hnsw.neighbors[cursor++] = id;
            if (cursor - begin == static_cast<size_t>(k)) {
                break;
            }
        }
    }
    for (faiss::idx_t chunk : prunable_chunks) {
        if (cursor >= end) {
            break;
        }
        index.hnsw.neighbors[cursor++] = chunk * chunk_size;
    }
    while (cursor < end) {
        index.hnsw.neighbors[cursor++] = -1;
    }

    faiss::HNSW::MinimaxHeap candidates(32);
    for (int i = 0; i < k; ++i) {
        candidates.push(scored[i].second, scored[i].first);
    }

    std::vector<float> distances(k, std::numeric_limits<float>::infinity());
    std::vector<faiss::idx_t> labels(k, -1);
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    RH block_result_handler(1, distances.data(), labels.data(), k);
    RH::SingleResultHandler result_handler(block_result_handler);
    faiss::VisitedTable visited(index.ntotal, index.hnsw.use_visited_hashset);
    faiss::HNSWStats stats;

    result_handler.begin(0);
    faiss::search_from_candidates(
            index.hnsw,
            *qdis,
            result_handler,
            candidates,
            visited,
            stats,
            0,
            0,
            nullptr);
    result_handler.end();

    EXPECT_GT(stats.level0_batch_calls, 0);
    EXPECT_GT(stats.level0_batch_candidates, 0);
    EXPECT_GT(stats.level0_pruned_chunks, 0);
    EXPECT_GT(stats.level0_pruned_candidates, 0);
    for (faiss::idx_t label : labels) {
        EXPECT_GE(label, 0);
    }
}
#endif
