/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexNSG.h>

#include <omp.h>

#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

namespace faiss {

using idx_t = Index::idx_t;
using storage_idx_t = NSG::storage_idx_t;
using namespace nsg;

/**************************************************************
 * IndexNSG implementation
 **************************************************************/

IndexNSG::IndexNSG(int d, int R, MetricType metric)
        : Index(d, metric),
          nsg(R),
          own_fields(false),
          storage(nullptr),
          is_built(false),
          GK(64),
          build_type(0) {
    nndescent_S = 10;
    nndescent_R = 100;
    nndescent_L = GK + 10;
    nndescent_iter = 10;
}

IndexNSG::IndexNSG(Index* storage, int R)
        : Index(storage->d, storage->metric_type),
          nsg(R),
          own_fields(false),
          storage(storage),
          is_built(false),
          GK(64),
          build_type(0) {
    nndescent_S = 10;
    nndescent_R = 100;
    nndescent_L = GK + 10;
    nndescent_iter = 10;
}

IndexNSG::~IndexNSG() {
    if (own_fields) {
        delete storage;
    }
}

void IndexNSG::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
    // nsg structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexNSG::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const

{
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");

    idx_t check_period = InterruptCallback::get_period_hint(d * nsg.search_L);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                nsg.search(*dis, k, idxi, simi, vt);
                maxheap_reorder(k, simi, idxi);

                vt.advance();
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexNSG::build(idx_t n, const float* x, idx_t* knn_graph, int GK) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
    FAISS_THROW_IF_NOT_MSG(!is_built, "The IndexNSG is already built");
    storage->add(n, x);
    ntotal = storage->ntotal;

    nsg::Graph<idx_t> knng(knn_graph, n, GK);

    // check the knn graph
    idx_t count = 0;
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < GK; j++) {
            idx_t id = knng.at(i, j);
            if (id < 0 || id >= n) {
                count += 1;
            }
        }
    }
    if (count > 0) {
        fprintf(stderr,
                "WARNING: the input knn graph "
                "has %ld invalid entries\n",
                count);
    }

    nsg.build(storage, n, knng, verbose);
    is_built = true;
}

void IndexNSG::build_knng(idx_t n, const float* x, std::vector<idx_t>& knng) {
    if (build_type == 0) { // build with brute force search
        storage->add(n, x);
        ntotal = storage->ntotal;
        FAISS_THROW_IF_NOT(ntotal == n);
        knng.resize(ntotal * (GK + 1));

        storage->assign(ntotal, x, knng.data(), GK + 1);

        for (idx_t i = 0; i < ntotal; i++) {
            // Remove knng[i, 0], assumed that knng[i, 0] is node i itself
            memmove(knng.data() + i * GK,
                    knng.data() + i * (GK + 1) + 1,
                    GK * sizeof(idx_t));
        }
    } else if (build_type == 1) { // build with NNDescent
        IndexNNDescent index(storage, GK);
        index.nndescent.S = nndescent_S;
        index.nndescent.R = nndescent_R;
        index.nndescent.L = nndescent_L;
        index.nndescent.iter = nndescent_iter;
        index.verbose = verbose;

        // prevent IndexNSG from deleting the storage
        index.own_fields = false;

        index.add(n, x);

        // storage->add is already implicit called in IndexNSG.add
        ntotal = storage->ntotal;
        knng.resize(ntotal * GK);

        // cast from idx_t to int
        const int* knn_graph = index.nndescent.final_graph.data();
        for (idx_t i = 0; i < ntotal * GK; i++) {
            knng[i] = knn_graph[i];
        }
    } else {
        FAISS_THROW_MSG("build_type should be 0 or 1");
    }
}

void IndexNSG::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) "
            "instead of IndexNSG directly");
    FAISS_THROW_IF_NOT(is_trained);

    FAISS_THROW_IF_NOT_MSG(
            !is_built, "NSG does not support incremental addition");

    std::vector<idx_t> knng;

    build_knng(n, x, knng);
    build(ntotal, x, knng.data(), GK);
}

void IndexNSG::reset() {
    nsg.reset();
    storage->reset();
    ntotal = 0;
    is_built = false;
}

void IndexNSG::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

/**************************************************************
 * IndexNSGFlat implementation
 **************************************************************/

IndexNSGFlat::IndexNSGFlat() {
    is_trained = true;
}

IndexNSGFlat::IndexNSGFlat(int d, int R, MetricType metric)
        : IndexNSG(new IndexFlat(d, metric), R) {
    own_fields = true;
    is_trained = true;
}

} // namespace faiss
