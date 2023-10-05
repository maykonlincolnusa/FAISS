/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>

#if defined USE_NVIDIA_RAFT
#include <faiss/gpu/impl/RaftIVFFlat.cuh>
#endif

#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFFlat* index,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // We haven't trained ourselves, so don't construct the IVFFlat
    // index yet
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  coarseQuantizer,
                  dims,
                  metric,
                  0,
                  nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // We could have been passed an already trained coarse quantizer. There is
    // no other quantizer that we need to train, so this is sufficient
    if (this->is_trained) {
        FAISS_ASSERT(this->quantizer);
        set_index_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
        updateQuantizer();
    }
}

GpuIndexIVFFlat::~GpuIndexIVFFlat() {}

void GpuIndexIVFFlat::set_index_(
        GpuResources* resources,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        /// Optional ScalarQuantizer
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space) {
#if defined USE_NVIDIA_RAFT

    if (config_.use_raft) {
        index_.reset(new RaftIVFFlat(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                useResidual,
                scalarQ,
                interleavedLayout,
                indicesOptions,
                space));
    } else
#else
    if (config_.use_raft) {
        FAISS_THROW_MSG(
                "RAFT has not been compiled into the current version so it cannot be used.");
    } else
#endif
    {
        index_.reset(new IVFFlat(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                useResidual,
                scalarQ,
                interleavedLayout,
                indicesOptions,
                space));
    }
}

void GpuIndexIVFFlat::reserveMemory(size_t numVecs) {
    if (config_.use_raft) {
        FAISS_THROW_MSG(
                "Pre-allocation of IVF lists is not supported with RAFT enabled.");
    }

    DeviceScope scope(config_.device);

    reserveMemoryVecs_ = numVecs;
    if (index_) {
        index_->reserveMemory(numVecs);
    }
}

void GpuIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
    DeviceScope scope(config_.device);

    // This will copy GpuIndexIVF data such as the coarse quantizer
    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();
    baseIndex_.reset();

    // The other index might not be trained
    if (!index->is_trained) {
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Otherwise, we can populate ourselves from the other index
    FAISS_ASSERT(is_trained);

    // Copy our lists as well
    set_index_(
            resources_.get(),
            d,
            nlist,
            index->metric_type,
            index->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace);
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFFlat::copyTo(faiss::IndexIVFFlat* index) const {
    DeviceScope scope(config_.device);

    // We must have the indices in order to copy to ourselves
    FAISS_THROW_IF_NOT_MSG(
            ivfFlatConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    GpuIndexIVF::copyTo(index);
    index->code_size = this->d * sizeof(float);

    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (index_) {
        // Copy IVF lists
        index_->copyInvertedListsTo(ivf);
    }
}

size_t GpuIndexIVFFlat::reclaimMemory() {
    DeviceScope scope(config_.device);

    if (index_) {
        return index_->reclaimMemory();
    }

    return 0;
}

void GpuIndexIVFFlat::reset() {
    DeviceScope scope(config_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void GpuIndexIVFFlat::updateQuantizer() {
    FAISS_THROW_IF_NOT_MSG(
            quantizer, "Calling updateQuantizer without a quantizer instance");

    // Only need to do something if we are already initialized
    if (index_) {
        index_->updateQuantizer(quantizer);
    }
}

void GpuIndexIVFFlat::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // just in case someone changed our quantizer
    verifyIVFSettings_();

    if (this->is_trained) {
        FAISS_ASSERT(index_);
        return;
    }

    FAISS_ASSERT(!index_);

#if defined USE_NVIDIA_RAFT
    if (config_.use_raft) {
        // No need to copy the data to host
        trainQuantizer_(n, x);
    } else
#else
    if (config_.use_raft) {
        FAISS_THROW_MSG(
                "RAFT has not been compiled into the current version so it cannot be used.");
    } else
#endif
    {
        // FIXME: GPUize more of this
        // First, make sure that the data is resident on the CPU, if it is not
        // on the CPU, as we depend upon parts of the CPU code
        auto hostData = toHost<float, 2>(
                (float*)x,
                resources_->getDefaultStream(config_.device),
                {n, this->d});
        trainQuantizer_(n, hostData.data());
    }

    // The quantizer is now trained; construct the IVF index
    set_index_(
            resources_.get(),
            this->d,
            this->nlist,
            this->metric_type,
            this->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace);
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    if (reserveMemoryVecs_) {
        if (config_.use_raft) {
            FAISS_THROW_MSG(
                    "Pre-allocation of IVF lists is not supported with RAFT enabled.");
        } else
            index_->reserveMemory(reserveMemoryVecs_);
    }

    this->is_trained = true;
}

void GpuIndexIVF::trainQuantizer_(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    if (n == 0) {
        // nothing to do
        return;
    }

    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (this->verbose) {
            printf("IVF quantizer does not need training.\n");
        }

        return;
    }

    if (this->verbose) {
        printf("Training IVF quantizer on %ld vectors in %dD\n", n, d);
    }

    quantizer->reset();

#if defined USE_NVIDIA_RAFT

    if (config_.use_raft) {
        const raft::device_resources& raft_handle =
                resources_->getRaftHandleCurrentDevice();

        raft::neighbors::ivf_flat::index_params raft_idx_params;
        raft_idx_params.n_lists = nlist;
        raft_idx_params.metric = metric_type == faiss::METRIC_L2
                ? raft::distance::DistanceType::L2Expanded
                : raft::distance::DistanceType::InnerProduct;
        raft_idx_params.add_data_on_build = false;
        raft_idx_params.kmeans_trainset_fraction = 1.0;
        raft_idx_params.kmeans_n_iters = cp.niter;
        raft_idx_params.adaptive_centers = !cp.frozen_centroids;

        auto raft_index = raft::neighbors::ivf_flat::build(
                raft_handle, raft_idx_params, x, n, (idx_t)d);

        raft_handle.sync_stream();

        quantizer->train(nlist, raft_index.centers().data_handle());
        quantizer->add(nlist, raft_index.centers().data_handle());
    } else
#else
    if (config_.use_raft) {
        FAISS_THROW_MSG(
                "RAFT has not been compiled into the current version so it cannot be used.");
    } else
#endif
    {
        // leverage the CPU-side k-means code, which works for the GPU
        // flat index as well
        Clustering clus(this->d, nlist, this->cp);
        clus.verbose = verbose;
        clus.train(n, x, *quantizer);
    }
    quantizer->is_trained = true;
    FAISS_ASSERT(quantizer->ntotal == nlist);
}


} // namespace gpu
} // namespace faiss
