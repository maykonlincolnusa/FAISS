# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project DOES NOT adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
at the moment.

We try to indicate most contributions here with the contributor names who are not part of
the Facebook Faiss team.  Feel free to add entries here if you submit a PR.

## [Unreleased]

## [1.9.0] - 2024-09-24
- Add performance regression tests (#3793)
- Add reverse factory string util, add StringIOReader, add centralized JK (#3879)
- Fix CI 2.0: Compile SQ for avx512 (#3880)
- fix open source CI (#3878)
- Add AVX-512 implementation for the distance and scalar quantizer functions. (#3853)
- torch.distributed kmeans (#3876)
- begin torch_contrib (#3872)
- rewrite python kmeans without scipy (#3873)
- Introduce QuantizerTemplateScaling for SQ (#3870)
- FIx a bug for a non-simdlib code of ResidualQuantizer (#3868)
- more telemetry classes (#3854)
- simplify and refactor create_servicelab_experiment utility (#3867)
- RCQ search microbenchmark (#3863)
- Adding Documentation for ROCm (#3856)
- Add explicit instanciations of `search_dispatch_implem` (#3860)
- add benchmarking for hnsw flat based on efSearch (#3858)
- Fix several typos in code detected by lintian (#3861)
- add hnsw flat benchmark (#3857)
- Add a dockerfile for development (#3851)
- assign_index should default to null (#3855)
- add hnsw unit test for PR 3840 (#3849)
- Adding bucket/path (blobstore) in dataset descriptor (#3848)
- Allow k and M suffixes in IVF indexes (#3812)
- Fixing headers as per OSS requirement (#3847)
- Fix an incorrectly counted the number of computed distances for HNSW (#3840)
- Re-enable Query_L2_MMCodeDistance and Query_IP_MMCodeDistance tests for ROCm (#3838)
- Upgrade to ROCm 6.2 (#3839)
- Do not unnecessarily install CUDA for ROCm
- Quiet down apt-get on ROCm builds (#3836)
- faster hnsw CPU index training (#3822)
- group SWIG tests into one file (#3807)
- Allow search Index without Gt (#3827)
- Add error for overflowing nbits during PQ construction (#3833)
- remove compile options label from gbench (#3834)
- Prevent reordering of imports by auto formatter to avoid crashes (#3826)
- more refactor and add encode/decode steps to benchmark (#3825)
- create perf_tests directory and onboard all scalar quantizer types to benchmark (#3824)
- Build SVE CI with openblas that was compiled with USE_OPENMP=1 (#3776)
- always upload pytest results (#3808)
- Fix deprecated use of 0/NULL (#3817)
- Use weights_only for load (#3796)
- Nightly failure fix - ignore searched vectors with identical distances (#3809)
- Some small improvements. (#3692)
- Re-order imports in Python tests to avoid crashing (#3806)
- Fix bench_fw_codec
- Add standalone Link-Time Optimization option to CMake (#2943)
- Adding support for index builder (#3800)
- Fix parameter names in docstrings (#3795)
- avx512 compilation option (#3798)
- add AMD_ROCM as part of get_compile_options (#3790)
- Enable most of C++ tests on ROCm (#3786)
- fix get_compile_options bug (#3785)
- Add sampling fields to dataset descriptor (#3782)
- Specify to retry only on failed jobs (#3772)
- Move static functions to header file (#3757)
- add reconstruct support to additive quantizers (#3752)
- delete circle CI config (#3732)
- fix ARM64 SVE CI due to openblas version bump (#3777)
- Enable Python tests for ROCm (#3763)
- Reorder imports in torch_test_contrib_gpu (#3761)
- Add midding hipStream SWIG typedef to fix ROCm memleak in Python (#3760)
- introduce options for reducing the overhead for a clustering procedure (#3731)
- minor refactor to avoid else block in IVFPQ reconstruct_from_offset. (#3753)
- Add hnsw search params for bounded queue option (#3748)
- Fixing initialization of dictionary in dataclass (#3749)
- Containerize ROCm build and move it to AMD GPU runners (#3747)
- Adding embedding column to dataset descriptor (#3736)
- Install gpg for ROCm builds (#3744)
- Use $HOME variable to find Conda binaries instead of hard-coded path (#3743)
- Append ROCm flag to compile definitions for Python instead of overwriting (#3742)
- Auto-retry failed CI builds once (#3740)
- Add human readable names to PR build steps (#3739)
- Introduce retry-build workflow (#3718)
- Add labels to test-results file and include ROCm flag (#3738)
- Move nightly builds to 11pm PT (#3737)
- Update code comment regarding PQ's search metrics (#3733)
- Unpin gxx_linux-64 requirement (#3655)
- CMake step to symlink system libraries for RAFT and ROCm (#3725)
- Turn on blocking build for AVX512 and SVE on GHA (#3717)
- ROCm CMake configuration cleanup (#3716)
- ROCm linter and shellcheck warning cleanup (#3715)
- Enable ROCm in build-only mode (#3713)
- ROCm support (#3462)
- suppress warning (#3708)
- Include libfaiss_python_callbacks.so in python installation. (#2062)
- fbcode//faiss/tests (#3707)
- turn on SVE opt mode in CI (#3703)
- Merge cmake command in the cmake build action (#3702)
- Gate ARM SVE build behind base Linux build (#3701)
- Add sve targets (#2886)
- Fix radius search with HSNW and IP (#3698)
- Add ARM64 build to build_cmake actions for SVE (#3653)
- add get_version() for c_api. (#3688)
- Moved statements to faiss.ai (#3694)
- Back out "Add warning on adding nbits to LSH index factory" (#3690)
- Add warning on adding nbits to LSH index factory (#3687)
- First attempt at LSH matching with nbits (#3679)
- 1720 - expose FAISS version field to c_api (#3635)
- Fix CI for AVX512 (#3649)
- QINCo implementation in CPU Faiss (#3608)
- Add search functionality to FlatCodes (#3611)
- add dispatcher for VectorDistance and ResultHandlers
- Set verbosoe before train (#3619)
- Rename autoclose to autoclose.yml (#3618)
- Create autoclose GHA workflow (#3614)
- Fix typo in matrix mult (#3607)
- Adding missing includes which are necessary for building (#3609)
- Non-Blocking AVX512 Build on self-hosted github runner (#3602)
- Fix seg faults in CAGRA C++ unit tests (#3552)
- Add SQ8bit signed quantization (#3501)
- Refactor bench_fw to support train, build & search in parallel (#3527)
- Adding faiss bench_fw to bento faiss kernel (#3531)
- Add ABS_INNER_PRODUCT metric (#3524)
- Bump libraft to 24.06 to unblock nightly RAFT builds (#3522)
- Unbreak RAFT conda builds (#3519)
- fix Windows build - signed int OMP for MSVC (#3517)
- Consolidate build environment configuration steps in cmake builds (#3516)
- typo in test_io_no_storage (#3515)
- Add conda bin to path early in the cmake GitHub action (#3512)
- add use_raft to knn_gpu (torch) (#3509)
- fix spurious include to land the cagra diff (#3502)
- Interop between CAGRA and HNSW (#3252)
- Update .gitignore (#3492)
- Add cpp tutorial for index factory refine index construction (#3494)
- add skip_storage flag to HNSW (#3487)
- Adding buck target for experiment bench_fw_ivf (#3423)
- sys.big_endian to sys.byteorder (#3422)
- fix algorithm of spreading vectors over shards (#3374)
- Workaround for missing intrinsic on gcc < 9 (#3481)
- Add python tutorial on different indexs refinement and respect accuracy measurement (#3480)
- Delete Raft Handle (#3435)
- Remove duplicate NegativeDistanceComputer instances (#3450)
- Remove extra semi colon from deprecated/libmccpp/ThreadSafeClientPool.h (#3479)
- Disable CircleCI builds (#3477)
- Gate all PR builds behind linux-x86_64-cmake in GitHub Actions (#3476)
- Add display names to all PR build jobs on GitHub Actions (#3475)
- QT_bf16 for scalar quantizer for bfloat16 (#3444)
- Add tutorial for FastScan with refinement for cpp (#3474)
- Fix CUDA 11.4.4 nightly in GitHub Actions (#3473)
- Fix cron schedule for nightlies via GitHub Actions (#3470)
- Add FastScan refinement tutorial for python (#3469)
- Add tutorial on PQFastScan for cpp (#3468)
- Missed printing 'D' (#3433)
- Add tutorial for FastScan (#3465)
- Enable nightly builds via GitHub Actions (#3467)
- Fix CUDA 11.4.4 builds under CircleCI (#3466)
- Properly pass the label for conda upload steps (#3464)
- Fix linter warnings in faiss-gpu Conda build script (#3463)
- Relax version requirements for action steps (#3461)
- Enable linux-x86_64-GPU-packages-CUDA-11-4-4 build via GitHub Actions (#3460)
- Workaround for CUDA 11.4.4 build in Conda on Ubuntu 22 / v6 kernel (#3459)
- Cleaning up more unnecessary print (#3455)
- GitHub Actions files cleanup (#3454)
- Delete all remaining print (#3452)
- Improve testing code step 1 (#3451)
- Implement METRIC.NaNEuclidean (#3414)
- Enable both RAFT package builds and CUDA 12.1.1 GPU package build (#3441)
- stabilize formatting for bench_cppcontrib_sa_decode.cpp (#3443)
- Add cuda-toolkit package dependency to faiss-gpu and faiss-gpu-raft conda build recipes (#3440)
- Remove unused variables in faiss/IndexIVFFastScan.cpp (#3439)
- interrupt for NNDescent (#3432)
- fix install instructions (#3442)
- Get rid of redundant instructions in ScalarQuantizer (#3430)
- Add disabled linux-x86_64-AVX512-cmake build on GitHub Actions (#3428)
- Enable linux-x86_64-GPU-cmake build on GitHub Actions (#3427)
- Update system dependencies to enable CUDA builds on v6 kernel and newer libc (#3426)
- PowerPC, improve code generation for function fvec_L2sqr (#3416)
- Enable linux-x86_64-GPU-w-RAFT-cmake build via GitHub Actions (#3418)
- TimeoutCallback C++ and Python (#3417)
- Enable osx-arm64-packages build via GitHub Actions (#3411)
- Change linux-arm64-packages build to use 2-core-ubuntu-arm for better availability (#3410)
- Enable packages builds on main for windows, linux-arm64, linux-x86_64 via GitHub Actions (#3409)
- Enable linux-arm64-conda check via GitHub Actions (#3407)
- Enable windows-x86_64-conda build via GitHub Actions (#3406)
- Enable linux-x86_64-conda build via GitHub Actions (#3405)
- Add format check
- Fix #3379: Add tutorial for HNSW index (#3381)
- Add linux-x86_64-AVX2-cmake build
- Initial config and linux-x86_64-cmake build job only
- Fix deprecated use of 0/NULL in faiss/python/python_callbacks.cpp + 1
- Demo on how to address mulitple index contents
- fix raft log spew
- Fix swig osx (#3357)
- Fix IndexBinary.assign Python method
- Few fixes in bench_fw to enable IndexFromCodec (#3383)
- support big-endian machines (#3361)
- Fix the endianness issue in AIX while running the benchmark. (#3345)
- Unroll loop in lookup_2_lanes (#3364)
- remove unused code (#3371)
- Switch clang-format-11 to clang-format-18 (#3372)
- Update required cmake version to 3.24. (#3305)
- Apply clang-format 18
- Remove unused variables in faiss/IndexIVF.cpp
- Switch sprintf to snprintf (#3363)
- selector parameter for FastScan (#3362)
- Improve filtering & search parameters propagation (#3304)
- Support for Remove ids from IVFPQFastScan index (#3354)
- Revert D55723390: Support for Remove ids from IVFPQFastScan index
- Change index_cpu_to_gpu to throw for indices not implemented on GPU (#3336)
- Support for Remove ids from IVFPQFastScan index (#3349)
- Implement reconstruct_n for GPU IVFFlat indexes (#3338)
- Support of skip_ids in merge_from_multiple function of OnDiskInvertedLists (#3327)
- Fix missing overload variable in Rocksdb ivf demo (#3326)
- Throw when attempting to move IndexPQ to GPU (#3328)
- Add the ability to clone and read binary indexes to the C API. (#3318)
- AVX512 for PQFastScan (#3276)
- Fix faiss swig build with version > 4.2.x (#3315)
- Fix problems when using 64-bit integers. (#3322)
- Change cmake to build googletest from source (#3319)
- Fix IVFPQFastScan decode function (#3312)
- enable rapidsai-nightly channel for libraft (#3317)
- Adding test for IndexBinaryFlat.reconstruct_n() (#3310)
- Handling FaissException in few destructors of ResultHandler.h (#3311)
- Fix HNSW stats (#3309)
- move to raft 24.04 (#3302)
- Remove TypedStorage usage when working with torch_utils (#3301)
- RAFT 24.04 API changes (#3282)
- Use cmake's find_package to link to GTest (#3278)
- Back out "Remove swig version and always rely on the latest version" (#3297)
- Revert D54973709: Remove unused fallthrough
- Remove unused fallthrough (#3296)
- Remove swig version and always rely on the latest version (#3295)
- Dim reduction support in OIVFBBS (#3290)
- Removed index_shard_and_quantize OIVFBBS (#3291)
- AIX compilation fix for io classes (#3275)
- Change intall.md to reflect faiss 1.8.0
- Skip HNSWPQ sdc init with new io flag (#3250)

## [1.8.0] - 2024-02-27
### Added
- Added a new conda package faiss-gpu-raft alongside faiss-cpu and faiss-gpu
- Integrated IVF-Flat and IVF-PQ implementations in faiss-gpu-raft from RAFT by Nvidia [thanks Corey Nolet and Tarang Jain]
- Added a context parameter to InvertedLists and InvertedListsIterator
- Added Faiss on Rocksdb demo to showing how inverted lists can be persisted in a key-value store
- Introduced Offline IVF framework powered by Faiss big batch search
- Added SIMD NEON Optimization for QT_FP16 in Scalar Quantizer. [thanks Naveen Tatikonda]
- Generalized ResultHandler and supported range search for HNSW and FastScan
- Introduced avx512 optimization mode and FAISS_OPT_LEVEL env variable [thanks Alexandr Ghuzva]
- Added search parameters for IndexRefine::search() and IndexRefineFlat::search()
- Supported large two-level clustering
- Added support for Python 3.11 and 3.12
- Added support for CUDA 12

### Changed
- Used the benchmark to find Pareto optimal indices. Intentionally limited to IVF(Flat|HNSW),PQ|SQ indices
- Splitted off RQ encoding steps to another file
- Supported better NaN handling
- HNSW speedup + Distance 4 points [thanks Alexandr Ghuzva]

### Fixed
- Fixed DeviceVector reallocations in Faiss GPU
- Used efSearch from params if provided in HNSW search
- Fixed warp synchronous behavior in Faiss GPU CUDA 12


## [1.7.4] - 2023-04-12
### Added
- Added big batch IVF search for conducting efficient search with big batches of queries
- Checkpointing in big batch search support
- Precomputed centroids support
- Support for iterable inverted lists for eg. key value stores
- 64-bit indexing arithmetic support in FAISS GPU
- IndexIVFShards now handle IVF indexes with a common quantizer
- Jaccard distance support
- CodePacker for non-contiguous code layouts
- Approximate evaluation of top-k distances for ResidualQuantizer and IndexBinaryFlat
- Added support for 12-bit PQ / IVFPQ fine quantizer decoders for standalone vector codecs (faiss/cppcontrib)
- Conda packages for osx-arm64 (Apple M1) and linux-aarch64 (ARM64) architectures
- Support for Python 3.10

### Removed
- CUDA 10 is no longer supported in precompiled packages
- Removed Python 3.7 support for precompiled packages
- Removed constraint for using fine quantizer with no greater than 8 bits for IVFPQ, for example, now it is possible to use IVF256,PQ10x12 for a CPU index

### Changed
- Various performance optimizations for PQ / IVFPQ for AVX2 and ARM for training (fused distance+nearest kernel), search (faster kernels for distance_to_code() and scan_list_*()) and vector encoding
- A magnitude faster CPU code for LSQ/PLSQ training and vector encoding (reworked code)
- Performance improvements for Hamming Code computations for AVX2 and ARM (reworked code)
- Improved auto-vectorization support for IP and L2 distance computations (better handling of pragmas)
- Improved ResidualQuantizer vector encoding (pooling memory allocations, avoid r/w to a temporary buffer)

### Fixed
- HSNW bug fixed which improves the recall rate! Special thanks to zh Wang @hhy3 for this.
- Faiss GPU IVF large query batch fix
- Faiss + Torch fixes, re-enable k = 2048
- Fix the number of distance computations to match max_codes parameter
- Fix decoding of large fast_scan blocks


## [1.7.3] - 2022-11-3
### Added
- Added sparse k-means routines and moved the generic kmeans to contrib
- Added FlatDistanceComputer for all FlatCodes indexes
- Support for fast accumulation of 4-bit LSQ and RQ
- Added product additive quantization
- Support per-query search parameters for many indexes + filtering by ids
- write_VectorTransform and read_vectorTransform were added to the public API (by @AbdelrahmanElmeniawy)
- Support for IDMap2 in index_factory by adding "IDMap2" to prefix or suffix of the input String (by @AbdelrahmanElmeniawy)
- Support for merging all IndexFlatCodes descendants (by @AbdelrahmanElmeniawy)
- Remove and merge features for IndexFastScan (by @AbdelrahmanElmeniawy)
- Performance improvements: 1) specialized the AVX2 pieces of code speeding up certain hotspots, 2) specialized kernels for vector codecs (this can be found in faiss/cppcontrib)


### Fixed
- Fixed memory leak in OnDiskInvertedLists::do_mmap when the file is not closed (by @AbdelrahmanElmeniawy)
- LSH correctly throws error for metric types other than METRIC_L2 (by @AbdelrahmanElmeniawy)

## [1.7.2] - 2021-12-15
### Added
- Support LSQ on GPU (by @KinglittleQ)
- Support for exact 1D kmeans (by @KinglittleQ)

## [1.7.1] - 2021-05-27
### Added
- Support for building C bindings through the `FAISS_ENABLE_C_API` CMake option.
- Serializing the indexes with the python pickle module
- Support for the NNDescent k-NN graph building method (by @KinglittleQ)
- Support for the NSG graph indexing method (by @KinglittleQ)
- Residual quantizers: support as codec and unoptimized search
- Support for 4-bit PQ implementation for ARM (by @vorj, @n-miyamoto-fixstars, @LWisteria, and @matsui528)
- Implementation of Local Search Quantization (by @KinglittleQ)

### Changed
- The order of xb an xq was different between `faiss.knn` and `faiss.knn_gpu`.
Also the metric argument was called distance_type.
- The typed vectors (LongVector, LongLongVector, etc.) of the SWIG interface have
been deprecated. They have been replaced with Int32Vector, Int64Vector, etc. (by h-vetinari)

### Fixed
- Fixed a bug causing kNN search functions for IndexBinaryHash and
IndexBinaryMultiHash to return results in a random order.
- Copy constructor of AlignedTable had a bug leading to crashes when cloning
IVFPQ indices.

## [1.7.0] - 2021-01-27

## [1.6.5] - 2020-11-22

## [1.6.4] - 2020-10-12
### Added
- Arbitrary dimensions per sub-quantizer now allowed for `GpuIndexIVFPQ`.
- Brute-force kNN on GPU (`bfKnn`) now accepts `int32` indices.
- Nightly conda builds now available (for CPU).
- Faiss is now supported on Windows.

## [1.6.3] - 2020-03-24
### Added
- Support alternative distances on GPU for GpuIndexFlat, including L1, Linf and
Lp metrics.
- Support METRIC_INNER_PRODUCT for GpuIndexIVFPQ.
- Support float16 coarse quantizer for GpuIndexIVFFlat and GpuIndexIVFPQ. GPU
Tensor Core operations (mixed-precision arithmetic) are enabled on supported
hardware when operating with float16 data.
- Support k-means clustering with encoded vectors. This makes it possible to
train on larger datasets without decompressing them in RAM, and is especially
useful for binary datasets (see https://github.com/facebookresearch/faiss/blob/main/tests/test_build_blocks.py#L92).
- Support weighted k-means. Weights can be associated to each training point
(see https://github.com/facebookresearch/faiss/blob/main/tests/test_build_blocks.py).
- Serialize callback in python, to write to pipes or sockets (see
https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning).
- Reconstruct arbitrary ids from IndexIVF + efficient remove of a small number
of ids. This avoids 2 inefficiencies: O(ntotal) removal of vectors and
IndexIDMap2 on top of indexIVF. Documentation here:
https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes.
- Support inner product as a metric in IndexHNSW (see
https://github.com/facebookresearch/faiss/blob/main/tests/test_index.py#L490).
- Support PQ of sizes other than 8 bit in IndexIVFPQ.
- Demo on how to perform searches sequentially on an IVF index. This is useful
for an OnDisk index with a very large batch of queries. In that case, it is
worthwhile to scan the index sequentially (see
https://github.com/facebookresearch/faiss/blob/main/tests/test_ivflib.py#L62).
- Range search support for most binary indexes.
- Support for hashing-based binary indexes (see
https://github.com/facebookresearch/faiss/wiki/Binary-indexes).

### Changed
- Replaced obj table in Clustering object: now it is a ClusteringIterationStats
structure that contains additional statistics.

### Removed
- Removed support for useFloat16Accumulator for accumulators on GPU (all
accumulations are now done in float32, regardless of whether float16 or float32
input data is used).

### Fixed
- Some python3 fixes in benchmarks.
- Fixed GpuCloner (some fields were not copied, default to no precomputed tables
with IndexIVFPQ).
- Fixed support for new pytorch versions.
- Serialization bug with alternative distances.
- Removed test on multiple-of-4 dimensions when switching between blas and AVX
implementations.

## [1.6.2] - 2020-03-10

## [1.6.1] - 2019-12-04

## [1.6.0] - 2019-09-24
### Added
- Faiss as a codec: We introduce a new API within Faiss to encode fixed-size
vectors into fixed-size codes. The encoding is lossy and the tradeoff between
compression and reconstruction accuracy can be adjusted.
- ScalarQuantizer support for GPU, see gpu/GpuIndexIVFScalarQuantizer.h. This is
particularly useful as GPU memory is often less abundant than CPU.
- Added easy-to-use serialization functions for indexes to byte arrays in Python
(faiss.serialize_index, faiss.deserialize_index).
- The Python KMeans object can be used to use the GPU directly, just add
gpu=True to the constuctor see gpu/test/test_gpu_index.py test TestGPUKmeans.

### Changed
- Change in the code layout: many C++ sources are now in subdirectories impl/
and utils/.

## [1.5.3] - 2019-06-24
### Added
- Basic support for 6 new metrics in CPU IndexFlat and IndexHNSW (https://github.com/facebookresearch/faiss/issues/848).
- Support for IndexIDMap/IndexIDMap2 with binary indexes (https://github.com/facebookresearch/faiss/issues/780).

### Changed
- Throw python exception for OOM (https://github.com/facebookresearch/faiss/issues/758).
- Make DistanceComputer available for all random access indexes.
- Gradually moving from long to uint64_t for portability.

### Fixed
- Slow scanning of inverted lists (https://github.com/facebookresearch/faiss/issues/836).

## [1.5.2] - 2019-05-28
### Added
- Support for searching several inverted lists in parallel (parallel_mode != 0).
- Better support for PQ codes where nbit != 8 or 16.
- IVFSpectralHash implementation: spectral hash codes inside an IVF.
- 6-bit per component scalar quantizer (4 and 8 bit were already supported).
- Combinations of inverted lists: HStackInvertedLists and VStackInvertedLists.
- Configurable number of threads for OnDiskInvertedLists prefetching (including
0=no prefetch).
- More test and demo code compatible with Python 3 (print with parentheses).

### Changed
- License was changed from BSD+Patents to MIT.
- Exceptions raised in sub-indexes of IndexShards and IndexReplicas are now
propagated.
- Refactored benchmark code: data loading is now in a single file.

## [1.5.1] - 2019-04-05
### Added
- MatrixStats object, which reports useful statistics about a dataset.
- Option to round coordinates during k-means optimization.
- An alternative option for search in HNSW.
- Support for range search in IVFScalarQuantizer.
- Support for direct uint_8 codec in ScalarQuantizer.
- Better support for PQ code assignment with external index.
- Support for IMI2x16 (4B virtual centroids).
- Support for k = 2048 search on GPU (instead of 1024).
- Support for renaming an ondisk invertedlists.
- Support for nterrupting computations with interrupt signal (ctrl-C) in python.
- Simplified build system (with --with-cuda/--with-cuda-arch options).

### Changed
- Moved stats() and imbalance_factor() from IndexIVF to InvertedLists object.
- Renamed IndexProxy to IndexReplicas.
- Most CUDA mem alloc failures now throw exceptions instead of terminating on an
assertion.
- Updated example Dockerfile.
- Conda packages now depend on the cudatoolkit packages, which fixes some
interferences with pytorch. Consequentially, faiss-gpu should now be installed
by conda install -c pytorch faiss-gpu cudatoolkit=10.0.

## [1.5.0] - 2018-12-19
### Added
- New GpuIndexBinaryFlat index.
- New IndexBinaryHNSW index.

## [1.4.0] - 2018-08-30
### Added
- Automatic tracking of C++ references in Python.
- Support for non-intel platforms, some functions optimized for ARM.
- Support for overriding nprobe for concurrent searches.
- Support for floating-point quantizers in binary indices.

### Fixed
- No more segfaults due to Python's GC.
- GpuIndexIVFFlat issues for float32 with 64 / 128 dims.
- Sharding of flat indexes on GPU with index_cpu_to_gpu_multiple.

## [1.3.0] - 2018-07-10
### Added
- Support for binary indexes (IndexBinaryFlat, IndexBinaryIVF).
- Support fp16 encoding in scalar quantizer.
- Support for deduplication in IndexIVFFlat.
- Support for index serialization.

### Fixed
- MMAP bug for normal indices.
- Propagation of io_flags in read func.
- k-selection for CUDA 9.
- Race condition in OnDiskInvertedLists.

## [1.2.1] - 2018-02-28
### Added
- Support for on-disk storage of IndexIVF data.
- C bindings.
- Extended tutorial to GPU indices.

[Unreleased]: https://github.com/facebookresearch/faiss/compare/v1.8.0...HEAD
[1.8.0]: https://github.com/facebookresearch/faiss/compare/v1.7.4...v1.8.0
[1.7.4]: https://github.com/facebookresearch/faiss/compare/v1.7.3...v1.7.4
[1.7.3]: https://github.com/facebookresearch/faiss/compare/v1.7.2...v1.7.3
[1.7.2]: https://github.com/facebookresearch/faiss/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/facebookresearch/faiss/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/facebookresearch/faiss/compare/v1.6.5...v1.7.0
[1.6.5]: https://github.com/facebookresearch/faiss/compare/v1.6.4...v1.6.5
[1.6.4]: https://github.com/facebookresearch/faiss/compare/v1.6.3...v1.6.4
[1.6.3]: https://github.com/facebookresearch/faiss/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/facebookresearch/faiss/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/facebookresearch/faiss/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/facebookresearch/faiss/compare/v1.5.3...v1.6.0
[1.5.3]: https://github.com/facebookresearch/faiss/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/facebookresearch/faiss/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/facebookresearch/faiss/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/facebookresearch/faiss/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/facebookresearch/faiss/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/facebookresearch/faiss/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/facebookresearch/faiss/releases/tag/v1.2.1
