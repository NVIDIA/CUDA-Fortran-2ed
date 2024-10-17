# Source Code from CUDA Fortran for Scientists and Engineers (2ed)

This repository contains the sourse code from the book
CUDA Fortran for Scientists and Engineers,
Best Practices for Efficient CUDA Fortran Programming, Second Edition,
arranged by chapter.

# Copyright and License

SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. \
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); \
you may not use the files in this directory except in compliance with the License. \
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software \
distributed under the License is distributed on an "AS IS" BASIS, \
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. \
See the License for the specific language governing permissions and \
limitations under the License. 

## Part 1: CUDA Fortran Programming

### Chapter 1: Introduction
  * Section 1.3.1: `increment.f90` and `increment.cuf` demonstrate differences
  between Fortran and CUDA Fortran versions of a simple code

  * Section 1.3.2: `multiblock.cuf` demonstrates using multiple thread blocks

  * Section 1.3.3: `multidim.cuf` demonstrates how mutiple dimensions are
  accommodated in CUDA Fortran kernels

  * Section 1.3.4: `explicitInterface.cuf` demonstrates how explicit interfaces
  are used when device code is defined outside a `use`-d module

  * Section 1.3.5: `managed.cuf` and `managedImplicit.cuf` demonstrate use
  of managed memory 

  * Section 1.3.6: `multidimCUF.cuf`, `managedCUF.cuf`, and `managedCUF2.f90`
  demonstrate use of CUF kernels

  * Section 1.4.1: `deviceQuery.cuf` demonstrates how to determine device
  properties at runtime, and `pciBusID.cuf` demonstrates how to determine
  the PCI bus for a specified device

  * Section 1.5: `errorHandling.cuf`, `syncError.cuf`, and `asynError.cuf`
  demonstrate different aspects of error handling of device code

  * Section 1.7: `version.cuf` demonstrates how to determine the CUDA
  driver and CUDA Toolkit versions at runtime.

### Chapter 2: Correctness, Accuracy and Debugging

  * Section 2.1.1: `accuracy.cuf` demonstrates some accuracy issues with
  summations using a single accumulator

  * Section 2.1.2: `fma.cuf` demonstrates how to verify if a fused
  multiply-add (FMA) is used

  * Section 2.2.1: `print.cuf` shows how to print from device code

  * Section 2.2.2: `debug.cuf` is used for debugging with `cuda-gdb`

  * Section 2.2.3: `memcheck.cuf` and `initcheck.cuf` demonstrate how the
  `compute-sanitizer` can be used to check for out-of-bounds
  and initialization errors

### Chapter 3: Performance Measurements and Metrics

  * Section 3.1.2: `events.cuf` demonstrates how to use CUDA events to
  time kernel execution

  * Section 3.1.3: `multidim.cuf` is used to demonstrate profiling by the
  Nsight Systems command-line interface `nsys`

  * Section 3.1.4.1: `nvtxBasic.cuf` demonstrates use of the basic NVTX tooling
  interfaces

  * Section 3.1.4.2: `nvtxAdv.cuf` and `nvtxAdv2.cuf` demonstrate use of the
  advanced NVTX tooling interfaces

  * Section 3.1.4.3: `nvtxAuto.cuf` is used to show how NVTX ranges can be
  automatically generated without modification of source code (see Makefile)

  * Section 3.2: `limitingFactor.cuf` is used to show how kernels can be modified
  to determine performance limiting factors (instruction vs. memory)

  * Section 3.3.1: `peakBandwidth.cuf` uses device management API routines to
  determine the theoretical peak bandwidth

  * Section 3.3.2: `effectiveBandwidth.cuf` uses a simply copy kernel to calculate
  a representative achievable bandwidth
    
### Chapter 4: Synchronization

  * Section 4.1.2.1: `twoKernels.cuf` demonstrates synchronization characteristics
  of kernels run in different streams

  * Section 4.1.3: `pipeline.cuf` demonstrates overlapping data transfers and kernel
  execution

  * Section 4.1.4.2: `streamSync.cuf` demonstrates use of `cudaStreamSycnhronize()`

  * Section 4.1.4.3: `eventSync.cuf` demonstrates use of `cudaEventSycnhronize()`

  * Section 4.1.5.1: `defaultStream.cuf`, `defaultStreamVar.cuf`,
  and `defaultStreamVarExplicit.cuf` show how to set the default stream used
  for kernel launches, data transfers, and reduction operations

  * Section 4.1.5.2: `differentStreamTypes.cuf` and `concurrentKernels.cuf` demonstrate
  characteristics of non-blocking streams

  * Section 4.2.1: `sharedExample.cuf` demonstrates use of static and dynamic shared memory,
  ``sharedMultiple.cuf` shows how offsets are used when multiple dynamic shared memory arrays
  are declared as assumed size arrays

  * Section 4.2.2: `syncthreads.cuf` demonstrates use of `syncthreads_*()` variants

  * Section 4.2.3: `ballot.cuf` demonstrates use of the warp ballot functions

  * Section 4.2.3.1: `shfl.cuf` demonstrates use of the warp shuffle function `__shfl_xor()`

  * Section 4.2.4: `raceAndAtomic.cuf` and `raceAndAtomicShared.cuf` demonstrate
  how atomic operations can be used to avoid race conditions when modifying global
  and shared memory

  * Section 4.2.5: `threadfence.cuf` demonstrates how `threadfence()` is used to order
  memory accesses

  * Section 4.2.6: `cgReverse.cuf` is an cooperative group version of the
  `sharedExample.cuf` code from section 4.2.1

  * Section 4.2.6.1: `smooth.cuf` demonstrate use of grid synchronization via cooperative
  groups

  * Section 4.2.6.2: `swap.cuf` demonstrates how to use distributred shared memory via
  thread block clusters

  
### Chapter 5: Optimization

  * Section 5.1.1: `HDtransfer.cuf` shows performance of data transfers between host
  and device using pageable and pinned host memory, `sliceTransfer.cuf` shows (when
  profiled with `nsys`) that multiple transfers of array slices can be mapped to a single
  `cudaMemcpy2D()` call, and `async.cuf` demonstrates piplining of data transfers and
  kernel execution in different streams to achieve overlap

  * Section 5.2.2.1: `assumedShapeSize.cuf` shows (when compiled with `-gpu=ptxinfo`)
  how assumed-shape array declaration of kernel arguments results in large register
  useage relative to assume-size declarations

  * Section 5.2.2.2: `stide.cuf` and `offset.cuf` are used to determine the effective
  bandwidth of accessing global data with various strides and offsets

  * Section 5.2.3: `local.cuf` shows how to check for local memory usage

  * Section 5.2.4: `constant.cuf` and `constantAttribute.cuf` demonstrate use and
  verification of user-allocated constant memory

  * Section 5.2.5: `loads.cuf` demonstrates caching behavior of loads from global memory

  * Section 5.2.6.1: `maxSharedMemory.cuf` shows how to reserved the maximum amount of
  shared memory allowable

  * Section 5.2.6.2: `transpose.cuf` uses a progressive sequence of kernels to show
  the benefits of various shared-memory optimization strategies when performing a
  matrix transpose

  * Section 5.2.7: `spill.cuf` demonstrates the use of the `launch_bounds()` attribute

  * Section 5.3.1: `parallelism.cuf` demonstrates how the execution configuration and
  occupancy affect performance

  * Section 5.3.2.1: `parallelismPipeline.cuf` demonstrates asynchronous transfers between
  global and shared memory using the pipeline primitives interface

  * Section 5.3.2.2: `cufILP.cuf` demonstrates how to achieve instruction-level
  parallelism in CUF kernels

  * Section 5.4.1.4: `fma.cuf` is used to demonstrate how `-gpu=[no]fma` is used to
  contol use of fused multiply-add instructions

### Chapter 6: Porting Tips and Techniques

  * Section 6.1: `portingBase.f90` is a host code ported to CUDA using managed memory
  (`portingManaged.cuf`) and global memory (`portindDevice.cuf`)

  * Section 6.2: Condition inclusion of code using the predefined symbol `_CUDA`
  (`portingManaged_CUDA.F90`, `portingDevice_CUDA.F90`) and the `!@cuf` sentinel
  (`portingManagedSent.F90`, `portingDeviceSent.F90`)

  * Section 6.3.1-2: Porting of `laplace2D.f90` code via variable ranaming via
  `use` statements (`laplace2DUse.F90`) and via `associate` blocks
  (`portingAssociate.f90`, `laplace2DAssoc.f90`)

  * Section 6.4: The module `union_m.cuf` contains a C-like union for
  reduction of global memory footprint of work arrays
  
  * Section 6.5: The modules `compact_m.cuf` and the optimized `compactOpt_m.cuf` contain
  routines for array compaction
  
### Chapter 7: Interfacing with CUDA C Code and CUDA Libraries

  * Section 7.1: `callingC.cuf` shows how to interface CUDA Fortran with CUDA C routines
  in `c.cu`

  * Sections 7.2.1-2: `sgemmLegacy.cuf` and `sgemmNew.cuf` demonstrate how to interface with
  cuBLAS library using the legacy and new cuBLAS APIs

  * Section 7.2.3: `getrfBatched.cuf` shows how to interface with batched cuBLAS routines

  * Section 7.2.4: `gemmPerf.cuf` shows how to opt in to using the TF32 format and tensor cores
  for matrix mutiplication

  * Section 7.3: `cusparseMV.cuf` and `cusparseMV_ex.cuf` demonstrate use of the cuSPARSE library

  * Section 7.4: ``potr.cuf` demonstrates use of the cuSOLVER library

  * Section 7.5: `matmulTC.cuf` and `matmulTranspose.cuf` demonstrate
  use of the tensor core library through and overloaded `matmul()` routine as well as
  through the cuBLAS interfaces through the use of the `cutensorEx` module

  * Section 7.5.1: `cutensorContraction.cuf` illustrates use of the low-level cuTENSOR
  interfaces

  * Section 7.6: `testSort.cuf` use interfaces to the Thrust C++ template library to
  sort an array

### Chapter 8: MultiGPU-Programming

  * Section 8.1: `minimal.cuf` shos how to select, and allocate global memory on,
  different devices at runtime

  * Section 8.1.1.1: `p2pAccess.cuf` shows how to check for peer-to-peer access
  between devices

  * Section 8.1.2: `directTransfer.cuf` show how to transfer data between global memory
  on different devices without staging through the host memory, `p2pBandwidth.cuf` measures
  the bandwidth of transfers between GPUs

  * Section 8.1.3: `transposeP2P.cuf` performs a distributed transpose using P2P transfers

  * Section 8.2.1: `mpiDevices.cuf` shows how MPI ranks are mapped to devices based on
  the compute mode, and `assignDevice.cuf` shows how to ensure each MPI rank maps to
  a different device regardless of the compute mode setting, through a routine in the
  `mpiDeviceUtil.cuf` module

  * Section 8.2.2: `transposeMPI.cuf` and `transposeCAMPI.cuf` are MPI and
  CUDA-aware MPI versions of the distributed transpose (similar to the P2P transpose
  performed in Section 8.1.3)


## Part 2: Case Studies

### Chapter 9: Monte Carlo Method

  * Section 9.1: `generate_randomnumbers.cuf` demonstrates use of the CURAND library to
  generate random numbers

  * Section 9.2: `compute_pi.cuf` computes pi using the Monte Carlo technique

  * Section 9.2.1: `ieee_accuracy.f90` is used to illustrate accuracy issues
  related to FMA

  * Section 9.3: `pi_performance.CUF` measures performance of the pi calculation
  using shared memory, shuffle, atomic locks, and cooperative group kernels

  * Section 9.3.1: `shflExample.cuf` demonstrates use of the warp shuffle instructions

  * Section 9.3.3: `testPiGridGroup.cuf` shows how to use the grid_group cooperative
  group to perform reductions

  * Section 9.4: `accuracy_sum.cuf` demonstrate issues encountered with accuracy
  of summations

  * Section 9.5: `montecarlo_european_option.cuf` uses Monte Carlo methods to
  price European options

### Chapter 10: Finite Difference Method

  * Section 10.1: `finiteDifference.cuf` calculates a numerical derivatives using a
  nine-point stencil

  * Section 10.1.2: `limitingFactor.cuf` uses modified derivative kernels to
  isolate the limiting factor

  * Section 10.1.4: `finiteDifferenceStr.cuf` calculated derivatives on
  non-uniform grid

  * Section 10.2: `laplace2D.cuf` is a finite difference solution ot the
  2D Laplace equation

### Chapter 11: Applications of the Fast Fourier Transform

  * Section 11.1: `fft_test_c2c.cuf` and `fft_test_r2c.cuf` demonstrate
  use of the CUFFT library

  * Section 11.2: `fft_derivative.cuf` demonstrates use of the CUFFT routines
  to calculate derivatives

  * Section 11.3: `exampleOverlapFFT.cuf` performs a convolution via FFTs

  * Section 11.4: `ns2d.cuf` is a vortex simulation using FFTs
 
### Chapter 12: Ray Tracing

  * Section 12.1: `ppmExample.f90` generates a simple PPM file, the format used
  for images in this chapter

  * Section 12.2: `rgb_m.F90` contains the RGB derived type and overloaded
  operations

  * Section 12.3: `ray.F90` uses the ray derived type in the first ray tracing
  code

  * Section 12.4: `sphere.F90` shows how intersections of rays with a sphere
  are calculated

  * Section 12.5: `normal.F90` calculates surface normals, and `twoSpheres.F90`
  accommodates mutiple objects

  * Section 12.6: `antialias.F90` shows are multiple rays per pixel are used
  in antialiasing

  * Section 12.7.1: `diffuse.F90` generates an image of a sphere with a Lambertian
  or diffuse surface

  * Section 12.7.2: `metal.F90` generates an image of a metalic and diffuse
  spheres

  * Section 12.7.3: `dielectric.F90` generates an image with glass, metal, and
  diffuse spheres

  * Section 12.8: `camera.F90` implements a positionable camera

  * Section 12.9: `defocusBlur.F90` implements focal length effects

  * Section 12.10: `cover.F90` generates a scene with many spheres

  * Section 12.11: `triangle.F90` implements triangular objects

  * Section 12.12: `lights.f90` implements lighted objects

  * Section 12.13: `texture.F90` implements a textured surface
