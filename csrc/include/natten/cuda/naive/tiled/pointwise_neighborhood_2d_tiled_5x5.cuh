/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/
/*! \file
    \brief Pointwise-Neighborhood kernel for 2D data.
           Computes attention weights between query points and their
   corresponding key neighborhood. Extra kernel with fused bias (relative
   positional bias.)
           + Tiled kernels for NA with window size 3, 5, 7, 9, 11, and 13 (only
   32 dim per head supported, and these kernels will not be updated anymore in
   favor of the cutlass kernels.)
   
*/

#pragma once
// TODO: remaining dependency to torch: getCurrentCUDAStream
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "natten/cuda/naive/natten_commons.cuh"
#include "natten/cuda/naive/natten_tiled_macros.cuh"
#include "natten/cuda/naive/tiled/base.cuh"

namespace natten {
namespace cuda {
namespace naive {

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Tiled NA //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/////////////// 5x5
// DILATION是定义的时候需要的参数
template <typename scalar_t, int DILATION>
struct PointwiseNeighborhood2DFull5x5 : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DFull5x5() : Base() {}

  static __host__ int get_dim(int dim) {
    return 32;
  }

  __device__ void launch(Params p) {
    // 如果 DILATION > 0 那么就用 DILATION的值，否则用Params中的值，dilation表示膨胀值
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use
    // blockIdx since blockDim will be 1 and threadIdx will always be 0. 
    // 因为批处理头每个线程块的步长为1，所以我们可以只使用blockIdx，因为blockDim将为1，而threadIdx将始终为0。
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / p.heads;
    const int h = z - b * p.heads;
    // 即 z = b * p.heads + h
    // 这里相当于获得了z对应的batch编号和head编号
    /* 
    const dim3 grid(  // 计算方法类似于 (dimension_size + threads_per_dimension * dilation - 1) / threads_per_dimension
        (xsize + XTHREADS * dilation - 1) / XTHREADS,   // grid.x = (5 * width + 20 * dilation - 1) / 20
        (ysize + YTHREADS * dilation - 1) / YTHREADS,   // grid.y = (5 * height + 20 * dilation - 1) / 20
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS); // grid.z = (batch_dim + 1 - 1) / 1
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS); // block = (20, 20, 1)      
    */

    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {

    // TILE_5 * KERNEL_SIZE_5 = 20
    // block一行和一列都是20个元素，lti相当于线性化后的block下标
    const int lti = threadIdx.y * (TILE_5 * KERNEL_SIZE_5) + threadIdx.x;
    // query_stride_0(dim * width * height * heads)
    // query_stride_1(dim * width * height)
    const int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    // si 和 sj 分别代表当前处理的的行和列索引
    // TILE对应线程块处理的query区域大小
    // 假设dilation为1，即不考虑膨胀，那么：si = blockIdx.y * TILE，   sj = blockIdx.x * TILE
    const int si = int(blockIdx.y / dilation) * (TILE_5 * dilation) +
        (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_5 * dilation) +
        (blockIdx.x % dilation);
    // sni 和 snj 是调用 get_window_start 函数计算得到的邻域起始点。
    // 暂且没有研究dilation的情况
    const int sni = get_window_start(
        si, p.height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    const int snj = get_window_start(
        sj, p.width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    // 用于存储key和value的局部共享片
    // 每个线程块要处理TILE_5 * TILE_5个query
    __shared__ scalar_t tile[TILE_5 * TILE_5][DIM_32 + 3];
    // 对应需要 KTILE_5 * KTILE_5个key  8 = 4 + 2 + 2
    __shared__ scalar_t kTile[KTILE_5 * KTILE_5][DIM_32 + 3];

    /* 
      query tile 
      把query对应放到tile片中，以便快速访问
    */
    // qtx表示当前是第几个query
    const int qtx = lti / QSTRIDE_5;
    // qty表示当前是query中的第几个维度
    // 每个线程处理 QITERS_5 个维度，所以还需要乘上它
    const int qty = (lti - qtx * QSTRIDE_5) * QITERS_5;
    if (qtx < TILE_5 * TILE_5) {
      // 此时qi表示当前query在窗口内的的行号
      int qi = qtx / TILE_5;
      // (qtx - qi * TILE_5)表示当前query在窗口内的的列号
      // 最后qj表示在整体上的列号
      const int qj = (qtx - qi * TILE_5) * dilation + sj;
      // 同理计算qi在整体上的行号
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width) {
#pragma unroll
        // 将对应QITERS_5个数拷贝到共享内存中
        for (int ti = 0; ti < QITERS_5; ++ti)
          tile[qtx][qty + ti] = p.query
                                    [batchHeadOffset + qi * p.query_stride_2 +
                                     qj * p.query_stride_3 + qty + ti];
      }
    }
    /* key tile */
    
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE_5 * KTILE_5) {
      int bi = ktx / KTILE_5;
      const int bj = (ktx - bi * KTILE_5) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        const int64_t keyOffset = batchHeadOffset + bi * p.query_stride_2 +
            bj * p.query_stride_3 + kty;
#pragma unroll
        for (int ti = 0; ti < KITERS_32; ++ti)
          kTile[ktx][kty + ti] = p.key[keyOffset + ti];
      }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_5;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_5;
    const int jj = threadIdx.x / KERNEL_SIZE_5;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_5;
    const int i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      const int ni = get_window_start(
          i, p.height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
      const int nj = get_window_start(
          j, p.width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
      scalar_t updt = scalar_t(0);
      const int queryIdx = ii * TILE_5 + jj;
      const int keyIdx = int((ni + ki * dilation - sni) / dilation) * KTILE_5 +
          int((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
        updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

      const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE_5 + kj;
      if (p.bias) {
        const int pi = get_pb_start(
            i, p.height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int pj = get_pb_start(
            j, p.width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int64_t biasIndex =
            h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
        updt += p.bias[biasIndex];
      }
      p.attn[index] = updt;
    }
    //}
  }


/*
LaunchParams lp = Kernel5x5::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
*/
  static LaunchParams get_launch_params(
      int batch_dim,
      int height,
      int width,
      int attention_span,
      int dilation) {
    int xsize = width * KERNEL_SIZE_5; // 5 * width
    int ysize = height * KERNEL_SIZE_5;// 5 * height
    // 这些宏定义了在处理特定核大小时，每个轴上线程的数量，用于确保线程块内的线程总数不超过 GPU 的限制。
    int XTHREADS = XYTHREADS_5;  // 20
    int YTHREADS = XYTHREADS_5;  // 20
    int BATCHTHREADS = BATCHTHREADS_5; // 1
    const dim3 grid(  // 计算方法类似于 (dimension_size + threads_per_dimension * dilation - 1) / threads_per_dimension
        (xsize + XTHREADS * dilation - 1) / XTHREADS,   // grid.x = (5 * width + 20 * dilation - 1) / 20
        (ysize + YTHREADS * dilation - 1) / YTHREADS,   // grid.y = (5 * height + 20 * dilation - 1) / 20
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS); // grid.z = (batch_dim + 1 - 1) / 1
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS); // block = (20, 20, 1)
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, int DILATION>
struct PointwiseNeighborhood2DHalf5x5 : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DHalf5x5() : Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;
  using HalfType = typename HalfHelper::ElementVector;

  static __host__ int get_dim(int dim) {
    return 16;
  }

  __device__ void launch(Params p) {
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use
    // blockIdx since blockDim will be 1 and threadIdx will always be 0. const
    // int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / p.heads;
    const int h = z - b * p.heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_5 * KERNEL_SIZE_5) + threadIdx.x;
    const int stride2 = DIMHALF_32 * p.width;
    const int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    const int si = int(blockIdx.y / dilation) * (TILE_5 * dilation) +
        (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_5 * dilation) +
        (blockIdx.x % dilation);
    const int sni = get_window_start(
        si, p.height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    const int snj = get_window_start(
        sj, p.width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    __shared__ HalfType tile[TILE_5 * TILE_5][DIM_32 + 3];
    __shared__ HalfType kTile[KTILE_5 * KTILE_5][DIM_32 + 3];
    auto query2 = HalfHelper::typecast(p.query);
    auto key2 = HalfHelper::typecast(p.key);

    /* query tile */
    const int qtx = lti / DIMHALF_32;
    const int qty = lti - qtx * DIMHALF_32;
    if (qtx < TILE_5 * TILE_5) {
      int qi = qtx / TILE_5;
      const int qj = (qtx - qi * TILE_5) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width) {
        tile[qtx][qty] =
            query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty];
      }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE_5 * KTILE_5) {
      int bi = ktx / KTILE_5;
      const int bj = (ktx - bi * KTILE_5) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        const int64_t keyOffset =
            batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
        for (int ti = 0; ti < KHALFITERS_32; ++ti)
          kTile[ktx][kty + ti] = key2[keyOffset + ti];
      }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_5;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_5;
    const int jj = threadIdx.x / KERNEL_SIZE_5;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_5;
    const int i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      const int ni = get_window_start(
          i, p.height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
      const int nj = get_window_start(
          j, p.width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
      auto updt = HalfHelper::zero();
      const int queryIdx = ii * TILE_5 + jj;
      const int keyIdx = int((ni + ki * dilation - sni) / dilation) * KTILE_5 +
          int((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int dimOffset = 0; dimOffset < DIMHALF_32; ++dimOffset)
        updt = HalfHelper::fma(
            tile[queryIdx][dimOffset], kTile[keyIdx][dimOffset], updt);

      const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE_5 + kj;
      scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
      if (p.bias) {
        const int pi = get_pb_start(
            i, p.height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int pj = get_pb_start(
            j, p.width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int64_t biasIndex =
            h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
        acc = HalfHelper::add(acc, p.bias[biasIndex]);
      }
      p.attn[index] = acc;
    }
    //}
  }


/*
  batch_dim = batch_size * heads
  spatial_size = height * width
  attention_span = kernel_size * kernel_size
*/
  static LaunchParams get_launch_params(
      int batch_dim,
      int height,
      int width,
      int attention_span,
      int dilation) {
    int xsize = width * KERNEL_SIZE_5;
    int ysize = height * KERNEL_SIZE_5;
    int XTHREADS = XYTHREADS_5;
    int YTHREADS = XYTHREADS_5;
    int BATCHTHREADS = BATCHTHREADS_5;
    const dim3 grid(
        (xsize + XTHREADS * dilation - 1) / XTHREADS,
        (ysize + YTHREADS * dilation - 1) / YTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
