#pragma once
#ifndef CUDA_INTELLISENSE_ATTRIBUTE_H
#define CUDA_INTELLISENSE_ATTRIBUTE_H

#ifdef __INTELLISENSE__

#define __CUDACC__

#define __global__ 
#define __host__ 
#define __device__ 
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#define __constant__ 
#define __shared__ 
#define __restrict__
#define __noinline__
#define __forceinline__
#define __managed__

#endif //__INTELLISENSE__
#endif // CUDA_INTELLISENSE_ATTRIBUTE_H