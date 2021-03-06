/*
 * MIT License
 * 
 * Copyright (c) 2022 Jueon Park
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

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