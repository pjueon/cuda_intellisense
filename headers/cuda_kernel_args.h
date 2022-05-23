#pragma once
#ifndef CUDA_KERNEL_ARGS_H
#define CUDA_KERNEL_ARGS_H

#ifdef __INTELLISENSE__

#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)

#else

#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif

// code reference for macro overloading: https://stackoverflow.com/a/69945225

#define _EXPAND(x) x
#define _GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME

// <<< cuda kernel arguments >>>
#define KERNEL_ARGS(...) _EXPAND(_GET_MACRO(__VA_ARGS__, KERNEL_ARGS4, KERNEL_ARGS3, KERNEL_ARGS2, KERNEL_ARGS1)(__VA_ARGS__))

#endif