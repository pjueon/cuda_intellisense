# cuda_intellisense
A simple python script to fix cuda C++ intellisense for visual studio. 

## Background 
If you write cuda C++ code with Visual Studio, you might face some annoying red line like this:  

![before](img/before.jpg)

Typicially you can find theses red lines on:
- texture reference, surface reference  (when you didn't define macro `__CUDACC__`)
- kernel arguments (`<<< ... >>>`)
- cuda macros like `__global__`, `__host__` (when you explicitly defined macro `__CUDACC__`)

With cuda_intellisense, you can fix this!

![after](img/after.jpg)


## Installation
You can install cuda_intellisense by running `scripts/cuda_intellisense.py`.   

**Note**  
You should run it as administrator.

```shell
python scripts/cuda_intellisense.py [options]
```

### Install options
|Option|Description|
|------|-----------|
|`--path=`, `-p=`|installation path. default value: `${CUDA_PATH}/include`|
|`--cuda_path=`|name of the cuda path environment variable (ex> `CUDA_PATH_v10_2`). default value: `CUDA_PATH`|
|`--version`|show the version of cuda_intellisense.|
|`--help`, `-h`|show help.|


## Usage
After installation, visual studio will not complain about cuda code except for the kernel arguments(`<<< ... >>>`).
For the kernel arguments, you can use `KERNEL_ARGS` macro.

It works just like `<<< ... >>>`: 
- `KERNEL_ARGS(grid, block)` is equal to `<<<grid, block>>>`
- `KERNEL_ARGS(grid, block, sh_mem)` is equal to `<<<grid, block, sh_mem>>>`
- `KERNEL_ARGS(grid, block, sh_mem, stream)` is equal to `<<<grid, block, sh_mem, stream>>>`

**Note**  
**You should add `cuda_intellisense/kernel_args.h` file to your source tree** unless those who didn't install the cuda_intellisense won't be able to build your code.
Create `cuda_intellisense` directory to your project and copy `kernel_args.h` file from `headers` directory of this repository or your installation path.   

```cpp
# include "cuda_intellisense/kernel_args.h" // for KERNEL_ARGS macro

/* ... */

// equal to addKernel <<<grid, block>>> (dev_c, dev_b, dev_a);
addKernel KERNEL_ARGS(grid, block) (dev_c, dev_b, dev_a);
```


## How does it work?
The trick is using the [`__INTELLISENSE__`](https://devblogs.microsoft.com/cppblog/troubleshooting-tips-for-intellisense-slowness/) macro, which is only defined when using the IntelliSense compiler. cuda_intellisense uses `#ifdef __INTELLISENSE__` macro to hide stuff from the IntelliSense compiler. 
So the IntelliSense compiler cannot see the cuda codes that it cannot understand and stop complainning about them.


## Test environment
cuda_intellisense was tested on the following environment. 

- OS: Windows 10 
- Visual Studio version: 2019
- CUDA Toolkit version: 10.2, 11.1 

