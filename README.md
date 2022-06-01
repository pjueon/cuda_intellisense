# cuda_intellisense
A simple python script to fix cuda C++ intellisense for visual studio. 

## Install cuda_intellisense
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

```cpp
# include "cuda_intellisense/kernel_args.h" // for KERNEL_ARGS macro

// cuda kernel example
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    /* ... */
    
    // equal to addKernel <<<grid, block>>> (dev_c, dev_b, dev_a);
    addKernel KERNEL_ARGS(grid, block) (dev_c, dev_b, dev_a);

    /* ... */
    return 0;
}
```

**Note**  
`KERNEL_ARGS` macro is defined in `kernel_args.h` file. 
**You should add this header file to your project** unless those who didn't install the cuda_intellisense won't be able to build your project.
Create `cuda_intellisense` directory to your project and copy`kernel_args.h` file from `headers` directory of this repository or your installation path.   