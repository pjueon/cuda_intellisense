# cuda_intellisense
A simple python script to fix cuda C++ intellisense for visual studio. 

## Install cuda_intellisense
You can install cuda_intellisense by running `scripts/cuda_intellisense.py`.   

```shell
python scripts/cuda_intellisense.py [options]
```

**Note**  
You should run it as administrator.

### Install options
|Option|Description|
|------|-----------|
|`--path=`, `-p=`|installation path. default value: `${CUDA_PATH}/include`|
|`--cuda_path=`|name of the cuda path environment variable (ex> `CUDA_PATH_v10_2`). default value: `CUDA_PATH`|
|`--version`|show the version of cuda_intellisense.|
|`--help`, `-h`|show help.|


## Check the version of cuda_intellisense 
From terminal:
```shell
python scripts/cuda_intellisense.py --version
```

From C++ source code:
```cpp
// defined as a macro
CUDA_INTELLISENSE_VERSION
```