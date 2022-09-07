__global__ convKernel(input,output,Kweight,bias):
    output = 0;
    bind(output[.],blockIdx)
    bind(output[,],threadIdx)
    for i in range(0,.,.)
        for j in range(0,.,.)
            .. for z in range(0,.,.)
                    output[] = input[..]*weight[..]
        
    return output

__global__ pooling(buff):
    ...

__global__ dense(input,output,Kweight,bias):
    ...

# Host Codes
__host__ inference():
    memorycpyH2D(input,input_host,input_size) # copy input to device
    convKernel<<<grid,block>>>(input,output,Kweight,bias)
    pooling<<<..>>>(..)
    ... # Launch other kernels in DNN
    dense <<<..>>> (..)
    memcpyD2H(output_host, out, output_size) # copy output to host


