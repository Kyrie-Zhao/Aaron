"\n#ifdef _WIN32\n  using uint = unsigned int;\n  using uchar = unsigned char;\n  using ushort = unsigned short;\n  using int64_t = long long;\n  using uint64_t = unsigned long long;\n#else\n  #define uint unsigned int\n  #define uchar unsigned char\n  #define ushort unsigned short\n  #define int64_t long long\n  #define uint64_t unsigned long long\n#endif\nextern \"C\" __global__ void fused_nn_conv2d_add_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {\n  float compute[8];\n  __shared__ float pad_temp_shared[112];\n  __shared__ float placeholder_shared[288];\n  for (int yy_init = 0; yy_init < 2; ++yy_init) {\n    compute[(yy_init)] = 0.000000e+00f;\n    compute[((yy_init + 2))] = 0.000000e+00f;\n    compute[((yy_init + 4))] = 0.000000e+00f;\n    compute[((yy_init + 6))] = 0.000000e+00f;\n  }\n  if (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) < 112) {\n    if (((int)threadIdx.x) < 14) {\n      pad_temp_shared[(((((int)threadIdx.z) * 14) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.z) * 784) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)))];\n    }\n  }\n  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {\n    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) < 32) {\n      if (((((int)threadIdx.z) * 12) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 3)) < 96) {\n        if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 288) {\n          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 36) {\n            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];\n          }\n        }\n      }\n    }\n  }\n  __syncthreads();\n  for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {\n    for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {\n      for (int yy = 0; yy < 2; ++yy) {\n        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[(((((yy * 28) + (ry_inner * 28)) + ((int)threadIdx.x)) + rx_inner))] * placeholder_shared[((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner))]));\n        compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((yy * 28) + (ry_inner * 28)) + ((int)threadIdx.x)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 72))]));\n        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[(((((yy * 28) + (ry_inner * 28)) + ((int)threadIdx.x)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 144))]));\n        compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((yy * 28) + (ry_inner * 28)) + ((int)threadIdx.x)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 216))]));\n      }\n    }\n  }\n  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {\n    T_add[((((((((int)blockIdx.z) * 21632) + (((int)threadIdx.z) * 676)) + (((int)blockIdx.y) * 52)) + (ax2_inner_inner_inner * 26)) + ((int)threadIdx.x)))] = (compute[(ax2_inner_inner_inner)] + placeholder2[(((int)threadIdx.z))]);\n    T_add[(((((((((int)blockIdx.z) * 21632) + (((int)threadIdx.z) * 676)) + (((int)blockIdx.y) * 52)) + (ax2_inner_inner_inner * 26)) + ((int)threadIdx.x)) + 5408))] = (compute[((ax2_inner_inner_inner + 2))] + placeholder2[((((int)threadIdx.z) + 8))]);\n    T_add[(((((((((int)blockIdx.z) * 21632) + (((int)threadIdx.z) * 676)) + (((int)blockIdx.y) * 52)) + (ax2_inner_inner_inner * 26)) + ((int)threadIdx.x)) + 10816))] = (compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((int)threadIdx.z) + 16))]);\n    T_add[(((((((((int)blockIdx.z) * 21632) + (((int)threadIdx.z) * 676)) + (((int)blockIdx.y) * 52)) + (ax2_inner_inner_inner * 26)) + ((int)threadIdx.x)) + 16224))] = (compute[((ax2_inner_inner_inner + 6))] + placeholder2[((((int)threadIdx.z) + 24))]);\n  }\n}\n\n"