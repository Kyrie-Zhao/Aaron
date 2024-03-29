/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cuda_module.cc
 */
#include "cuda_module.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "cuda_common.h"

namespace tvm {
namespace runtime {
using tvm::runtime::fcache_zzh;
CUfunction fcache_zzh = NULL;
using detail::params_zzh;
namespace detail {
  void** params_zzh = 0;
}
// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CUDAModuleNode : public runtime::ModuleNode {
 public:
  explicit CUDAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~CUDAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(static_cast<int>(i)));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
      }
    }
  }

  const char* type_key() const final { return "cuda"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cu") {
      ICHECK_NE(cuda_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, cuda_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
  }

  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
      printf("here load the input data\n");
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetFunction " << func_name << " failed with error: " << msg;
    }
    return func;
  }
  // get a global var from primary context in device_id
  CUdeviceptr GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUdeviceptr global;
    size_t nbytes;

    CUresult result = cuModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str());
    ICHECK_EQ(nbytes, expect_nbytes);
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetGlobal " << global_name << " failed with error: " << msg;
    }
    return global;
  }
  void setCUFunction(std::array<CUfunction, kMaxNumGPUs> tmpCUFunction){
    std::cout<<typeid(tmpCUFunction).name();
    std::cout<<"worinimade tmpCUFunction\n";
    //fcache_zzh = tmpCUFunction[0];

  }
  void testFunc(){
    printf("test");
  }
  void SetGrid(int gridnum_1,int gridnum_2,int gridnum_3){
    std::cout<<"\nset grid num \n"<<gridnum_1;
    GridNum_zzh[0] = gridnum_1;
    GridNum_zzh[1] = gridnum_2;
    GridNum_zzh[2] = gridnum_3;
    }
  void SetBlock(int blocknum_1,int blocknum_2,int blocknum_3){
    std::cout<<"\nset BLOCK num \n"<<blocknum_1;
    Blocknum_zzh[0] = blocknum_1;
    Blocknum_zzh[1] = blocknum_2;
    Blocknum_zzh[2] = blocknum_3;
    }
  void setParams(void** params){
    std::cout<<"\n"<<typeid(params).name()<<"\n";
    std::cout<<"worinimade params\n";
    //void** a= params;
    params_zzh = params;
    }
 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string cuda_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class CUDAWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(CUDAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }
    //CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    
    //std::vector<float> stream_kernel_lat_vector; 
    cudaEvent_t start, end,kernel_start_tmp,kernel_end_tmp;
    cudaStream_t streams[2];
    CUresult result,dummy;

    cudaEvent_t m,n;
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaEventCreate(&kernel_start_tmp);
    cudaEventCreate(&kernel_end_tmp);

    cudaEventCreate(&m);
    cudaEventCreate(&n);

    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    float latency = 0.0f;
    float kernel_latency = 0.0f;
    float average_kernel_latency = 0.0f;
    //MODE 0:一个kernel 1:两个kernel，不完全overlap 2:两个kernel，基本完全overlap
    #define MODE 0
    #if MODE == 0
    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);

    cudaEventRecord(start, streams[0]);
    for (int i = 0; i < 100; i++) {    
    
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
                                                 
    }
    cudaStreamSynchronize(streams[0]);
    //cudaStreamSynchronize(streams[1]);
    cudaEventRecord(end, streams[0]);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&latency, start, end) ;  
    std::cout<<"cuda module calculating time:"<<1000*(latency/100)<<"\n";//average one kernel time in us
    #elif MODE == 1
    //wl.grid_dim(0) += 500;
    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[1], void_args, nullptr);
    //cudaDeviceSynchronize();
    cudaEventRecord(start, streams[1]);
    for (int i = 0; i < 10; i++) {    
    
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
    //cudaDeviceSynchronize();
    //cudaEventRecord(m, streams[0]);
    
    //cudaStreamWaitEvent(streams[1],m,0);
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[1], void_args, nullptr);
    //cudaEventRecord(n, streams[1]);
    
    //cudaDeviceSynchronize();                                                         
    }
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaEventRecord(end, streams[1]);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&latency, start, end) ;  
    std::cout<<1000*(latency/10)<<"cuda module calculating time\n";//average one kernel time in us
    #elif MODE ==2
    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[1], void_args, nullptr);
    //cudaDeviceSynchronize();
    cudaEventRecord(start, streams[0]);
    for (int i = 0; i < 100; i++) {    
    
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
    //cudaDeviceSynchronize();
    //cudaEventRecord(m, streams[0]);
    
    //cudaStreamWaitEvent(streams[1],m,0);
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[1], void_args, nullptr);
    //cudaEventRecord(n, streams[1]);
    
    //cudaDeviceSynchronize();                                                         
    }
    cudaStreamSynchronize(streams[0]);
    //cudaStreamSynchronize(streams[1]);
    cudaEventRecord(end, streams[0]);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&latency, start, end) ;  
    std::cout<<"cuda module calculating time:"<<1000*(latency/100)<<"\n";//average one kernel time in us
    #elif MODE == 3
    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
 
    //cudaEventRecord(m, streams[0]);
    //cudaStreamWaitEvent(streams[1], m,0);

    dummy = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[1], void_args, nullptr);
    //cudaEventRecord(n, streams[1]);
    //cudaStreamWaitEvent(streams[0], n,0);
    //cudaDeviceSynchronize();
    cudaEventRecord(start, streams[0]);
    for (int i = 0; i < 100; i++) {    
    
    //cudaEventRecord(kernel_start_tmp, streams[0]);
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[0], void_args, nullptr);
    //cudaDeviceSynchronize();
    //cudaEventRecord(kernel_end_tmp, streams[0]);
    //cudaEventElapsedTime(&kernel_latency, kernel_start_tmp, kernel_end_tmp);
    //stream_kernel_lat_vector.push_back(kernel_latency);
    //std::cout<<kernel_latency<<"\n";

    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                    wl.block_dim(2), 0, streams[1], void_args, nullptr);
  
    //cudaEventRecord(m, streams[1]);
    //cudaStreamWaitEvent(streams[0],m,0);
    //cudaDeviceSynchronize();                                                         
    }

    cudaEventRecord(end, streams[0]);
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&latency, start, end) ;  
    
    //for(int j=0;j<100;j++){
    //  std::cout<<"["<<stream_kernel_lat_vector[j]<<"]\n";
    //}

    std::cout<<1000*(latency/100)<<"cuda module calculating time\n";//average one kernel time in us
    
    #endif
    //cudaEventElapsedTime(&latency, start, end)
    //cudaStreamWaitEvent(event1,stream[0]);
    m_->SetGrid(wl.grid_dim(0),wl.grid_dim(1),wl.grid_dim(2));
    m_->SetBlock(wl.block_dim(0),wl.block_dim(1),wl.block_dim(2));
    m_->setParams(void_args);
    fcache_zzh = fcache_[device_id];
    //printf("runtime kernel arguments 1 %f ~~~~~~~~~~~~~~~!!!!!\n",*(float**)(params_zzh[0]));
    //printf("runtime kernel arguments 2 %f ~~~~~~~~~~~~~~~!!!!!\n",*(float**)(params_zzh[1]));
    //printf("runtime kernel arguments 3 %f ~~~~~~~~~~~~~~~!!!!!\n",*(float*)(params_zzh[2]));
    printf("runtime kernel arguments 4 %f ~~~~~~~~~~~~~~~!!!!!\n",*(float**)(params_zzh[3]));
    /***
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char* msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
         << ")\n";
      std::string cuda = m_->GetSource("");
      if (cuda.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// CUDA Source\n"
           << "// -----------\n"
           << cuda;
      }
      LOG(FATAL) << os.str();
    }***/
  }
  
 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUfunction, kMaxNumGPUs> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

class CUDAPrepGlobalBarrier {
 public:
  CUDAPrepGlobalBarrier(CUDAModuleNode* m, ObjectPtr<Object> sptr) : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const TVMArgs& args, TVMRetValue* rv) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] =
          m_->GetGlobal(device_id, runtime::symbol::tvm_global_barrier_state, sizeof(unsigned));
    }
    CUDA_DRIVER_CALL(cuMemsetD32(pcache_[device_id], 0, 1));
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUdeviceptr, kMaxNumGPUs> pcache_;
};

PackedFunc CUDAModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  if (name == symbol::tvm_prepare_global_barrier) {
    return PackedFunc(CUDAPrepGlobalBarrier(this, sptr_to_self));
  }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.thread_axis_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module CUDAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source) {
  auto n = make_object<CUDAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}

// Load module from module.
Module CUDAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

Module CUDAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_cubin").set_body_typed(CUDAModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_ptx").set_body_typed(CUDAModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cuda").set_body_typed(CUDAModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
