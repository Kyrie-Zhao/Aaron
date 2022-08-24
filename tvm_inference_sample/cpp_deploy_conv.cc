#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cuda.h>

#include <cstdio>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <string>
#include <iterator>
#include <algorithm>
using namespace std;
template <class Type>

#define N 9999

Type stringToNum(const string& str)
{
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}


void DeployGraphRuntime() {
  
  

  constexpr int dtype_code= kDLFloat;//2U;
  constexpr int dtype_bits=32;
  constexpr int dtype_lines=1;
  constexpr int device_type= 2;
  constexpr int device_id=0;
  int ndim=4;
  int64_t in_shape[4]={16,1,28,28};
  int64_t out_shape[4]={16,32,26,26};

  DLTensor* DLTX=nullptr;
  DLTensor* DLTY=nullptr;

  TVMArrayAlloc(in_shape,ndim,dtype_code,dtype_bits,dtype_lines,device_type,device_id,&DLTX);
  TVMArrayAlloc(out_shape,ndim,dtype_code,dtype_bits,dtype_lines,device_type,device_id,&DLTY);

  float img[12544];
  float rslt[346112];
  //std::vector<float> input(16 * 28 * 28);
  for (size_t i = 0; i < 3*28*28; i++)
      img[i] = 0.1;
  ifstream in("/home/aiteam/tiwang/data.txt");
  //int image[784];
  //string s;
  int image_index=0;
  /*
  while(getline(in,s))
  {
      image[i]=stringToNum<int>(s);
      ++i;
  }*/
  bool enabled = tvm::runtime::RuntimeEnabled("cuda");
  if (!enabled) 
  {
      LOG(INFO) << "Skip heterogeneous test because cuda is not enabled."<< "\n";
      return;
  }

  LOG(INFO) << "Running graph runtime...";
  // load in the library
  DLContext ctx{kDLGPU, 0};
  cout << typeid( ctx ).name() << endl; 
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("/home/lenovo/Desktop/zzh/phd/new_tenset/tenset/apps/howto_deploy/conv.so");
  cout << typeid( mod_factory ).name() << endl; 
  //std::cout << mod_factory << std::endl;
  // create the graph runtime module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
  cout << typeid( gmod ).name() << endl; 
  cout << "context set done /n";
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  cout << typeid( set_input ).name() << endl;
  cout << "set input done /n";
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  cout << "get output done /n";
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
  cout << "finish running /n";


  // Use the C++ API
  //while(getline(in,s))
  //{
   //   if(image_index%28==0)
    //      printf("\n");
      //static_cast<float*>(x->data)[image_index]=((float)stringToNum<int>(s))/255;
     // img[image_index]=((float)stringToNum<int>(s))/255;
      
     // int a=stringToNum<int>(s);
      //printf("%4d",a);
      //image_index++;
  //}
  TVMArrayCopyFromBytes(DLTX,&img[0],16*28*28*4);
  // set the right input
  set_input("x", DLTX);
  // tvm::runtime::PackedFunc run = mod->GetFunction("run");
  // run the code
  //#pragma omp parallel for
  for (int i=0;i<10;i++){
    printf("Done.\n");
    //TVMArrayCopyFromBytes(DLTY,&rslt[0],346112);
    run();
    get_output(0,DLTY);
    TVMArrayCopyToBytes(DLTY,&rslt[0],346112);
  }
  LOG(INFO) << "run success";
  for (int tmp=0;tmp<10;tmp++){
    LOG(INFO)<<rslt[tmp];
  }
  // get the output
  //get_output(0, DLTY);
  //TVMArrayCopyToBytes(DLTY,&rslt[0],10*sizeof(float));
  
  //for(int i=0;i<10;++i)
  //{
   //   LOG(INFO)<<rslt[i];
    //  //LOG(INFO)<<static_cast<float*>(y->data)[i];
  //}
}



int main(void) {
  //DeploySingleOp();
  DeployGraphRuntime();
  return 0;
}