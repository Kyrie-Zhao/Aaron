# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Makefile Example to deploy TVM modules.
#TVM_ROOT=$(shell cd ../..; pwd)
TVM_ROOT=$(shell cd /home/lenovo/Desktop/zzh/phd/new_tenset/tenset; pwd)
$(info $$var is [${TVM_ROOT}])
#TVM_ROOT=$(/home/lenovo/Desktop/zzh/phd/new_tenset/tenset)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I/usr/local/cuda-10.2/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>


PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -lpthread -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda

.PHONY: clean all

all: lib/libtvm_runtime_pack.o lib/cpp_deploy_pack

.PHONY: lib/libtvm_runtime_pack.o
lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc 
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

.PHONY: lib/cpp_deploy_pack
lib/cpp_deploy_pack: cpp_deploy_conv.cc lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)