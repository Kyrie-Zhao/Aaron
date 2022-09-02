import tvm
from tvm import relay
import time
import numpy as np
#from convNative import Net
from tvm.contrib.download import download_testdata
import tvm
from tvm import relay, auto_scheduler
            #import tvm.relay.testing
from tvm.contrib import graph_runtime
# PyTorch imports
import torch
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self):
      super(Conv, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)

    def forward(self, x):
      x = self.conv1(x)
      return x

model = Conv()
print(list(model.parameters()))

input_shape = [16, 1, 28, 28]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
input_name = "input0"
shape_list = [(input_name,input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(params["conv1.bias"].shape)
print(params["conv1.weight"].shape)


network = "conv"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-cuda-2000.json" % (network)
#log_file = "best_iso-IOS.json"
lib_name = "%s-2000-lib.so" % (network)
input_shape = (16, 1, 28, 28)
#input_shape = (1,192, 35, 35)

#tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
"""
print(len(tasks))
print(tasks[0].compute_dag)
cuda_source = tasks[0].print_best(log_file, print_mode="cuda")
with open("cudasource_conv.cu","w") as f:
  json.dump(cuda_source,f)
  f.close()
print(tasks[0].print_best(log_file))
sch, args = tasks[0].apply_best(log_file)
print(sch)
print(args)
print(tvm.lower(sch, args, simple_mode=True))"""


with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        #mod, lib, params= relay.build(mod, target="c", params=params)
        lib1 = relay.build(mod,tvm.target.Target('c --link-params --runtime=c '), params=params)
        #lib1.save('c', fmt='c')
        #tvm.module
        #lib1.export_library("compiled_model.tar")
        #libmod = lib1.get_lib()
        #print(libmod)
        #print(lib1.get_params()) 
        print("fuk")     
        #print(graph)
        print(type(lib))
        #print(params)
        
        #source_code = lib.imported_modules[0].get_source()
        print(str(lib.get_source()))
        #print(source_code)
        #source_code = lib.imported_modules[0].get_source()
        #print(source_code)
        #with open("cudasource_inception.cu","w") as f:
        #  json.dump(source_code,f)
        #  f.close()
        #print(source_code)
    