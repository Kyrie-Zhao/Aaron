import numpy as np
import ios
import tvm
import time
import json
import ios.tvm_utils
from tvm import relay, auto_scheduler
#import tvm.relay.testing
from tvm.contrib import graph_runtime
import sys
from ios.models import inception_v3_block,inception_v3
import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda-10.2/bin"
graph = inception_v3_block()

#print(enumerate(graph.blocks))
#graph = ios.optimize(graph, batch_size=1, opt_type='dp_parallel_merge', compute_weight=False,verbose=True)
#graph.init_weights()
#print("Graph Optimization")
#ios.draw(graph, fname=f'draw/opt_cpuAlex.png', label=f'optimized_graph Graph')
#sys.exit(1)
network = "inceptionBlock"
#background = "Heavy(8_8_8)-NI"
background = "iso-NI"
batch_size = 1
layout = "NCHW"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s-%s.json" % (network, layout, batch_size, target.kind.name,background)
lib_name = "%s-%s-%s-lib.so" % (network, layout,background)
input_shape = (1,192, 35, 35)

mod, params = ios.tvm_utils.graph2relay(graph, batch_size=1)
#print(mod)
#sys.exit(1)
a = -1
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
for idx,task in enumerate(tasks):
    print(task.workload_key)
    a += 1
    if a==5:
        print("task X...")
        print("====== %d %s ===="%(idx,task.workload_key))
        log_file="best_iso-IOS-damn.json"
        sch, args = task.apply_best(log_file)
        print(sch)
        print(args)
        print(tvm.lower(sch, args, simple_mode=True))
        #print((task.compute_dag))
        print("Equivalent python schedule:")
        
        print(task.print_best(log_file, print_mode="schedule"))
        sys.exit(1)
        print("CUDA source code:")
        cuda_source = task.print_best(log_file, print_mode="cuda")
        with open("cudasource_task5.cu","w") as f:
            json.dump(cuda_source,f)
            f.close()

        print(type(task.print_best(log_file, print_mode="cuda")))
        print("end")
        sys.exit(1)
sys.exit(1) 
print("Begin tuning...")
"""measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)   

tuner = auto_scheduler.TaskScheduler(tasks, task_weights)#,load_model_file="xgb.pkl")
tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)

tuner.tune(tune_option)
"""
#log_file="best_iso-IOS-damn.json"
#sch, args = task.apply_best(log_file)
#print(tvm.lower(sch, args, simple_mode=True))
#print("task X...")

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
        print(lib)
        lib.export_library(lib_name)
            
        ctx = tvm.context(str(target), 0)
        lib: tvm.runtime.Module = tvm.runtime.load_module(lib_name)
        print("loaded...")
        gmod = graph_runtime.GraphModule(lib["default"](ctx))
# use the graph module.
print(lib.get_source())
sys.exit(1)
#gmod.set_input("input",input_shape)
time_tmp=[]
for i in range(0,1000):
    time_start = time.perf_counter()
    _ = gmod.run()
    time_end = time.perf_counter()
    time_tmp.append(time_end-time_start)
time_tmp.sort()
#time_tmp = [x*1000 for x in time_tmp]
print(time_tmp)
print(f'Code+Parallel Inference: {(np.mean(time_tmp[:-20])*1000):.3f} ms')

sys.exit(1)
graph.sequential_schedule()

seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=1024, profile_stage=True)
print(np.mean(seq_latency[0]))
#print(np.mean(stage_latency))
"""print(optimized_graph.blocks[0].stages)
print(optimized_graph.parts)"""
