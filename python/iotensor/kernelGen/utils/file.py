from tvm.contrib import util

def SaveModule(libData,jsonData,paramsData):
    temp = util.tempdir()
    path_lib = temp.relpath("deploy_lib.tar")
    libData.export_library(path_lib)
    with open(temp.relpath(jsonData), "w") as fo:
        fo.write(graph)
    with open(temp.relpath(paramsData), "wb") as fo:
        fo.write(relay.save_param_dict(paramsData))
        print(temp.listdir())

def LoadModule(jsonfile,path_lib,paramsfile):
    # load the module back.
    loaded_json = open(temp.relpath(jsonfile)).read()
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(temp.relpath(paramsfile), "rb").read())
    return [loaded_json,loaded_lib,loaded_params]