#include <tvm/runtime/crt/module.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_layout_transform_3(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_layout_transform_2(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_conv2d_NCHWc_add(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t _lookup_linked_param(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)fused_layout_transform_3,
    (TVMBackendPackedCFunc)fused_layout_transform_2,
    (TVMBackendPackedCFunc)fused_nn_contrib_conv2d_NCHWc_add,
    (TVMBackendPackedCFunc)_lookup_linked_param,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\004fused_layout_transform_3\000fused_layout_transform_2\000fused_nn_contrib_conv2d_NCHWc_add\000_lookup_linked_param\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
;