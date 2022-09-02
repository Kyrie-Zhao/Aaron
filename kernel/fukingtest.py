import re

import tvm
from tvm import te
import numpy as np
from tvm import topi
from tvm.contrib.nvcc import have_fp16, have_int8, have_bf16
import tvm.testing
import pytest

tx = te.thread_axis("threadIdx.x")
bx = te.thread_axis("blockIdx.x")
def test_try_unaligned_vector_load():
    def get_compute(N, C_N, offset):
        A = te.placeholder((N,), name="A", dtype="float16")
        C = te.compute((C_N,), lambda i: A[i + offset], name="C")
        return N, C_N, A, C

    def get_compute_unaligned():
        return get_compute(3, 2, 1)

    def get_compute_aligned():
        return get_compute(4, 2, 2)

    def build(A, C, N, C_N):
        s = te.create_schedule(C.op)
        oi, ii = s[C].split(C.op.axis[0], factor=2)
        s[C].bind(oi, te.thread_axis("threadIdx.x"))
        s[C].vectorize(ii)  # BUG: misalignment

        tgt = tvm.target.Target("cuda")
        dev = tvm.context(tgt.kind.name, 0)
        f = tvm.build(s, [A, C], tgt, name="foo")
        print(type(f.imported_modules[0]))
        kernel_source = f.imported_modules[0].get_source()

        a_data = np.arange(0, N).astype(A.dtype)
        a = tvm.nd.array(a_data, dev)
        c = tvm.nd.array(np.zeros(C_N, dtype=C.dtype), dev)
        f(a, c)

        return kernel_source

    N, C_N, A, C = get_compute_unaligned()
    kernel_source = build(A, C, N, C_N)
    # (uint1*)(A + (1)) is invalid
    print("nimabi")
    print(kernel_source)
test_try_unaligned_vector_load()