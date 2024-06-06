import os
import tvm
from tvm import te


# Baseline
def make_conv1d_cpu_scheduler_baseline(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, M + N - 1), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )

    s = te.create_schedule(B.op)
    return s, A, W, B


def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    B = te.placeholder((N,), name="B")

    padding_size = N - 1

    A_padded = te.compute(
                        (M + 2 * padding_size,),
                        lambda n: tvm.tir.if_then_else(
                            tvm.tir.any(n < padding_size, n >= (M + padding_size)),
                            tvm.tir.const(0.0, "float32"),
                            A[n - padding_size]
                        ),
                        name="A_padded",
                    )
    
    k = te.reduce_axis((0, N), "k")

    C = te.compute(
        (M + N - 1,),
        lambda n: te.sum(A_padded[n + padding_size - k] * B[k], axis=k),
        name="C",
    )

    s = te.create_schedule(C.op)
    n = C.op.axis[0]

    xo, xi = s[C].split(n, 16)
    ko, ki = s[C].split(k, 16)
    s[C].reorder(xo, ko, ki, xi)
    s[C].unroll(ki)
    s[C].vectorize(xi)

    return s, A, B, C


# Baseline
def make_conv1d_gpu_scheduler_baseline(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, M + N - 1), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )

    s = te.create_schedule(B.op)

    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")

    n = B.op.axis[0]
    bn, tn = s[B].split(n, factor=32)

    s[B].bind(bn, block_x)
    s[B].bind(tn, thread_x)

    return s, A, W, B


def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    padding_size = N - 1

    A_padded = te.compute(
        (M + 2 * padding_size,),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.any(n < padding_size, n >= M + padding_size),
            tvm.tir.const(0.0, "float32"),
            A[n - padding_size]
        ),
        name="A_padded"
    )

    k = te.reduce_axis((0, N), "k")
    C = te.compute(
        (M + N - 1,),
        lambda n: te.sum(A_padded[n + padding_size - k] * W[k], axis=k),
        name="C"
    )

    s = te.create_schedule(C.op)

    factor = 64
    xo, xi = s[C].split(s[C].op.axis[0], factor=factor)
    ko, ki = s[C].split(k, factor=factor)

    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")

    #compute A_padded inside the xi loop of C
    s[A_padded].compute_at(s[C], xi)

    s[C].bind(xo, block_x)
    s[C].bind(xi, thread_x)

    s[C].reorder(xo, ko, xi, ki)
    s[C].unroll(ki)

    return s, A, W, C


# Baseline
def make_gemm_gpu_scheduler_baseline(M, K, N):
    # A = te.placeholder((M, K), name="A")
    # B = te.placeholder((K, N), name="B")

    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    # Default schedule
    s = te.create_schedule(C.op)

    # the i-th block is indexed by blockIdx.x.
    # the number of threads in each block is blockDim.x
    # and the i-th thread within a block is indexed by threadIdx.x
    # overall index of a thread can be calculated as
    # 𝑖=blockIdx.x×blockDim.x+threadIdx.x
    block_x = te.thread_axis("blockIdx.y")
    block_y = te.thread_axis("blockIdx.x")

    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    s[C].bind(y, block_y)
    s[C].bind(x, block_x)

    return s, A, B, C


def make_gemm_gpu_scheduler(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")

    k = te.reduce_axis((0, K), "k")

    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = te.create_schedule(C.op)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    xo, xi = s[C].split(s[C].op.axis[0], factor=8)
    yo, yi = s[C].split(s[C].op.axis[1], factor=8)

    s[C].bind(xo, block_x)
    s[C].bind(yo, block_y)
    s[C].bind(xi, thread_x)
    s[C].bind(yi, thread_y)

    k = s[C].op.reduce_axis[0]
    ko, ki = s[C].split(k, factor=8)
    s[C].reorder(xo, yo, ko, xi, yi, ki)

    s[C].unroll(ko)

    return s, A, B, C


# Baseline
def make_dwsp_conv2d_gpu_scheduler_baseline(B, C, H, W, K):
    pad = (K - 1) // 2

    inp = te.placeholder((B, C, H, W), dtype="float32", name="input")
    ker = te.placeholder((C, 1, K, K), dtype="float32", name="kernel")

    # reduction axes for the kernel dimensions
    ry = te.reduce_axis((0, K), name='ry')
    rx = te.reduce_axis((0, K), name='rx')

    output = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            inp[b, c, h + ry - pad, w + rx - pad] * ker[c, 0, ry, rx],
            axis=[ry, rx],
            where=(h + ry >= pad) & (h + ry < H + pad) & (w + rx >= pad) & (w + rx < W + pad)),
        name="output"
    )

    s = te.create_schedule(output.op)

    b, c, h, w = s[output].op.axis

    ho, hi = s[output].split(h, factor=32)
    wo, wi = s[output].split(w, factor=32)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    s[output].bind(ho, block_y)
    s[output].bind(wo, block_x)
    s[output].bind(hi, thread_y)
    s[output].bind(wi, thread_x)

    return s, inp, ker, output


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    pad = (K - 1) // 2

    inp = te.placeholder((B, C, H, W), dtype="float32", name="input")
    ker = te.placeholder((C, 1, K, K), dtype="float32", name="kernel")

    # reduction axes for the kernel dimensions
    ry = te.reduce_axis((0, K), name='ry')
    rx = te.reduce_axis((0, K), name='rx')

    output = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            inp[b, c, h + ry - pad, w + rx - pad] * ker[c, 0, ry, rx],
            axis=[ry, rx],
            where=(h + ry >= pad) & (h + ry < H + pad) & (w + rx >= pad) & (w + rx < W + pad)),
        name="output"
    )

    s = te.create_schedule(output.op)

    # Axis setup
    b, c, h, w = s[output].op.axis
    ry, rx = s[output].op.reduce_axis

    # fusing batch and channel
    bc = s[output].fuse(b, c)

    #blocking and threading
    ho, hi = s[output].split(h, factor=32)
    wo, wi = s[output].split(w, factor=32)

    #reorder to interleve reduction and spatial axis
    s[output].reorder(bc, ho, wo, ry, rx, hi, wi)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    s[output].bind(bc, block_y)
    s[output].bind(ho, block_x)
    s[output].bind(hi, thread_y)
    s[output].bind(wi, thread_x)

    wi, vec = s[output].split(wi, factor=4)
    s[output].vectorize(vec)

    s[output].unroll(ry)
    s[output].unroll(rx)

    return s, inp, ker, output
