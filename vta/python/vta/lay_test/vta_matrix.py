import os
import tvm
from tvm import te
import vta
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm import rpc
import numpy as np
from vta.testing import simulator
from tvm.topi.utils import get_const_tuple
"""
简单矩阵乘法算子的VTA实现：
1、RPC设置
2、计算部分的定义
    2.1 搞清楚一个算子的计算阶段，计算过程
3、调度部分的定义
"""

# 从 3rdparty/vta-hw/config/vta_config.json 文件载入 VTA 参数
env = vta.get_env()
print(f'env.wgt_scope:{env.wgt_scope}')
# 从操作系统环境中读取 Pynq RPC 主机 IP 地址和端口号
host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))


# 在 Pynq 上配置 bitstream 和运行时系统，
# 以匹配 vta_config.json 文件指定的 VTA 配置。
if env.TARGET in ["pynq", "de10nano"]:
    # 确保使用 RPC=1 编译 TVM
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)

    # 重新配置 JIT runtime
    vta.reconfig_runtime(remote)

    # 用预编译的 VTA bitstream 程序编程 FPGA。
    # 通过将路径传递给 bitstream 文件而不是 None，
    # 您可以使用自定义 bitstream 编程 FPGA。
    vta.program_fpga(remote, bitstream=None)

# 在仿真模式下，在本地托管 RPC 服务器。
elif env.TARGET in ["sim", "tsim"]:
    remote = rpc.LocalSession()
    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")




# compute define
"""
首先描述存在于 main memory 中的输入张量 A 和 B。

其次，需要声明中间张量 A_buf 和 B_buf，它们将存在于 VTA 的 on-chip buffers 中。有了这个额外的计算阶段，就可以显式地分阶段 cached 读和写。

接着，描述了 A_buf 和 B_buf 上的矩阵乘法运算，以产生 product matrix C_buf。

最后的运算是强制转换和复制回 DRAM，到结果张量 C。
"""
"""
数据平铺：
以平铺数据格式描述占位符张量 A和 B，以匹配 VTA 张量核心施加的数据布局要求。
VTA 是围绕 tensor core 设计的，它在激活矩阵和权值矩阵之间执行周期的矩阵-矩阵运算，将结果矩阵添加到累加器矩阵
VTA 的特点之一是，它只支持 DRAM 存储在窄化的 env.inp_dtype 数据类型格式。这使能够减少内存传输的数据占用，但也使能够将宽的累加器数据类型量化为与输入激活数据类型匹配的数据格式。这意味着在神经网络推理的背景下，激活某一层后的输出可以直接被下一层 consumed。
"""


# 输出通道因子 m - 总共 16 x 16 = 256 输出通道
m = 16
# 输入通道因子 n -总计 16x16=256 个输入通道
n = 16
# Batch 因子 o （使用单个 batch 推理）
o = 1
# tiled 据格式的 A 占位符张量
A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
# tiled 据格式的 B 占位符张量
B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
# A copy buffer
A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
# B copy buffer
B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf") 


# Outer input feature reduction axis
ko = te.reduce_axis((0, n), name="ko")
# Inner input feature reduction axis
ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
# Describe the in-VTA matrix multiplication
C_buf = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda bo, co, bi, ci: te.sum(
        A_buf[bo, ko, bi, ki].astype(env.acc_dtype) * B_buf[co, ko, ci, ki].astype(env.acc_dtype),
        axis=[ko, ki],
    ),
    name="C_buf",
)

C = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
)

print(f'C.op:{C.op}\n')
# 调度的实现

"""调度是对原始计算的一组变换，它在不影响正确性的情况下变换计算的实现。这个简单的 VTA 编程教程旨在演示基本的调度变换，将原始的调度映射到 VTA 硬件原语（primitive）。"""

# Let's take a look at the generated schedule
# 在构造了调度后，默认情况下，调度按以下方式计算 C
# 默认调度，不会编译到VTA上，也就是默认在你的CPU上运行
s = te.create_schedule(C.op)
print(f's:{s}\n')
tvm.lower(s, [A, B, C], simple_mode=True).show()

"""
为了获得正确的代码生成，需要应用调度原语和代码注解，将调度变换为可以直接 lower 至 VTA 硬件 intrinsic 的调度。这些包括：

DMA 复制运算，将全局作用域张量复制到局部作用域张量。
Buffer 作用域.首先，设置 buffer 的作用域来告诉 TVM 这些 buffer 将存在于 VTA 的 on-chip SRAM cache 中。
下面，告诉 TVM, A_buf，B_buf，C_buf 将分别存在于 VTA 的 on-chip 输入，权重和累加器（accumulator）内存中


用来做矩阵乘法的张量运算。
"""

# Set the intermediate tensor's scope to VTA's on-chip buffers
s[A_buf].set_scope(env.inp_scope)
s[B_buf].set_scope(env.wgt_scope)
s[C_buf].set_scope(env.acc_scope)

# Move buffer copy into matrix multiply loop
s[A_buf].compute_at(s[C_buf], ko)
s[B_buf].compute_at(s[C_buf], ko)

# Tag the buffer copies with the DMA pragma to insert a DMA transfer
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)
# Let's take a look at the transformed schedule
tvm.lower(s, [A, B, C], simple_mode=True).show()

# 张量化
s[C_buf].reorder(
    ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1], s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki
)
s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

# Let's take a look at the finalized schedule
vta.lower(s, [A, B, C], simple_mode=True).show()

# TVM计算
# Build GEMM VTA kernel
my_gemm = vta.build(s, [A, B, C],
                    target="ext_dev",
                 name="my_gemm")

# Write the compiled module into an object file.
temp = tvm.contrib.utils.tempdir()
my_gemm.save(temp.relpath("gemm.o"))

# Send the executable over RPC
remote.upload(temp.relpath("gemm.o"))

# Load the compiled module
f = remote.load_module("gemm.o")

# 运行函数
"""
编译后的 TVM 函数使用简洁的 C API，可以从代码语言调用。

TVM 在 python 中提供了数组 API 来帮助快速测试和创建原型。数组 API 基于 DLPac 标准。

首先创建远程上下文（remote context）（用于在 Pynq 上远程执行）。

然后 tvm.nd.array() 相应地格式化数据。

f() 运行实际的计算。

numpy() 以可解释的格式将结果数组复制回来。
"""

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the A and B arrays randomly in the int range of (-128, 128]
A_orig = np.random.randint(-128, 128, size=(o * env.BATCH, n * env.BLOCK_IN)).astype(A.dtype)
B_orig = np.random.randint(-128, 128, size=(m * env.BLOCK_OUT, n * env.BLOCK_IN)).astype(B.dtype)

# Apply packing to the A and B arrays from a 2D to a 4D packed layout
A_packed = A_orig.reshape(o, env.BATCH, n, env.BLOCK_IN).transpose((0, 2, 1, 3))
B_packed = B_orig.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
A_nd = tvm.nd.array(A_packed, ctx)
B_nd = tvm.nd.array(B_packed, ctx)
C_nd = tvm.nd.array(np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(C.dtype), ctx)

# Clear stats
if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()

# Invoke the module to perform the computation
f(A_nd, B_nd, C_nd)


# Compute reference result with numpy
C_ref = np.dot(A_orig.astype(env.acc_dtype),
               B_orig.T.astype(env.acc_dtype)).astype(C.dtype)
C_ref = C_ref.reshape(o,
                      env.BATCH,
                      m,
                      env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(C_ref, C_nd.numpy())

# Print stats
if env.TARGET in ["sim", "tsim"]:
    sim_stats = simulator.stats()
    print("Execution statistics:")
    for k, v in sim_stats.items():
        print(f"\t{k:<16}: {v:>16}")

print("Successful matrix multiply test!")

