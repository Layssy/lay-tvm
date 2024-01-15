import os
import tvm
from tvm import te
import vta
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
from vta.testing import simulator

env = vta.get_env()

host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))


if env.TARGET in ["pynq", "de10nano"]:
    # 确保使用 RPC=1 编译 TVM
    assert tvm.runtime.enabled("rpc")
    remote = tvm.rpc.connect(host, port)
    # 重新配置 JIT runtime
    vta.reconfig_runtime(remote)

    # 用预编译的 VTA bitstream 编程 FPGA。
    # 通过将 path 传递给 bitstream 文件而不是 None，
    # 您可以使用自定义 bitstream 编程 FPGA。
    vta.program_fpga(remote, bitstream=None)

# 在仿真模式中，在本地托管 RPC 服务器。
elif env.TARGET in ("sim", "tsim", "intelfocl"):
    remote = tvm.rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")


# print(f'env:{env}\n')
m = 16
n = 1024
A_orig = np.random.randint(-128, 128, size=(m, n)).astype(env.acc_dtype)
B_orig = np.random.randint(-128, 128, size=(m, n)).astype(env.acc_dtype)
A_packed = A_orig.reshape(m//env.BATCH, env.BATCH, n//env.BLOCK_OUT, env.BLOCK_OUT).transpose((0, 2, 1, 3))
B_packed = B_orig.reshape(m//env.BATCH, env.BATCH, n//env.BLOCK_OUT, env.BLOCK_OUT).transpose((0, 2, 1, 3))
print(f'A_packed:{A_packed.shape}\n')
print(f'B_packed:{type(B_packed)}\n')

# 输出通道因子 m -总共 64 x 16 = 1024 输出通道
_m = n//env.BLOCK_OUT
# Batch 因子 o - 总共 16 x 1 = 1
_o = m//env.BATCH
# VTA 向量数据 shape
shape = (_o, _m, env.BATCH, env.BLOCK_OUT)

# env.acc_dtype, env.inp_dtype, env.out_dtype, env.wgt_dtype--> int32 int8 int8 int8
# print(env.acc_dtype, env.inp_dtype, env.out_dtype, env.wgt_dtype)

# 平铺 A, B 占位符张量数据
# A,B 的类型tvm.te.tensor.Tensor'
A = te.placeholder(shape, name="A", dtype=env.acc_dtype)
B = te.placeholder(shape, name="B", dtype=env.acc_dtype)

# 硬件加速器的特点之一是，必须对 on-chip memory 进行显式管理。这意味着需要描述中间张量 A_buf 和 B_buf，它们可以具有与原始占位符张量 A 和 B 不同的内存作用域
# 稍后在调度阶段，可以告诉编译器 A_buf 和 B_buf 将存在于 VTA 的 on-chip buffer（SRAM）中，而 A 和 B 将存在于 main memory（DRAM）中。将 A_buf 和 B_buf 描述为恒等函数计算的运算结果
# A copy buffer
A_buf = te.compute(shape, lambda *i: A[i], "A_buf")
# B copy buffer
B_buf = te.compute(shape, lambda *i: B[i], "B_buf")
print(f'A_buf:{A_buf}')
print(f'A_buf:{type(A_buf)}')
# 描述 VTA 中的 ALU 加法
fcompute = lambda *i: A_buf[i].astype(env.acc_dtype) + B_buf[i].astype(env.acc_dtype)
C_buf = te.compute(shape, fcompute, name="C_buf")
# 转换为输出类型，并发送到 main memory
fcompute = lambda *i: C_buf[i].astype(env.inp_dtype)
C = te.compute(shape, fcompute, name="C")
# print(f'C:{C}')

#调度计算

func_name = "add"
te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": func_name})
#print(f'te_func:{te_func}\n')
# print(f'te_func_type:{type(te_func)}\n')
MyModule = IRModule({func_name: te_func})
sch = tvm.tir.Schedule(MyModule)
print("sch = tvm.tir.Schedule(MyModule)")
sch.mod.show()

# 定义调度
#虽然此调度是合理的，但它不会编译到 VTA。为了获得正确的代码生成（code generation），需要应用调度原语（scheduling primitives）和代码注解（code annotation），将调度变换为可以直接 lower 到 VTA 硬件 intrinsics。
# DMA copy 运算将把全局作用域的张量复制到局部作用域的张量。
# 执行向量加法的向量 ALU 运算。
# 可以用来测试compute阶段是否正确，因为不会编译到VTA上
s = te.create_schedule(C.op)
print(f's:{s}\n')
#simulator.clear_stats()
#cost = evaluator(a, b, c)
#stats = simulator.stats()

#Buffer 作用域
# 首先，设置复制 buffer 的作用域，以指示 TVM 这些中间张量将存储在 VTA 的 on-chip SRAM buffer 中。
# print(f'env.acc_scope:{env.acc_scope}\n')# local.acc_buffer VTA \ 本地cpu的accumulator buffer  VTA 的 on-chip accumulator buffer 中，该 buffer 作为 VTA 的通用寄存器（register）文件
s[A_buf].set_scope(env.acc_scope)
s[B_buf].set_scope(env.acc_scope)
s[C_buf].set_scope(env.acc_scope)
# print(f's[A_buf].scope{s[A_buf].scope}')
# print(f's[B_buf].scope{s[B_buf].scope}')
print("after set set_scope tvm.lower\n")
tvm.lower(s, [A, B, C], simple_mode=True).show()
# 使用了pragma操作，将一个关于 A_buf 的 DMA 操作的提示应用于调度器。这可能是在指导调度器生成特定于 DMA 操作的优化或生成代码。
# 一般情况下，pragma语句在TVM中用于向调度器传递额外的信息，以指导生成的代码的行为。这些信息可能涉及内存布局、数据传输、并行化策略等方面。
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)
print("after set pragma tvm.lower\n")
tvm.lower(s, [A, B, C], simple_mode=True).show()

s[C_buf].pragma(C_buf.op.axis[0], env.alu)
# 查看最终的 schedule
print("final tvm.lower\n")
tvm.lower(s, [A, B, C], simple_mode=True).show()
# ctx = tvm.target.Target("ext_dev", host=env.target_host)
target = "ext_dev"
my_vadd = vta.build(s, [A, B, C], target=target, name="my_vadd")

print(f'my_vadd:{my_vadd}\n')
# print(f'my_vadd:{type(my_vadd)}\n') # my_vadd:<class 'tvm.driver.build_module.OperatorModule'>

temp = tvm.contrib.utils.tempdir()
# print(temp)
my_vadd.save(temp.relpath("vadd.o"))
remote.upload(temp.relpath("vadd.o"))
f = remote.load_module("vadd.o")

#  This is the TVM execution context obtained earlier, indicating the target device (e.g., GPU, CPU) where you want to perform computations.
ctx = remote.ext_dev(0)
# print(f'ctx:{type(ctx)}\n')

from tvm.topi.utils import get_const_tuple
# This is a common step when working with TVM to pass data to the target device before performing computations.
A_nd = tvm.nd.array(A_packed, ctx)
B_nd = tvm.nd.array(B_packed, ctx)
C_nd = tvm.nd.empty(get_const_tuple(C.shape), C.dtype, ctx)

# 通过调用好的f算子，来执行A，B，C的加法操作
f(A_nd, B_nd, C_nd)

# 对原数据进行处理，然后和算子得出的结果进行比较
C_ref = (A_orig.astype(env.acc_dtype) + B_orig.astype(env.acc_dtype)).astype(C.dtype)
C_ref = C_ref.reshape(m//env.BATCH, env.BATCH, n//env.BLOCK_OUT, env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(C_ref, C_nd.numpy())
print("ALU 加法测试成功！")

time_f = f.time_evaluator(f.entry_name, ctx, number=20)
if env.TARGET in ["sim", "tsim"]:
    print("1")
    # Check if we're in local RPC mode (allows us to rebuild the
    # runtime on the fly when varying the VTA designs)
    local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
    if local_rpc:
        print("1.1")
        if env.TARGET == "sim":
            print("1.1.1")
            remote.get_function("vta.simulator.profiler_clear")()
        else:
            print("1.1.2")
            remote.get_function("vta.tsim.profiler_clear")()
        cost = time_f(A_nd, B_nd, C_nd)
        if env.TARGET == "sim":
            print("1.1.3")
            stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
        else:
            print("1.1.4")
            stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
    else:
        print("1.2")
        simulator.clear_stats()
        cost = time_f(A_nd, B_nd, C_nd)
        stats = simulator.stats()
        print(f'cost:{cost}\nstats:{stats}')
        
else:
    print("2")
    cost = time_f(A_nd, B_nd, C_nd)
