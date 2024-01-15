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

"""Testing topi upsample operator for VTA"""

import json
import os

import pytest
import numpy as np
from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.contrib import utils
from tvm.contrib.pickle_memoize import memoize
from tvm import topi
import tvm.topi.testing
import vta
from vta.top.vta_upsample import upsampling
from vta.top.vta_upsample import schedule_upsampling_packed
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator
import torch
import torch.nn as nn

# Define your workload namedtuple (similar to Conv2DWorkload)
Workload = namedtuple(
    "UpsampleWorkload",
    [
        "batch",
        "channels",
        "in_height",
        "in_width",
        "scale_h",
        "scale_w",
        "layout",
        "method",
        "align_corners",
        "output_shape",
    ],
)



# Get batch info from env
env = vta.get_env()

# Example 1
wl1 = Workload(
    batch=env.BATCH,
    channels= env.BLOCK_IN,
    in_height=2,
    in_width=2,
    scale_h=2,
    scale_w=2,
    layout="NCHW",
    method="nearest_neighbor",
    align_corners=True,
    output_shape=None,
)

# Example 2
wl2 = Workload(
    batch=env.BATCH,
    channels=env.BLOCK_IN,
    in_height=24,
    in_width=24,
    scale_h=3,
    scale_w=3,
    layout="NCHW",
    method="nearest_neighbor",
    align_corners=False,
    output_shape=None,
)

# Example 3
wl3 = Workload(
    batch=env.BATCH,
    channels=env.BLOCK_IN,
    in_height=16,
    in_width=16,
    scale_h=2,
    scale_w=1,
    layout="NCHW",
    method="nearest_neighbor",
    align_corners=False,
    output_shape=None,
)

# Create a list of UpsampleWorkload instances
upsample_workloads = [wl1, wl2, wl3]

# # ResNet18 workloads
# resnet_wkls = [
#     # Workloads of resnet18 on imagenet
#     # ('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
#     ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C8", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C9", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C10", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C11", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
# ]

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x



def run_upsampling(env, remote, wl, target, check_correctness=True):
    # Define the VTA target
    vta_target = env.target
    # print(f'vta_target:{env.target_host}\n')
    # Define input and output shapes based on the workload
    batch, channels, in_height, in_width = wl.batch, wl.channels, wl.in_height, wl.in_width
    data_shape = (batch, channels, in_height, in_width)
    data_shape1 =  (batch // env.BATCH, channels //  env.BLOCK_IN, in_height, in_width , env.BATCH,  env.BLOCK_IN)

    # Define the input tensor using the input shape
    # env.inp_dtype
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    # data1 = te.placeholder(data_shape1, name="data", dtype=env.acc_dtype)
    data1 = te.placeholder(data_shape1, name="data", dtype=env.acc_dtype)

    # Call the upsampling computation function with the input tensor and workload parameters
    with target:
        layout = "NCHW%dn%dc" % (env.BATCH,  env.BLOCK_IN)
        output = upsampling(data1, wl.scale_h, wl.scale_w,
                        layout, wl.method, wl.align_corners,
                        wl.output_shape)
        # output = topi.right_shift(output, 8)
        # output = my_clip(output, 0, (1 << env.OUT_WIDTH - 1) - 1)
        # output = topi.cast(output, env.out_dtype)
        # print(f'test_output:{output.op}')
        # output:Tensor(shape=[1, 4, 32, 16], op.name=resize)
         
    # Build the module and upload it to the remote
    with target:
        # output = topi.cast(output, env.out_dtype)
        s = schedule_upsampling_packed([output])
        if "vta" in target.keys:
            with vta.build_config(debug_flag=True, disabled_pass={"tir.CommonSubexprElimTIR"}):
                mod = vta.build(
                    s, 
                    [data1, output], 
                    target = tvm.target.Target(target, host=env.target_host),
                    name ="upsample")
                vta.lower(s, [data1, output], simple_mode=True).show()
                # tvm.lower(s, 
                #     [data, output],  name ="upsampling").show()
        else:
            mod = tvm.build(
                s, 
                [data1, output], 
                target = tvm.target.Target(target, host=env.target_host), 
                name="upsample")

    temp = utils.tempdir()
    mod.save(temp.relpath("upsampling.o"))
    remote.upload(temp.relpath("upsampling.o"))
    f = remote.load_module("upsampling.o")
    dev = remote.ext_dev(0)
    remote.device(str(target))
    samples = 4
    # #计算时间的测试
    time_f = f.time_evaluator(f.entry_name, dev, number=samples)
    
    # 系统的torch自带的
    data_min, data_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
    input_shape = (wl.batch, wl.channels, wl.in_width, wl.in_height)
    # input_shape = (wl.batch // env.BATCH, wl.channels// 2,  wl.in_height, wl.in_width, env.BATCH, 2)
    data_np = np.random.randint(data_min, data_max, size=input_shape).astype(np.float32)
    out_height = int(wl.in_height * wl.scale_h)
    out_width = int(wl.in_width * wl.scale_w)
    # # 将NumPy数组转换为PyTorch张量
    data_tensor = torch.from_numpy(data_np)
    # 使用torch.nn.functional.upsample进行上采样
    res_upsample = nn.Upsample(scale_factor=(wl.scale_h, wl.scale_w), mode='nearest')
    upsampled_tensor = res_upsample(data_tensor)
    res_ref = upsampled_tensor.numpy()
    # res_ref = res_ref >> 8
    # res_ref = np.clip(res_ref, 0, (1 << env.OUT_WIDTH - 1) - 1)
    # res_ref = res_ref.astype(env.out_dtype)
    
    # # # 自己写的计算
    input_shape = (wl.batch // env.BATCH,env.BATCH, wl.channels//  env.BLOCK_IN,  env.BLOCK_IN,  wl.in_height, wl.in_width)
    # # data_np = np.random.randint(data_min, data_max + 1, size=input_shape).astype(np.float32)
    data_np = data_np.reshape(input_shape).transpose((0, 2, 4, 5, 1, 3))
    data_np = data_np.astype(data1.dtype)
    # print(f'after_data_np:{data_np}')
    output_shape = (wl.batch // env.BATCH, wl.channels //  env.BLOCK_IN, int(wl.in_height * wl.scale_h), int(wl.in_width * wl.scale_w) , env.BATCH,  env.BLOCK_IN)
    # (batch, channels, int(in_height * wl.scale_h), int(in_width * wl.scale_w))
    output_np = np.zeros(topi.utils.get_const_tuple(output.shape)).astype(output.dtype)
    # np.empty(output_shape, dtype=env.acc_dtype)

    # # # Transfer data to the device
    data_arr = tvm.nd.array(data_np, dev)
    output_arr = tvm.nd.array(output_np, dev)
    # tvm.nd.array(res_np, dev)

    # # # # Run the upsampling function
    f(data_arr, output_arr)

    # # # Retrieve the results
    output_np = output_arr.asnumpy()
    # print(f'output_np_before:{output_np}')
    output_np = output_np.transpose(
                (0, 4, 1, 5, 2, 3)).reshape(wl.batch, wl.channels,int(wl.in_height * wl.scale_h), int(wl.in_width * wl.scale_w))
    # print(f'output_np:{output_np.shape}')
    if env.TARGET in ["sim", "tsim"]:
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            if env.TARGET == "sim":
                remote.get_function("vta.simulator.profiler_clear")()
            else:
                remote.get_function("vta.tsim.profiler_clear")()
            cost = time_f(data_arr, output_arr)
            # time_f(data_arr, kernel_arr, res_arr)
            if env.TARGET == "sim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            else:
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            cost = time_f(data_arr, output_arr)
            # time_f(data_arr, kernel_arr, res_arr)
            stats = simulator.stats()
    else:
        cost = time_f(data_arr, kernel_arr, res_arr)
    correct = True
    if check_correctness:
        output_np = output_np.astype(env.out_dtype)
        # print(f'output_np:{output_np}')
        # print(f'output_np_shape:{output_np.shape}\noutput_np:{output_np}\n')
        # print(f'res_ref:{res_ref}\n')
        # print(f'res_ref_shape:{res_ref.shape}\nres_ref:{res_ref}\n')
        for i in range(output_np.shape[0]):
            for j in range(output_np.shape[1]):
                for m in range(output_np.shape[2]):
                    for n in range(output_np.shape[3]):
                        if output_np[i][j][m][n] == res_ref[i][j][m][n]:
                            pass
                        else:
                            correct = False
                            print(f'false:_i:{i} j:{j} m:{m} n:{n}')
                            # print(f'output_np[i][j][m][n]:{output_np[i][j][m][n]}\nres_ref[i][j][m][n]:{res_ref[i][j][m][n]}')
                            break 
                    if not correct:
                        break
                if not correct:
                    break    
            if not correct:
                break    

                
        # correct = np.array_equal(output_arr, res_ref)
    status = "PASSED" if correct else "FAILED"
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"
    simulator.clear_stats()
    cost = time_f(data_arr, output_arr)
    stats = simulator.stats()
    for k, v in stats.items():
        print("\t{:<16}: {:>16}".format(k, v))
    print("%s UPSample TEST %s" % (device, correct))
   


@pytest.mark.parametrize("device", ["vta", "arm_cpu"])
def test_upsample(device):

    def _run(env, remote):
        if device == "vta":
            # print(f'device:{device}')
            target = env.target
            # print(f'vta_target:{env.target}\n')
            if env.TARGET not in ["sim", "tsim", "intelfocl"]:
                assert tvm.runtime.enabled("rpc")
                program_fpga(remote, bitstream=None)
                reconfig_runtime(remote)
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        with autotvm.tophub.context(target):  # load pre-tuned schedule parameters
            for wl in upsample_workloads:
                # print(wl)
                # print(target)
                run_upsampling(env, remote, wl, target)
                # break

    vta.testing.run(_run)


if __name__ == "__main__":
    # test_upsample(device="arm_cpu")
    test_upsample(device="vta")