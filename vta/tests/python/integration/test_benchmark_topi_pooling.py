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

# Modified by contributors from Intel Labs

"""Testing topi pooling operator for VTA"""

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
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator

Workload = namedtuple("Pool2DWorkload", ['type', 'batch', 'filter',
    'height', 'width', 'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

env = vta.get_env()

# Pooling workloads
pool_wklds = [
    # ('pool.max.small', Workload('max', env.BATCH, env.BLOCK_OUT, 2, 2, 2, 2, 0, 0, 2, 2)),
    ('pool.max.pad',    Workload('max', env.BATCH, env.BLOCK_OUT, 2, 2, 2, 2, 1, 1, 2, 2)),
    # ('pool.max.medium', Workload('max', env.BATCH, env.BLOCK_OUT * 2, 20, 20, 4, 4, 0, 0, 4, 4)),
    # ('pool.max.resnet', Workload('max', env.BATCH, env.BLOCK_OUT, 111, 111, 3, 3, 1, 1, 2, 2)),
    # ('pool.avg.tiny',  Workload('avg', env.BATCH, env.BLOCK_OUT, 1, 1, 1, 1, 0, 0, 1, 1)),
    # ('pool.avg.small', Workload('avg', env.BATCH, env.BLOCK_OUT, 2, 2, 2, 2, 0, 0, 1, 1)),
    # ('pool.avg.medium', Workload('avg', env.BATCH, env.BLOCK_OUT * 4, 3, 3, 3, 3, 0, 0, 1, 1)),
    # ('pool.avg.resnet', Workload('avg', env.BATCH, 512, 7, 7, 7, 7, 0, 0, 1, 1)),
]

def run_pool(env, remote, wl, target,
               check_correctness=True, print_ir=False,
               samples=4):

    # Workload assertions
    assert wl.hpad == wl.wpad
    assert wl.hkernel == wl.wkernel
    assert wl.hstride == wl.wstride

    eff_h = (wl.height - wl.hkernel + (2 * wl.hpad))
    eff_w = (wl.width - wl.wkernel + (2 * wl.wpad))
    assert eff_h % wl.hstride == 0, "Height arguments do not divide evenly"
    assert eff_w % wl.wstride == 0, "Width arguments do not divide evenly"
    fout_height = (eff_h // wl.hstride) + 1
    fout_width = (eff_w // wl.wstride) + 1

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)

    # Derive shapes depending upon packing
    a_shape = (wl.batch, wl.filter, wl.height, wl.width)  # 根据硬件开始调整data的layout，如果数据的尺寸不能刚好装在VTA上怎么办？
    if data_pack:
        assert wl.batch % env.BATCH == 0, "BATCH does not evenly divide workload batch"
        assert wl.filter % env.BLOCK_OUT == 0, "BLOCK_OUT does not evenly divide workload filter"

        data_shape = (wl.batch//env.BATCH, wl.filter//env.BLOCK_OUT,
                      wl.height, wl.width, env.BATCH, env.BLOCK_OUT)              
        data = te.placeholder(data_shape, name="data", dtype=env.acc_dtype)
    else:
        data_shape = a_shape
        data = te.placeholder(data_shape, name="data", dtype=env.acc_dtype)

    # Define base computation schedule
    with target:
        if data_pack:
            res = vta.top.pooling_packed(data, (wl.hkernel, wl.wkernel), (wl.hstride, wl.wstride), 
                           (wl.hpad, wl.wpad, wl.hpad, wl.wpad), wl.type, False, layout, True)
            res = topi.cast(res, env.out_dtype)
            s = vta.top.schedule_pooling_packed([res], layout)
            if print_ir:
                print("Print TVM")
                print(tvm.lower(s, [data, res], simple_mode=True))
                print("Print VTA")
                print(vta.lower(s, [data, res], simple_mode=True))
        else:
            res = topi.nn.pool(data, (wl.hkernel, wl.wkernel), (wl.hstride, wl.wstride), 
                           (wl.hpad, wl.wpad, wl.hpad, wl.wpad), wl.type, layout=layout)
            s = topi.generic.schedule_pool([res], layout)
            if print_ir:
                print(tvm.lower(s, [data, res], simple_mode=True))

    # Derive number of ops
    num_ops = wl.batch * wl.filter * fout_height * fout_width * wl.hkernel * wl.wkernel 

    # @memoize("vta.tests.test_benchmark_topi.pooling.verify_nhwc")
    def get_ref_data():
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(env.acc_dtype)
        r_np = my_pool(a_np, (fout_height, fout_width),(wl.hpad,wl.wpad),(wl.hkernel,wl.wkernel),(wl.hstride,wl.wstride), wl.type, "NCHW")  # 这个函数不对，测试错误的
        return a_np, r_np

    def my_pool(np_data, out_size,pad,kernel,stride,pool_type, layout):
        assert layout=="NCHW"
        ishape=np_data.shape
        n,c,h,w=ishape
        oshape=(n,c)+out_size
        # padding
        pshape=(n,c,h+pad[0]*2,w+pad[1]*2)
        np_pad = np.full(pshape,-128).astype(np_data.dtype)
        for i in range(n):
            for j in range(c):
                for x in range(pad[0],pshape[2]-pad[0]):
                    for y in range(pad[1],pshape[3]-pad[1]):
                        np_pad[i][j][x][y]=np_data[i][j][x-pad[0]][y-pad[1]]
        # pooling
        np_out = np.zeros(oshape).astype(np_data.dtype)
        if pool_type=="avg":
            np_op=np.mean
        else:
            np_op=np.max
        for i in range(n):
            for j in range(c):
                for x in range(oshape[2]):
                    for y in range(oshape[3]):
                        np_out[i][j][x][y]=np_op(np_pad[i,j,x*stride[0]:x*stride[0]+kernel[0],y*stride[1]:y*stride[1]+kernel[1]])          
        return np_out     
        
    # Data in original format
    data_np, res_ref = get_ref_data()
    print("input")
    print(data_np)
    print("output")
    print(res_ref)

    if data_pack:
        data_np = data_np.reshape(wl.batch//env.BATCH, env.BATCH, 
                                  wl.filter//env.BLOCK_OUT, env.BLOCK_OUT, 
                                  wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
    # Build
    if "vta" in target.keys:
        mod = vta.build(s, [data, res],
                        target=target,
                        target_host=env.target_host,
                        name="pooling")
        # print("Print VTA")
        vta.lower(s, [data, res], simple_mode=True).show()
        # print(vta.lower(s, [data, res], simple_mode=True))
    else:
        mod = tvm.build(s, [data, res],
                        target=target,
                        target_host=env.target_host,
                        name="pooling")
    temp = utils.tempdir()
    mod.save(temp.relpath("pooling.o"))
    
    remote.upload(temp.relpath("pooling.o"))
    f = remote.load_module("pooling.o")
    ctx = remote.device(str(target))   # 改

    res_np = np.zeros(topi.utils.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, ctx)
    res_arr = tvm.nd.array(res_np, ctx)

    # In vta sim mode, collect simulator runtime statistics
    stats = {}
    cost = None
    if env.TARGET in ["sim", "tsim", "bsim"]:
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            if env.TARGET != "tsim":
                remote.get_function("vta.simulator.profiler_clear")()
            else:
                remote.get_function("vta.tsim.profiler_clear")()
            f(data_arr, res_arr)
            if env.TARGET != "tsim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            else:
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            f(data_arr, res_arr)
            stats = simulator.stats()
    else:
        f(data_arr, res_arr)
    # Check correctness
    correct = False
    if check_correctness:
        res_orig = res_arr.asnumpy()
        if data_pack:
            res_orig = res_orig.transpose(
                (0, 4, 1, 5, 2, 3)).reshape(wl.batch, wl.filter, fout_height, fout_width)
            res_ref = res_ref.astype(env.out_dtype)
        if wl.type == "max":
            atol = 0.0
        else:
            atol = 1.0 # division can be approximate
        correct = np.allclose(res_orig, res_ref, atol = atol)
        if atol == 1.0 and not np.allclose(res_orig, res_ref, atol = 0.0):
            print("Note: Arrays do not match exactly for average pool, atol = 1.0 was required for PASS")
            print("This is likely due to div approximation by shifts/adds and int rounding")
    status = "PASSED" if correct else "FAILED"
    
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"
        if not correct:
            print("Golden: ")
            print(res_ref)
            print("VTA: ")
            print(res_orig)

    print("%s POOLING TEST %s" % (device, status))
    return correct, cost, stats

@pytest.mark.parametrize("device", ["vta", "arm_cpu"])
def test_pool(device):
    def _run(env, remote):
        if device == "vta":
            target = env.target
            if env.TARGET not in ["sim", "tsim", "bsim"]:
                assert tvm.runtime.enabled("rpc")
                program_fpga(remote, bitstream=None)
                reconfig_runtime(remote)
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        with autotvm.tophub.context(target): # load pre-tuned schedule parameters
            for name, wl in pool_wklds:
                print("\n", name, "=>")
                print(wl)
                correct, _, _ = run_pool(env, remote, wl, target, print_ir = False)
                assert correct, "Test Failure for: " + device + " " + name
    vta.testing.run(_run)

if __name__ == "__main__":
    # test_pool(device="arm_cpu")
    test_pool(device="vta")