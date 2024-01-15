



"""Pooling operator declaration and schedule registration for VTA."""

# Modified by contributors from Intel Labs

import math
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from .utils import is_packed_layout
from ..environment import get_env

def get_factors(x):
    assert x >= 1
    factors = []
    for i in range(1, x+1):
        if x % i == 0:
            factors.append(i)
    return factors

def get_shifts_and_addsubs(x, num_bits):
    """ Determine the sequence of shifts and adds/subs to approximate division
        We have num_bits overall to work with
    """
    assert x >= 1 # a fraction <= 1
    frac = float(1)/float(x)
    div = len(bin(x)[:1:-1]) - 1 # number of bits
    sel = []
    remain = frac

    for i in range(div, num_bits):
        if remain >= 2**(-1*i): # do we get closer to zero
            remain -= 2**(-1*i)
            sel.append(i)
    # print(num_bits, "bit approximation of 1 /", x, "has error:", remain)
    # post-process sel array to incremental mode
    if len(sel) > 1:
        cur = sel[0]
        for i in range(1, len(sel)):
            sel[i] = sel[i] - cur
            cur += sel[i]
    return sel

# scaling
def get_scaling_bits(min_a, max_a, b, full_width):
    max_val = (abs(min_a) + max_a) * b
    scaling_factor = math.floor((1 << (full_width - 1)) / max_val)
    scaling_bits = len(bin(scaling_factor)[:1:-1]) - 1
    # print("Division approximation upscales input by",
    print(scaling_bits, "based on summation of", b, "values between", min_a, "and", max_a)
    return scaling_bits

@autotvm.register_topi_compute("pooling_packed.vta")
def pooling_packed(cfg, data, kernel, strides, padding, pool_type,
                   ceil_mode, layout, count_include_pad,out_dtype='int32',
                   in_range=None, out_range=None):
    """ Packed pooling function."""

    if in_range is None:
        in_range = [-128, 127]
    if out_range is None:
        out_range = [-128, 127]
    # three are no use??
    _ = cfg
    _ = ceil_mode
    _ = count_include_pad
    # print("layout:",layout)
    # print("data.shape:",data.shape)
    if not is_packed_layout(layout):
        raise topi.InvalidShapeError()
    # data.shape: NCHW n c
    if pool_type == "avg": # global average pool for now (1 output per filter)
        oshape = [data.shape[0], data.shape[1], 1, 1, data.shape[4], data.shape[5]]  # output shape: batch, channel, 1, 1, n, c
        x = te.reduce_axis((0, data.shape[2]), name="x")  # H
        y = te.reduce_axis((0, data.shape[3]), name="y")  # W
        res = te.compute(oshape, lambda i, j, ox, oy, k, m:
                         te.sum(data[i, j, x, y, k, m], axis=[x, y]), name="res")

        # input and output ranges
        in_bits = len(bin(in_range[1]-in_range[0]+1)[:1:-1]) - 1  # compute range will take how_much bits
        out_bits = len(bin(out_range[1]-out_range[0]+1)[:1:-1]) - 1

        assert in_range[1] >= in_range[0], "Input range must contain >= 1 value"
        assert out_range[1] >= out_range[0], "Output range must contain >= 1 value"
        assert (1 << in_bits) == in_range[1]-in_range[0]+1, "In range must span power-of-two"   # must be 2**n
        assert (1 << out_bits) == out_range[1]-out_range[0]+1, "Out range must span power-of-two"
        assert in_bits >= out_bits, "Input range must be at least as large as output range"

        # approximate the division operation
        divisor = (data.shape[2] * data.shape[3]).value    # H*W
        # print("divisor:",divisor)

        sbits = get_scaling_bits(in_range[0], in_range[1], divisor, 32)
        sel_vec = get_shifts_and_addsubs(divisor, 32)

        if in_range[0] < 0: # if there could be negative inputs
            res = res + abs(in_range[0])*divisor # make the sum positive

        res = te.compute(oshape, lambda i, j, ox, oy, k, m:
                         res[i, j, ox, oy, k, m] << sbits, name="scale_up")
        scratch = te.compute(oshape, lambda i, j, ox, oy, k, m: 0, name="zero") # init temp buffer
        for ziter in sel_vec: # shift and accumulate loop
            res = topi.right_shift(res, ziter)
            scratch = scratch + res
        # scale down, but take into account potential output range restriction vs. input range
        scratch = te.compute(oshape, lambda i, j, ox, oy, k, m:
                             scratch[i, j, ox, oy, k, m] >> (sbits + in_bits - out_bits),
                             name="scale_down")
        if out_range[0] < 0:
            scratch = scratch + out_range[0] # subtract back to output range if needed
        return scratch
    elif pool_type=='max':
        if not is_packed_layout(layout):
            raise topi.InvalidShapeError()
        if padding[0]:
            # pad_data=data
            pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], pad_value=-128,name="pad_data")   # ->TVM topi  extend the height and width for data
            # scratch = te.compute(oshape, lambda i, j, ox, oy, k, m: -128, name="min_value") # init temp buffer
            # pad_data=te.compute(
            #     oshape,
            #     lambda i,j,ox,oy,k,m:
            #         scratch[i,j,ox,oy,k,m],                                
            # )
        else: 
            pad_data = data
        assert len(data.shape) == 6   
        # assert len(kernel.shape) == 2
        # # print("pad_data",pad_data.shape)
        oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel[0]) // strides[0] + 1)
        owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel[1]) // strides[1] + 1)
        oshape = (data.shape[0], data.shape[1], oheight, owidth, data.shape[4], data.shape[5])
        # # print("oshape",oshape)
        ishape = topi.utils.get_const_tuple(pad_data.shape) 
        kshape = kernel
        d_i = te.reduce_axis((0, kshape[0]), name="d_i")
        d_j = te.reduce_axis((0, kshape[1]), name="d_j")
        # k_o = te.reduce_axis((0, ishape[1]), name="k_o")
        # k_i = te.reduce_axis((0, ishape[-1]), name="k_i")
        hstride, wstride = strides
        # print("oshape",oshape)
        # scratch = te.compute(oshape, lambda i, j, ox, oy, k, m: -128, name="pad_data") # init temp buffer
        # pad_data=scratch
        res = te.compute(
            oshape,
            lambda b_o,k_o,i,j,b_i,k_i: te.max(
                pad_data[b_o,k_o,i*hstride+d_i,j * wstride + d_j,b_i,k_i],
                axis=[d_i,d_j],
            ),
            name="res",
            tag="max_pooling",
        )
        return res       
    # max pool is much simpler, just use the topi code
    # res = topi.nn.pool(data, kernel, stride, (1,1), padding, pool_type, layout=layout)
    # return res

@autotvm.register_topi_schedule("pooling_packed.vta")
def schedule_pooling_packed(cfg, outs, layout=None):
    """Schedule packed pooling"""
    assert len(outs) == 1
    env = get_env()
    _ = cfg
    _ = layout

    output = outs[0]
    assert "int" in output.op.input_tensors[0].dtype
    s = te.create_schedule(output.op)
    # print("schedule part")
    # print(outs, output.op)
    # print(s)
    def traverse_ops(op, pad_ops, pool_ops, div_ops, inps, seen):
        if op in seen:
            return pad_ops, pool_ops, div_ops, inps, seen
        seen.append(op)
        if isinstance(op, tvm.te.PlaceholderOp):
            inps.append(op) # end of the chain
            return pad_ops, pool_ops, div_ops, inps, seen
        if isinstance(op, tvm.te.ComputeOp):
            if isinstance(op.body, tvm.ir.container.Array) and \
               isinstance(op.body[0], tvm.tir.expr.Cast):
                assert op == output.op
            elif isinstance(op.body, tvm.ir.container.Array) and \
               isinstance(op.body[0], tvm.tir.expr.Reduce):
                assert len(pool_ops) == 0
                pool_ops.append(op)
            elif "pad" in op.name:
                assert len(pad_ops) == 0
                pad_ops.append(op)
            else: # a decomposed division op
                div_ops.append(op)
        else:
            print("Unknown:", type(op))

        for i in op.input_tensors: # recursive call for all inputs
            # print(f'op_inputensor_:{i.op}')
            pad_ops, pool_ops, div_ops, inps, seen = traverse_ops(i.op, pad_ops, pool_ops,
                                                                  div_ops, inps, seen)
        return pad_ops, pool_ops, div_ops, inps, seen

    # order of ops before output is: inp, [pad], pool, [div], output
    pad_ops = []
    pool_ops = []
    div_ops = []
    inps = []
    seen = []
    pad_ops, pool_ops, div_ops, inps, seen = traverse_ops(output.op, pad_ops, pool_ops,
                                                          div_ops, inps, seen)
    print("inps",inps[0].shape,inps)
    print("pad_ops",pad_ops)
    print("pool_ops",pool_ops)
    print()
    print("div_ops",div_ops)
    print("seen",seen)
    print("output",output)

    assert len(inps) == 1
    inp = inps[0]

    if pad_ops != []:
        pad_op = pad_ops[0]
    else:
        pad_op = None

    if pool_ops != []:
        pool_op = pool_ops[0]
    else:
        pool_op = None
    # add
    # s[inp].set_scope(env.inp_scope)
    
    p_i, p_j, p_oh, p_ow, p_x, p_y = s[pool_op].op.axis
    # print("s[pool_op].op.axis")
    # print(s[pool_op].op.axis)    # output data shape  
    p_kh, p_kw = s[pool_op].op.reduce_axis
    # print("s[pool_op].op.reduce_axis")   # reduced axis's range 
    # print(s[pool_op].op.reduce_axis)
    

    # start with assumption that entire frame, with BATCH * BLOCK_OUT components
    # per pixel, resides in the scratchpad
    acc_size = int(env.ACC_BUFF_SIZE//(env.ACC_WIDTH/8))
    scratch_axis = 1

    if len(div_ops) == 0:
        extra_space_factor = 1
    else:
        extra_space_factor = 2 # need another copy of output as temp storage
    # np.prod: get the miltiple result of all element
    scratch_size = np.prod(inp.shape[scratch_axis+1:]) + \
      extra_space_factor * np.prod(output.shape[scratch_axis+1:])

    if scratch_size > acc_size:
        if len(div_ops) == 0: # max pool
            # print("Splitting into row to reduce scratchpad utilization")
            ## print("Out height factors:", get_factors(output.shape[scratch_axis+1].value))

            scratch_axis += 1 # descend one level lower
            inps_per_out = int(round(inp.shape[scratch_axis].value/
                                     output.shape[scratch_axis].value))
            scratch_size = inps_per_out * np.prod(inp.shape[scratch_axis+1:]) \
                         + np.prod(output.shape[scratch_axis+1:]) # the row axis is gone
            if scratch_size > acc_size: #Cannot fit single row of pooling I/O in scratchpad
                # print("Splitting into single element to further reduce scratchpad utilization")
                scratch_axis += 1 # descend one level: now a single output pixel
                scratch_size = inps_per_out * inps_per_out * np.prod(inp.shape[scratch_axis+1:]) \
                             + np.prod(output.shape[scratch_axis+1:]) # the row/col axes are gone
            #todo: split height axis first according to factors, then compute_at into new axis
        else: # global average pool
            # print("Cannot split global average pool frame any further, ERROR")
            return s

    # print("Acc utilization:", scratch_size, "elems")
    # print("Acc size:", acc_size, "elems")

    # scratchpad and compute at the desired level
    output_store_pt = s[output].op.axis[scratch_axis]
    output_dma_pt = s[output].op.axis[scratch_axis+1]   # DMAcopy是按照这个轴进行的、
    # print("s[output].op.axis")
    # print(s[output].op.axis)
    # print(output_store_pt)
    # print(output_dma_pt)


    for diter in div_ops: # average pooling operation div sequence
        s[diter].set_scope(env.acc_scope)
        # s[diter].compute_at(s[output], output_store_pt)
        s[diter].pragma(s[diter].op.axis[0], env.alu)
    s[pool_op].set_scope(env.acc_scope)
    s[pool_op].reorder(p_kh, p_kw, p_i, p_j, p_oh, p_ow, p_x, p_y)
    # print("s[pool_up]",s[pool_op])
    # s[pool_op].compute_at(s[output], output_store_pt)
    
    # print("Unrolling kernel of height:", p_kh.dom.extent, "width:", p_kw.dom.extent)
    """不影响语句"""
    # s[pool_op].unroll(p_kh)
    # s[pool_op].unroll(p_kw)

    # ic_out, ic_inn = s[res_conv].split(ic, factor=ic_block)
    # set acc scope and alu pragma
    # s[pool_op].set_scope(env.acc_scope)
    # s[pool_op].compute_at(s[output], output_store_pt)
    s[pool_op].pragma(s[pool_op].op.axis[0], env.alu)

    env = get_env()
    
    if pad_op is None:  # 1
        cdata = s.cache_read(inps[0].output(0), env.acc_scope, pool_op)
    else:
        # s[pad_op].pragma(s[pad_op].op.axis[0], env.alu)
        s[pad_op].set_scope(env.acc_scope)
        # s[pad_op].compute_at(s[output], output_store_pt)
        cdata = pad_op.output(0)
        # print("pad_op",pad_op.output)
        # s[pad_op].pragma(s[pad_op].op.axis[0], env.dma_copy)
        # s[cdata].set_scope(env.acc_dtype)
    # cdata=pad_op.output(0)
    # print("cdata",cdata)

    
    # s[cdata].compute_at(s[output], output_store_pt)  # 将s[cdata]的计算附着到s[output]的output_store_pt轴上；
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[output].pragma(output_dma_pt, env.dma_copy)
    # print("schedule test")
    # print(s)
    return s