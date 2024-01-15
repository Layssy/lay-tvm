import math
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
from vta.top.utils import is_packed_layout
from vta.environment import get_env




@autotvm.register_topi_compute("upsampling_packed.vta")
def upsampling(
    cfg,
    data,
    scale_h,
    scale_w,
    layout="NCHW",
    method="nearest_neighbor",
    align_corners=False,
    output_shape=None,
):
    env = get_env()
    print(f'layout:{layout}')
    batch, channels, in_height, in_width, in_batch, in_channels   = data.shape
    # batch, channels, in_height, in_width = data.shape
    data_shape = data.shape
    # ishape = topi.utils.get_const_tuple(data.shape)
    out_height = int(in_height.value * scale_h)
    out_width = int(in_width.value * scale_w)
    # data_buf = te.compute(
    #     data_shape, lambda *i: data(*i),  
    #     name="data_buf",
    #     tag="data_buf"
    #     )
    # output = topi.nn.upsampling(
    #         # data_buf,
    #         data.astype(env.out_dtype),
    #         scale_h=scale_h,
    #         scale_w=scale_w,
    #         layout=layout,
    #         method=method,
    #         align_corners=align_corners,
    #         # output_shape=( batch, channels, out_height, out_width)
    #     )
    output = te.compute(
        (batch, channels, out_height, out_width, in_batch, in_channels),
        lambda b_o, c_o, i, j, b_i, c_i: data[b_o, c_o, te.round(te.floordiv(i, scale_h)).astype(env.acc_dtype), te.round(te.floordiv(j, scale_w)).astype(env.acc_dtype), b_i, c_i].astype(env.out_dtype), # 因为直接展示的int32的数据导致出现了后面数据全部为0的情况，请数据返回类型改为out_dtype就可以解决
        name="upsampling_packed",
        tag="upsampling_packed",
    )

    # output = topi.cast(output, env.acc_dtype)
    # /output.astype(env.acc_dtype)
    return output

@autotvm.register_topi_schedule("upsampling_packed.vta")
def schedule_upsampling_packed(cfg, outs, layout=None):
    # assert layout == "NCHW", "Only NCHW layout is supported for upsampling"
    assert len(outs) == 1
    
    env = get_env()
    output = outs[0]# output 是待优化的计算张量。
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    upsample_res = []
    assert "int" in output.op.input_tensors[0].dtype
    def _traverse(op):
        print(f'traverse_op:{op}')
        if topi.tag.is_broadcast(op.tag):
            
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    print(f'traverse_op:{op}')
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            # print(f'op.tag:{op.tag}')
            # assert op.tag == "injective"
            assert op.tag == "upsampling_packed"
            # print(f'injective:{op}')
            upsample_res.append(op)


    _traverse(output.op)
    assert len(upsample_res) == 1
    # print(f'ewise_ops:{type(ewise_ops)}\n{len(ewise_ops)}')
    # print(f'const_ops:{type(const_ops)}\n{len(const_ops)}')
    # print(f'len_upsample_res:{type(upsample_res)}\n{upsample_res}')
    upsample_stage = upsample_res[0].output(0)
    # ewise_ops = ewise_ops[0:-1]
   
    # batch, channels, in_height, in_width  = output.shape
    batch, channels, in_height, in_width, batch_inner, channels_inner  = output.shape


    # Create schedule
    s = te.create_schedule(output.op)
    # # Reorder axes
    bo, co, h, w, bi, ci= s[upsample_stage].op.axis
    cfg.define_split("tile_bo", bo, num_outputs=2)
    cfg.define_split("tile_co", co, num_outputs=2)
    cfg.define_split("tile_h", h, num_outputs=2)
    cfg.define_split("tile_w", w, num_outputs=2) 
    cfg.define_split("title_bi", bi, num_outputs=2)
    cfg.define_split("tile_ci", ci, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    cfg.define_knob("h_nthread", [1, 2])
    
    
    data, = upsample_stage.op.input_tensors
    print(f'data_type:{data}\ntype:{type(data)}')
    # cdata = s.cache_read(data, env.inp_scope, upsample_stage)
    cdata = s.cache_read(data, env.acc_scope, upsample_stage)
    # print(f'data:{type(data)}')
    # cdata = s.cache_read(data, env.inp_scope, upsample_stage)
    # cdata = data
    s[upsample_stage].set_scope(env.acc_scope)

    cache_read_ewise = []
    # for consumer, tensor in ewise_inputs:
    #     cache_read_ewise.append(s.cache_read(tensor, env.inp_scope, [consumer]))   

    # # # set ewise scope
    # for op in ewise_ops:
    #     print(f'op_ewise:{op}')
    #     s[op].set_scope(env.acc_scope)
    #     s[op].pragma(s[op].op.axis[0], env.alu)
        
        
    # for op in const_ops:
    #     s[op].compute_inline()     
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg["tile_co"].apply(s, output, x_co)
    x_i0, x_i1 = cfg["tile_h"].apply(s, output, x_i)
    x_j0, x_j1 = cfg["tile_w"].apply(s, output, x_j)
    s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_co1
    store_pt_out = x_j1
    # x_j0
    store_out = env.dma_copy
    # set all compute scopes
    # s[upsample_stage].compute_at(s[output], store_pt)
    # # for op in ewise_ops:
        # s[op].compute_at(s[output], store_pt)


    # for tensor in cache_read_ewise:
    #     # s[tensor].compute_at(s[output], store_pt)
    #     s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[upsample_stage].op.axis
    print(f's[upsample_stage_x_bi:{type(x_bi)}]')
    # # print(f'cdata:{cdata.op}')
    # s[cdata].compute_at(s[output], store_pt)
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    
    # s[upsample_stage].tensorize(x_bi, env.gemm)
    # s[upsample_stage].unroll(x_i)
    # s[upsample_stage].unroll(x_j)
    
    print(f's[cdata].op:{s[cdata].op}\ns[cdata].op.input_tensors:{s[cdata].op.input_tensors}\ns[cdata].op.input_tensors.dtype:{s[cdata].op.input_tensors[0].dtype}')
    
    # print()
    
    print(f'upsample_stage.op:{upsample_stage.op}\nupsample_stage.op.tensor:{upsample_stage.op.input_tensors}\nupsample_stage.op.tensor.dtype:{upsample_stage.op.input_tensors[0].dtype}')
    
    print()

    
    # print(f's[output].op:{s[output].op}\ns[output].op.tensor:{s[output].op.input_tensors}\ns[output].op.tensor.dtype:{s[output].op.input_tensors[0].dtype}')
    # s[upsample_stage].tensorize(x_bi, env.gemm)
    # x_co1
    s[output].pragma(store_pt_out, env.dma_copy)
    
    
    
    return s



