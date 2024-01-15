# import tvm
# from tvm import relay
# import torchvision.models as models
# import torch
# import numpy as np


# if __name__=='__main__':
#   #prepare model and input
#   model = models.resnet18(pretrained=True)
#   shape_list = [("input0",(1,3,224,224))]
#   fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
#   graph = torch.jit.trace(model,fake_input)
#   #main function
#   mod, params = relay.frontend.from_pytorch(graph, shape_list)
#   print("Parsed mod "+str(mod))
import tvm
from tvm import relay
import torch
import numpy as np
import torchvision.models as models



import sys
sys.path.append("..")
from visualize import RelayVisualizer


'''
调用流程：
1 tvm.target.Target(创建Target，用于codegen过程，此篇不涉及)
  1.1 Target::Target -> TargetInternal::FromString -> TargetInternal::FromRawString
  1.2 GetTargetKind(从名字得到Kind模版)
  1.3 TargetInternal::ParseType(解析添加额外参数)
  1.4 TargetInternal::FromConfig(从配置得到target)
2 tvm.transform.PassContext(构建优化用的Context，存储信息)
  2.1 PassContext::Create
  2.2 PassConfigManager::Global()->Legalize(基础检查)
  2.3 PassContext::Internal::EnterScope
  2.4 PassContext::EnterWithScope(当前ctx入栈) 
3 relay.Optimize(执行优化)
  3.1 autotvm.tophub.context(用于schedule查找最优配置)
    3.1.1 ApplyHistoryBest(构建一个Context种类)
    3.1.2 ApplyHistoryBest.load(从文件加载历史配置)
  3.2 bld_mod.optimize(执行优化)
    3.2.0 BindParamsByName(绑定参数)
    3.2.1 SimplifyInference(优化normalize系列操作)
    3.2.2 ToANormalForm(使用赋值格式修改IR结构)
    3.2.3 FoldScaleAxis(折叠Scale类型算子)
    3.2.4 FoldConstant(常量折叠)
    3.2.5 AlterOpLayout(转换算子layout，更换strategy)
    3.2.6 FuseOps(算子聚类)

'''

def auto_optimize(mod,target,params):
  mod,params=relay.optimize(mod, target=target, params=params)
  visualizer=RelayVisualizer()
  visualizer.visualize(mod,path="visualizes/optimized_mod.prototxt")
  return mod,params

def debug_optimize(mod,target,params):
  mod["main"]=relay.build_module.bind_params_by_name(mod["main"],params)
  #add transform passes
  seq = tvm.transform.Sequential(
    [
      relay.transform.SimplifyInference(),
      relay.transform.BackwardFoldScaleAxis(),
      relay.transform.ForwardFoldScaleAxis(),
      relay.transform.FoldConstant(),
      relay.transform.AlterOpLayout(),
      relay.transform.FoldConstant(),
      relay.transform.FuseOps(),
    ]
  )
  with target:
    mod=seq(mod)

  visualizer=RelayVisualizer()
  visualizer.visualize(mod,path="visualizes/fuse_ops.prototxt")
  return mod,params

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  #main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  #optimize the mod
  #step 1 create target
  target = tvm.target.Target("llvm", host="llvm")
  #step 1 create PassContext
  with tvm.transform.PassContext(opt_level=3):
    #step 3 optimize
    mod,params=auto_optimize(mod,target,params)
  print("optimize func "+str(mod["main"]))