import tvm

# 获取指定操作的策略
strategy = tvm.autotvm.task.DispatchContext.current.query(tvm.te.ops.nn.upsampling(0, 0, 0))

# 检查策略是否存在
if strategy is not None:
    print("策略已成功注册")
else:
    print("策略未注册")