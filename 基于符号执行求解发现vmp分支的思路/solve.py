import triton
import SupertracePybind as Supertrace
from supertrace_util import compatibleProcessing, initTritonCtxEnv, mergeRepeatIns, checkIndirectIns

tracepath = "11111111.trace64"

trace = Supertrace.parse_x64dbg_trace(tracepath)
record = trace.getRecord() # 获取trace的记录指令列表
print(f"trace instruction num: {len(record)}")

ctx = triton.TritonContext()
ctx.setArchitecture(triton.ARCH.X86_64)
ctx.setMode(triton.MODE.ALIGNED_MEMORY, True)
ctx.setMode(triton.MODE.AST_OPTIMIZATIONS, True)
ctx.setMode(triton.MODE.CONSTANT_FOLDING, True)
ctx.setMode(triton.MODE.ONLY_ON_SYMBOLIZED, True)
ctx.setMode(triton.MODE.SYMBOLIZE_INDEX_ROTATION, True)

# 开启 QF_ABV 逻辑，否则为 QF_BV 逻辑
# ctx.setMode(triton.MODE.MEMORY_ARRAY, True)
# ctx.setMode(triton.MODE.SYMBOLIZE_LOAD, True)
# ctx.setMode(triton.MODE.SYMBOLIZE_STORE, True)

ctx.setAstRepresentationMode(triton.AST_REPRESENTATION.PYTHON)
astctx = ctx.getAstContext()

record = mergeRepeatIns(ctx, record, True) # 合并trace里的rep指令
print(f"after mergeing 'rep' instructions, trace instruction num: {len(record)}")

threads = trace.user.meta.getThreads()
for th in threads:
    if th.id == record[0].thread_id:
        main_thread = th
        break
print(f"main thread id: {main_thread.id} ({hex(main_thread.id)})")
print(f"teb: {hex(main_thread.teb)}")
initTritonCtxEnv(ctx, record[0], main_thread.teb) # 初始化寄存器环境

for i, ins in enumerate(record):
    ttins = triton.Instruction()
    ttins.setAddress(ins.ins_address)
    ttins.setOpcode(ins.bytes)
    ctx.disassembly(ttins)
    if (i + 1 >= len(record)): nextIns = None
    else: nextIns = record[i + 1]

    if (ins.dbg_id == 0): # 在最开始处符号化输入参数x
        sym_x = ctx.symbolizeRegister(ctx.registers.ecx, "Sym_X")

    compatibleProcessing(ctx, ttins, ins, nextIns, True, False) # 执行指令

    if (checkIndirectIns(ttins)): # 判断是否是间接跳转
        ripExpr = ctx.getSymbolicRegister(ctx.registers.rip)
        if ((ripExpr is None) or (not ripExpr.isSymbolized())): # 判断是否被符号化（是否与x有关）
            continue
        ripAst = astctx.unroll(ripExpr.getAst()) # 展开AST节点

        # ripAst.evaluate(): 当前跳转地址
        satProve = (ripAst != astctx.bv(ripAst.evaluate(), ripAst.getBitvectorSize()))

        model = ctx.getModel(satProve, True)
        status = model[1]
        if (status == triton.SOLVER_STATE.SAT):
            print(f"[{hex(ins.dbg_id)}] 已证明成功! 求解时间: {model[2]} 模型: {model[0]}")
            break
        elif (status == triton.SOLVER_STATE.UNSAT):
            print(f"[{hex(ins.dbg_id)}] 证明失败 求解时间: {model[2]}")
        elif (status == triton.SOLVER_STATE.TIMEOUT):
            print(f"[{hex(ins.dbg_id)}] 证明过程已超时")
        elif (status == triton.SOLVER_STATE.OUTOFMEM):
            print(f"[{hex(ins.dbg_id)}] 证明过程内存消耗殆尽")
        elif (status == triton.SOLVER_STATE.UNKNOWN):
            print(f"[{hex(ins.dbg_id)}] 证明过程发生未知错误: {status}")
        else:
            print(f"[{hex(ins.dbg_id)}] 证明过程发生未知错误: {status}")