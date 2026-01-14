import triton
import SupertracePybind as Supertrace
from supertrace_util import compatibleProcessing, initTritonCtxEnv, mergeRepeatIns, checkIndirectIns

tracepath = "11111111.trace64"
#tracepath = "deadbeef.trace64"

trace = Supertrace.parse_x64dbg_trace(tracepath)
record = trace.getRecord() # 获取trace的记录指令列表
print(f"trace instruction num: {len(record)}")

modules = trace.user.meta.getModules()
for mod in modules:
    if mod.isMainModule:
        main_module = mod
        break

main_secs = main_module.getSections()
print("No\tName")
for i, sec in enumerate(main_secs):
    print(f"{i}\t{sec.name}")

vmpsec_begin = main_secs[4].addr
vmpsec_end = main_secs[7].addr
print(f"vmp begin: {hex(vmpsec_begin)}")
print(f"vmp end: {hex(vmpsec_end)}")

ctx = triton.TritonContext()
ctx.setArchitecture(triton.ARCH.X86_64)
ctx.setMode(triton.MODE.ALIGNED_MEMORY, True)
ctx.setMode(triton.MODE.AST_OPTIMIZATIONS, True)
ctx.setMode(triton.MODE.CONSTANT_FOLDING, True)
ctx.setMode(triton.MODE.ONLY_ON_TAINTED, True)
ctx.setMode(triton.MODE.TAINT_THROUGH_POINTERS, True) # 开启内存指针污染

record = mergeRepeatIns(ctx, record, True) # 合并trace里的rep指令
print(f"after mergeing 'rep' instructions, trace instruction num: {len(record)}")

threads = trace.user.meta.getThreads()
for th in threads:
    if th.id == record[0].thread_id:
        main_thread = th
        break
print(f"main thread id: {main_thread.id} ({hex(main_thread.id)})")
print(f"teb: {hex(main_thread.teb)}") # 获取线程TEB地址
initTritonCtxEnv(ctx, record[0], main_thread.teb) # 初始化寄存器环境

for i, ins in enumerate(record):
    if (i + 1 >= len(record)): nextIns = None
    else: nextIns = record[i + 1]

    ttins = triton.Instruction()
    ttins.setAddress(ins.ins_address)
    ttins.setOpcode(ins.bytes)
    ctx.disassembly(ttins)

    for memAcc in ins.mem_accs:
        if (memAcc.type == Supertrace.AccessType.READ and (vmpsec_begin <= memAcc.acc_address <= vmpsec_end)):
            ctx.taintMemory(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size)) # 污染来自VM区块的字节码（确保不会把混淆间接跳转也给识别进去）

    compatibleProcessing(ctx, ttins, ins, nextIns, True, False) # 执行指令

    if (checkIndirectIns(ttins) and (nextIns is not None) and ttins.isTainted()):
        jumpAddr = nextIns.ins_address
        print(f"{hex(ins.dbg_id)}\t{ttins.getDisassembly()}\t\t{hex(jumpAddr)}") # 打印指令序号、指令反汇编文本以及对应的跳转地址