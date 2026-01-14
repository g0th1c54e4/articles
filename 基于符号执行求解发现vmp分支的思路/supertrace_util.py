import SupertracePybind as Supertrace
import triton

def mergeRepeatIns(
    ctx: triton.TritonContext,
    record: list[Supertrace.InstructionRecord],
    sort_again: bool
) -> list[Supertrace.InstructionRecord]:

    if not record:
        return record

    result: list[Supertrace.InstructionRecord] = []

    i = 0
    n = len(record)

    while i < n:
        j = i + 1

        while j < n:
            # 当前指令
            ttins_cur = triton.Instruction()
            ttins_cur.setAddress(record[i].ins_address)
            ttins_cur.setOpcode(record[i].bytes)
            # 下一条指令
            ttins_next = triton.Instruction()
            ttins_next.setAddress(record[j].ins_address)
            ttins_next.setOpcode(record[j].bytes)

            ctx.disassembly(ttins_cur)
            ctx.disassembly(ttins_next)

            prefix_cur = ttins_cur.getPrefix()
            prefix_next = ttins_next.getPrefix()

            is_repeating = (
                ttins_cur.getAddress() == ttins_next.getAddress() and
                prefix_cur in (
                    triton.PREFIX.X86.REP,
                    triton.PREFIX.X86.REPE,
                    triton.PREFIX.X86.REPNE,
                ) and
                prefix_next in (
                    triton.PREFIX.X86.REP,
                    triton.PREFIX.X86.REPE,
                    triton.PREFIX.X86.REPNE,
                ) and
                prefix_cur == prefix_next
            )

            if not is_repeating:
                break

            j += 1

        # 合并 [i, j)
        merged = record[i]

        for k in range(i + 1, j):
            # 合并内存访问
            merged.mem_accs.extend(record[k].mem_accs)
            # 使用最后一次的寄存器状态
            merged.reg_dump = record[k].reg_dump

        result.append(merged)
        i = j

    if sort_again:
        for idx, rec in enumerate(result):
            rec.id = idx

    return result

def initTritonCtxEnv(
    ctx: triton.TritonContext,
    init_refer_ins: Supertrace.InstructionRecord,
    teb_addr: int
) -> None:

    regs = ctx.registers
    if (ctx.getArchitecture() == triton.ARCH.X86):
        init_regdump = init_refer_ins.reg_dump32
    elif ctx.getArchitecture() == triton.ARCH.X86_64:
        init_regdump = init_refer_ins.reg_dump64
    else:
        print("unknown arch")

    # === GPR ===
    if ctx.getArchitecture() == triton.ARCH.X86:
        ctx.setConcreteRegisterValue(regs.esp, init_regdump.regcontext.csp)
        ctx.setConcreteRegisterValue(regs.ebp, init_regdump.regcontext.cbp)
        ctx.setConcreteRegisterValue(regs.esi, init_regdump.regcontext.csi)
        ctx.setConcreteRegisterValue(regs.edi, init_regdump.regcontext.cdi)
        ctx.setConcreteRegisterValue(regs.eax, init_regdump.regcontext.cax)
        ctx.setConcreteRegisterValue(regs.ebx, init_regdump.regcontext.cbx)
        ctx.setConcreteRegisterValue(regs.ecx, init_regdump.regcontext.ccx)
        ctx.setConcreteRegisterValue(regs.edx, init_regdump.regcontext.cdx)
        ctx.setConcreteRegisterValue(regs.eip, init_regdump.regcontext.cip)

    elif ctx.getArchitecture() == triton.ARCH.X86_64:
        ctx.setConcreteRegisterValue(regs.rsp, init_regdump.regcontext.csp)
        ctx.setConcreteRegisterValue(regs.rbp, init_regdump.regcontext.cbp)
        ctx.setConcreteRegisterValue(regs.rsi, init_regdump.regcontext.csi)
        ctx.setConcreteRegisterValue(regs.rdi, init_regdump.regcontext.cdi)
        ctx.setConcreteRegisterValue(regs.rax, init_regdump.regcontext.cax)
        ctx.setConcreteRegisterValue(regs.rbx, init_regdump.regcontext.cbx)
        ctx.setConcreteRegisterValue(regs.rcx, init_regdump.regcontext.ccx)
        ctx.setConcreteRegisterValue(regs.rdx, init_regdump.regcontext.cdx)
        ctx.setConcreteRegisterValue(regs.rip, init_regdump.regcontext.cip)

        ctx.setConcreteRegisterValue(regs.r8,  init_regdump.regcontext.r8)
        ctx.setConcreteRegisterValue(regs.r9,  init_regdump.regcontext.r9)
        ctx.setConcreteRegisterValue(regs.r10, init_regdump.regcontext.r10)
        ctx.setConcreteRegisterValue(regs.r11, init_regdump.regcontext.r11)
        ctx.setConcreteRegisterValue(regs.r12, init_regdump.regcontext.r12)
        ctx.setConcreteRegisterValue(regs.r13, init_regdump.regcontext.r13)
        ctx.setConcreteRegisterValue(regs.r14, init_regdump.regcontext.r14)
        ctx.setConcreteRegisterValue(regs.r15, init_regdump.regcontext.r15)

    # === EFLAGS ===
    ctx.setConcreteRegisterValue(regs.eflags, init_regdump.regcontext.eflags)

    ctx.setConcreteRegisterValue(regs.zf, init_regdump.flags.z)
    ctx.setConcreteRegisterValue(regs.cf, init_regdump.flags.c)
    ctx.setConcreteRegisterValue(regs.af, init_regdump.flags.a)
    ctx.setConcreteRegisterValue(regs.of, init_regdump.flags.o)
    ctx.setConcreteRegisterValue(regs.pf, init_regdump.flags.p)
    ctx.setConcreteRegisterValue(regs.sf, init_regdump.flags.s)
    ctx.setConcreteRegisterValue(regs.tf, init_regdump.flags.t)
    ctx.setConcreteRegisterValue(getattr(regs, "if"), init_regdump.flags.i)
    ctx.setConcreteRegisterValue(regs.df, init_regdump.flags.d)

    # === Debug Registers ===
    ctx.setConcreteRegisterValue(regs.dr0, init_regdump.regcontext.dr0)
    ctx.setConcreteRegisterValue(regs.dr1, init_regdump.regcontext.dr1)
    ctx.setConcreteRegisterValue(regs.dr2, init_regdump.regcontext.dr2)
    ctx.setConcreteRegisterValue(regs.dr3, init_regdump.regcontext.dr3)
    ctx.setConcreteRegisterValue(regs.dr6, init_regdump.regcontext.dr6)
    ctx.setConcreteRegisterValue(regs.dr7, init_regdump.regcontext.dr7)

    # === TEB ===
    if ctx.getArchitecture() == triton.ARCH.X86:
        ctx.setConcreteRegisterValue(regs.fs, teb_addr)
    elif ctx.getArchitecture() == triton.ARCH.X86_64:
        ctx.setConcreteRegisterValue(regs.gs, teb_addr)

def getRegValByTTReg64(
    ins: Supertrace.InstructionRecord,
    ttreg
) -> int:
    ttregId = ttreg.getId()
    regcontext = ins.reg_dump64.regcontext

    if ttregId in (triton.REG.X86_64.RAX, triton.REG.X86_64.EAX):
        return regcontext.cax
    elif ttregId in (triton.REG.X86_64.RBX, triton.REG.X86_64.EBX):
        return regcontext.cbx
    elif ttregId in (triton.REG.X86_64.RCX, triton.REG.X86_64.ECX):
        return regcontext.ccx
    elif ttregId in (triton.REG.X86_64.RDX, triton.REG.X86_64.EDX):
        return regcontext.cdx
    elif ttregId in (triton.REG.X86_64.RSI, triton.REG.X86_64.ESI):
        return regcontext.csi
    elif ttregId in (triton.REG.X86_64.RDI, triton.REG.X86_64.EDI):
        return regcontext.cdi
    elif ttregId in (triton.REG.X86_64.RBP, triton.REG.X86_64.EBP):
        return regcontext.cbp
    elif ttregId in (triton.REG.X86_64.RSP, triton.REG.X86_64.ESP):
        return regcontext.csp
    elif ttregId in (triton.REG.X86_64.RIP, triton.REG.X86_64.EIP):
        return regcontext.cip
    elif ttregId in (triton.REG.X86_64.R8, triton.REG.X86_64.R8D):
        return regcontext.r8
    elif ttregId in (triton.REG.X86_64.R9, triton.REG.X86_64.R9D):
        return regcontext.r9
    elif ttregId in (triton.REG.X86_64.R10, triton.REG.X86_64.R10D):
        return regcontext.r10
    elif ttregId in (triton.REG.X86_64.R11, triton.REG.X86_64.R11D):
        return regcontext.r11
    elif ttregId in (triton.REG.X86_64.R12, triton.REG.X86_64.R12D):
        return regcontext.r12
    elif ttregId in (triton.REG.X86_64.R13, triton.REG.X86_64.R13D):
        return regcontext.r13
    elif ttregId in (triton.REG.X86_64.R14, triton.REG.X86_64.R14D):
        return regcontext.r14
    elif ttregId in (triton.REG.X86_64.R15, triton.REG.X86_64.R15D):
        return regcontext.r15
    elif ttregId == triton.REG.X86_64.EFLAGS:
        return regcontext.eflags
    elif ttregId == triton.REG.X86_64.CS:
        return regcontext.cs
    elif ttregId == triton.REG.X86_64.DS:
        return regcontext.ds
    elif ttregId == triton.REG.X86_64.ES:
        return regcontext.es
    elif ttregId == triton.REG.X86_64.FS:
        return regcontext.fs
    elif ttregId == triton.REG.X86_64.GS:
        return regcontext.gs
    elif ttregId == triton.REG.X86_64.SS:
        return regcontext.ss
    elif ttregId == triton.REG.X86_64.DR0:
        return regcontext.dr0
    elif ttregId == triton.REG.X86_64.DR1:
        return regcontext.dr1
    elif ttregId == triton.REG.X86_64.DR2:
        return regcontext.dr2
    elif ttregId == triton.REG.X86_64.DR3:
        return regcontext.dr3
    elif ttregId == triton.REG.X86_64.DR6:
        return regcontext.dr6
    elif ttregId == triton.REG.X86_64.DR7:
        return regcontext.dr7
    else:
        raise NotImplementedError(f"Unsupported register: {ttreg.getName()} ({ttregId})")

def getRegValByTTReg32(
    ins: Supertrace.InstructionRecord,
    ttreg
) -> int:
    ttregId = ttreg.getId()
    regcontext = ins.reg_dump32.regcontext

    if ttregId == triton.REG.X86.EAX:
        return regcontext.cax
    elif ttregId == triton.REG.X86.EBX:
        return regcontext.cbx
    elif ttregId == triton.REG.X86.ECX:
        return regcontext.ccx
    elif ttregId == triton.REG.X86.EDX:
        return regcontext.cdx
    elif ttregId == triton.REG.X86.ESI:
        return regcontext.csi
    elif ttregId == triton.REG.X86.EDI:
        return regcontext.cdi
    elif ttregId == triton.REG.X86.EBP:
        return regcontext.cbp
    elif ttregId == triton.REG.X86.ESP:
        return regcontext.csp
    elif ttregId == triton.REG.X86.EIP:
        return regcontext.cip

    elif ttregId == triton.REG.X86.EFLAGS:
        return regcontext.eflags

    elif ttregId == triton.REG.X86.CS:
        return regcontext.cs
    elif ttregId == triton.REG.X86.DS:
        return regcontext.ds
    elif ttregId == triton.REG.X86.ES:
        return regcontext.es
    elif ttregId == triton.REG.X86.FS:
        return regcontext.fs
    elif ttregId == triton.REG.X86.GS:
        return regcontext.gs
    elif ttregId == triton.REG.X86.SS:
        return regcontext.ss

    elif ttregId == triton.REG.X86.DR0:
        return regcontext.dr0
    elif ttregId == triton.REG.X86.DR1:
        return regcontext.dr1
    elif ttregId == triton.REG.X86.DR2:
        return regcontext.dr2
    elif ttregId == triton.REG.X86.DR3:
        return regcontext.dr3
    elif ttregId == triton.REG.X86.DR6:
        return regcontext.dr6
    elif ttregId == triton.REG.X86.DR7:
        return regcontext.dr7

    else:
        raise NotImplementedError(f"Unsupported register: {ttreg.getName()} ({ttregId})")

def _checkingAndFix(ctx: triton.TritonContext, ins: Supertrace.InstructionRecord, ttreg, dataNotMatchedWarning: bool) -> None:
    if (ctx.getArchitecture() == triton.ARCH.X86_64):
        RegRealVal = getRegValByTTReg64(ins, ttreg)
    elif (ctx.getArchitecture() == triton.ARCH.X86):
        RegRealVal = getRegValByTTReg32(ins, ttreg)
    else:
        raise NotImplementedError("Unsupported architecture")
    if (RegRealVal != ctx.getConcreteRegisterValue(ttreg)):
        if not ctx.isRegisterSymbolized(ttreg):
            ctx.setConcreteRegisterValue(ttreg, RegRealVal)
        if (dataNotMatchedWarning):
            print(f"[ins dbgid {hex(ins.dbg_id)}] reg {ttreg.getName()} value not matched warning, now fixed")


def ByteMask(nbytes: int) -> int:
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    return (1 << (nbytes * 8)) - 1

def compatibleProcessing(
    ctx: triton.TritonContext,
    ttins: triton.Instruction,
    ins: Supertrace.InstructionRecord,
    nextIns: Supertrace.InstructionRecord,
    regCheckFix: bool,
    dataNotMatchedWarning: bool
):
    for memAcc in ins.mem_accs:
        if (memAcc.acc_size == 0):
            continue
        if (memAcc.type != Supertrace.AccessType.READ):
            continue

        if (not ctx.isMemorySymbolized(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size))):
            data = memAcc.old_data & ByteMask(memAcc.acc_size)
            if (ctx.getConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size)) != data):
                # if (dataNotMatchedWarning):
                #     print(f"[ins dbgid {hex(ins.dbg_id)}] read memory not matched warning")
                ctx.setConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size), data)
        else:
            for i in range(memAcc.acc_size): # acc_address: 被读取的地址
                if (ctx.isMemorySymbolized(triton.MemoryAccess(memAcc.acc_address + i, 1))):
                    continue
                oldby = (memAcc.old_data >> (i * 8)) & 0xFF
                if (ctx.getConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address + i, 1)) != oldby):
                    # if (dataNotMatchedWarning):
                    #     print(f"[ins dbgid {hex(ins.dbg_id)}] read memory not matched warning")
                    ctx.setConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address + i, 1), oldby)

    exception = ctx.buildSemantics(ttins)
    if (exception != triton.EXCEPTION.NO_FAULT):
        print(f"[ins dbgid {hex(ins.dbg_id)}] processing exception occur warning! {exception}")
    
    if (nextIns is not None):
        if (ttins.getType() == triton.OPCODE.X86.CPUID):
            if (ctx.getArchitecture() == triton.ARCH.X86_64):
                ctx.setConcreteRegisterValue(ctx.registers.rax, nextIns.reg_dump64.regcontext.cax)
                ctx.setConcreteRegisterValue(ctx.registers.rbx, nextIns.reg_dump64.regcontext.cbx)
                ctx.setConcreteRegisterValue(ctx.registers.rcx, nextIns.reg_dump64.regcontext.ccx)
                ctx.setConcreteRegisterValue(ctx.registers.rdx, nextIns.reg_dump64.regcontext.cdx)
            elif (ctx.getArchitecture() == triton.ARCH.X86):
                ctx.setConcreteRegisterValue(ctx.registers.eax, nextIns.reg_dump32.regcontext.cax)
                ctx.setConcreteRegisterValue(ctx.registers.ebx, nextIns.reg_dump32.regcontext.cbx)
                ctx.setConcreteRegisterValue(ctx.registers.ecx, nextIns.reg_dump32.regcontext.ccx)
                ctx.setConcreteRegisterValue(ctx.registers.edx, nextIns.reg_dump32.regcontext.cdx)
            else:
                raise NotImplementedError("Unsupported architecture")
        elif (ttins.getType() == triton.OPCODE.X86.RDTSC):
            if (ctx.getArchitecture() == triton.ARCH.X86_64):
                ctx.setConcreteRegisterValue(ctx.registers.rax, nextIns.reg_dump64.regcontext.cax)
                ctx.setConcreteRegisterValue(ctx.registers.rdx, nextIns.reg_dump64.regcontext.cdx)
            elif (ctx.getArchitecture() == triton.ARCH.X86):
                ctx.setConcreteRegisterValue(ctx.registers.eax, nextIns.reg_dump32.regcontext.cax)
                ctx.setConcreteRegisterValue(ctx.registers.edx, nextIns.reg_dump32.regcontext.cdx)
            else:
                raise NotImplementedError("Unsupported architecture")
        elif (ttins.getType() == triton.OPCODE.X86.RDTSCP):
            if (ctx.getArchitecture() == triton.ARCH.X86_64):
                ctx.setConcreteRegisterValue(ctx.registers.rax, nextIns.reg_dump64.regcontext.cax)
                ctx.setConcreteRegisterValue(ctx.registers.rcx, nextIns.reg_dump64.regcontext.ccx)
                ctx.setConcreteRegisterValue(ctx.registers.rdx, nextIns.reg_dump64.regcontext.cdx)
            elif (ctx.getArchitecture() == triton.ARCH.X86):
                ctx.setConcreteRegisterValue(ctx.registers.eax, nextIns.reg_dump32.regcontext.cax)
                ctx.setConcreteRegisterValue(ctx.registers.ecx, nextIns.reg_dump32.regcontext.ccx)
                ctx.setConcreteRegisterValue(ctx.registers.edx, nextIns.reg_dump32.regcontext.cdx)
            else:
                raise NotImplementedError("Unsupported architecture")
        elif (ttins.getType() == triton.OPCODE.X86.RDSEED or ttins.getType() == triton.OPCODE.X86.RDRAND):
            opreg: triton.Register = ttins.getOperands()[0]
            if (ctx.getArchitecture() == triton.ARCH.X86_64):
                ctx.setConcreteRegisterValue(opreg, getRegValByTTReg64(nextIns, opreg))
            elif (ctx.getArchitecture() == triton.ARCH.X86):
                ctx.setConcreteRegisterValue(opreg, getRegValByTTReg32(nextIns, opreg))
            else:
                raise NotImplementedError("Unsupported architecture")
    
    for memAcc in ins.mem_accs:
        if (memAcc.acc_size == 0):
            continue
        if (memAcc.type != Supertrace.AccessType.WRITE):
            continue
        if (not ctx.isMemorySymbolized(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size))):
            newdata = memAcc.new_data & ByteMask(memAcc.acc_size)
            check = ctx.getConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size))
            if (check != newdata):
                if (dataNotMatchedWarning):
                    print(f"[ins dbgid {hex(ins.dbg_id)}] write memory not matched warning")
                ctx.setConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address, memAcc.acc_size), newdata)
        else:
            for i in range(memAcc.acc_size): # acc_address: 被写入的地址
                if (ctx.isMemorySymbolized(triton.MemoryAccess(memAcc.acc_address + i, 1))):
                    continue
                newby = (memAcc.new_data >> (i * 8)) & 0xFF
                checkby = ctx.getConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address + i, 1))
                if (checkby != newby):
                    if (dataNotMatchedWarning):
                        print(f"[ins dbgid {hex(ins.dbg_id)}] write memory not matched warning")
                    ctx.setConcreteMemoryValue(triton.MemoryAccess(memAcc.acc_address + i, 1), newby)
    
    if (regCheckFix and (nextIns is not None)):
        if (ctx.getArchitecture() == triton.ARCH.X86_64):
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rax)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rbx)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rcx)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rdx)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rsi)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rdi)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rsp)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rbp)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.rip)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.eflags)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r8)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r9)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r10)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r11)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r12)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r13)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r14)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.r15)
        elif (ctx.getArchitecture() == triton.ARCH.X86):
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.eax)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.ebx)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.ecx)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.edx)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.esi)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.edi)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.esp)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.ebp)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.eip)
            _checkingAndFix(ctx=ctx, ins=nextIns, dataNotMatchedWarning=dataNotMatchedWarning, ttreg=ctx.registers.eflags)
        else:
            raise NotImplementedError("Unsupported architecture")
        
def checkIndirectIns(ttins: triton.Instruction) -> bool:
    if (ttins.isControlFlow()):
        brtype = ttins.getType()
        if (brtype == triton.OPCODE.X86.JMP or brtype == triton.OPCODE.X86.RET or brtype == triton.OPCODE.X86.CALL):
            if (brtype != triton.OPCODE.X86.RET):
                ops = ttins.getOperands()
                assert(len(ops) == 1)
                return ops[0].getType() != triton.OPERAND.IMM
            return True
        else:
            return False
    else:
        return False
    
def getMemoryMapByAddr(trace: Supertrace.TraceData, addr: int) -> Supertrace.MemoryMapInfo:
    memoryMaps = trace.user.meta.getMemoryMaps()
    for mm in memoryMaps:
        if (mm.addr <= addr <= (mm.addr + mm.size)):
            return mm
    return None