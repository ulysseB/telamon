use codegen::*;
use device::x86::cpu::Cpu;
use ir::{self, dim, op, Operand, Size, Type};
use search_space::{DimKind, Domain, InstFlag};
use std::fs::File;
use std::io::Read;
// TODO(cc_perf): avoid concatenating strings.

/// Prints a `Function`.
pub fn function(function: &Function, cpu: &Cpu) -> String {

    let mut f = File::open("template/hello_world.c").expect("File not found");
    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .expect("something went wrong reading");
    contents
}

/// Prints a `Type` for the host.
fn cpu_type(t: &Type) -> &'static str {
    match *t {
        Type::Void => "void",
        Type::PtrTo(..) => " void *",
        Type::F(32) => "float",
        Type::F(64) => "double",
        Type::I(8) => "int8_t",
        Type::I(16) => "int16_t",
        Type::I(32) => "int32_t",
        Type::I(64) => "int64_t",
        ref t => panic!("invalid type for the host: {}", t)
    }
}

fn param_decl(param: &ParamVal, namer: &NameMap) -> String {
    format!(
        "{t} {name}",
        t = cpu_type(&param.t()),
        name = namer.name_param(param.key()),
        )
}

/// Prints an instruction.
fn inst(inst: &Instruction, namer: &mut NameMap, fun: &Function) -> String {
    let assignement = format!("{} =", namer.name_inst(inst).to_string());
    match *inst.operator() {
        op::Add(ref lhs, ref rhs, _) => {
            format!("{} {} + {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Sub(ref lhs, ref rhs, _) => {
            format!("{} {} - {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mul(ref lhs, ref rhs, _, return_type) => {
            format!("{} {} * {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, _ing) => {
            format!("{} {} * {} + {};", assignement, namer.name_op(mul_lhs), namer.name_op(mul_rhs), namer.name_op(add_rhs))
        },
        op::Div(ref lhs, ref rhs, _) => {
            format!("{} {} / {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mov(ref op) => {
            format!("{} {};", assignement, namer.name_op(op).to_string())
        },
        op::Ld(ld_type, ref addr, _) => {
            format!("{} *({} *){}", assignement, cpu_type(&ld_type), namer.name_op(addr))
        },
        op::St(ref addr, ref val, _,  _) => {
            format!("*({} *){} = {}", 
                    cpu_type(&addr.t()), 
                    namer.name_op(addr),
                    namer.name_op(val).to_string())
        },
        op::Cast(ref op, t) => {
            format!("{} ({}) {}", assignement, cpu_type(&t), namer.name_op(op))
        },
        op::TmpLd(..) | op::TmpSt(..) => panic!("non-printable instruction")
    }
}

fn cpu_loop(fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap)
    -> String
{
    match dim.kind() {
        DimKind::LOOP => {
            let idx = namer.name_index(dim.id()).to_string();
            String::from("goto ")
        }
        _ => { String::from("") }
    }
}
