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

fn cpu_loop(fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap)
    -> String
{
    match dim.kind() {
        DimKind::LOOP => String::from("goto "),
        _ => String::from("")
    }
}
