use codegen::*;
use device::x86::{Cpu, Namer};
use ir::{self, dim, op, Operand, Size, Type};
use itertools::Itertools;
use search_space::{DimKind, Domain, InstFlag};
use std::fs::File;
use std::io::Read;
// TODO(cc_perf): avoid concatenating strings.

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
    //let assignement = format!("{} =", namer.name_inst(inst).to_string());
    match *inst.operator() {
        op::Add(ref lhs, ref rhs, _) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} {} + {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Sub(ref lhs, ref rhs, _) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} {} - {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mul(ref lhs, ref rhs, _, return_type) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} {} * {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, _ing) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} {} * {} + {};", assignement, namer.name_op(mul_lhs), namer.name_op(mul_rhs), namer.name_op(add_rhs))
        },
        op::Div(ref lhs, ref rhs, _) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} {} / {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mov(ref op) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} {};", assignement, namer.name_op(op).to_string())
        },
        op::Ld(ld_type, ref addr, _) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} *(uint8_t *){};", assignement, namer.name_op(addr))
        },
        op::St(ref addr, ref val, _,  _) => {
            format!("*(uint8_t *){} = {};", 
                    namer.name_op(addr),
                    namer.name_op(val).to_string())
        },
        op::Cast(ref op, t) => {
            let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            format!("{} ({}) {};", assignement, cpu_type(&t), namer.name_op(op))
        },
        op::TmpLd(..) | op::TmpSt(..) => panic!("non-printable instruction")
    }
}

/// Prints a cfg.
fn cfg<'a>(fun: &Function, c: &Cfg<'a>, namer: &mut NameMap) -> String {
    match *c {
        Cfg::Root(ref cfgs) => cfg_vec(fun, cfgs, namer),
        Cfg::Loop(ref dim, ref cfgs) => cpu_loop(fun, dim, cfgs, namer),
        Cfg::Barrier => String::new(),
        Cfg::Instruction(ref i) => inst(i, namer, fun),
        Cfg::ParallelInductionLevel(ref level) =>
            String::new(),
    }
}

/// Prints a vector of cfgs.
fn cfg_vec(fun: &Function, cfgs: &[Cfg], namer: &mut NameMap) -> String {
    cfgs.iter().map(|c| cfg(fun, c, namer)).collect_vec().join("\n  ")
}


fn var_decls(namer: &Namer) -> String {
    let print_decl = |(&t, &n)| {
        let prefix = Namer::gen_prefix(&t);
        let mut s = format!("{} ", cpu_type(&t));
        s.push_str(&(0..n).map(|i| format!("{}{}", prefix, i)).collect_vec().join(", "));
        s.push_str(";\n  ");
        s
    };
    namer.num_var.iter().map(print_decl).collect_vec().join("\n  ")
}

fn standard_loop(fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap) -> String {
    let idx = namer.name_index(dim.id()).to_string();
    let ind_levels = dim.induction_levels().iter();
    let (var_init, var_step): (String, String) = ind_levels.map(|level| {
        let t = cpu_type(&level.t());
        let dim_id = level.increment.map(|(dim, _)| dim);
        let ind_var = namer.name_induction_var(level.ind_var, dim_id);
        let base_components = level.base.components().map(|v| namer.name(v));
        let init = match base_components.collect_vec()[..] {
            [ref base] => format!("{} = {};\n  ", ind_var, base),
            [ref lhs, ref rhs] =>
                format!(" {} = {} + {};\n  ", ind_var, lhs, rhs),
            _ => panic!(),
        };
        let step = if let Some((_, increment)) = level.increment {
            let step = namer.name_size(increment, level.t());
            format!("{} += {};\n  ", ind_var, step)
        } else { String::new() };
        (init, step)
    }).unzip();
    let loop_id = namer.gen_loop_id();
    format!(include_str!("template/loop.c.template"),
    id = loop_id,
    body = cfg_vec(fun, cfgs, namer),
    idx = idx,
    size = namer.name_size(dim.size(), Type::I(32)),
    induction_var_init = var_init,
    induction_var_step = var_step,
    )
}

fn cpu_loop(fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap)
    -> String
{
    match dim.kind() {
        DimKind::LOOP => {
            standard_loop(fun, dim, cfgs, namer)
        }
        DimKind::UNROLL => {standard_loop(fun, dim, cfgs, namer)}
        DimKind::VECTOR => {standard_loop(fun, dim, cfgs, namer)}
        _ => { unimplemented!() }
    }
}


/// Prints a `Function`.
pub fn function(function: &Function) -> String {
    let mut namer = Namer::default();
    let (param_decls, body);
    {
        let name_map = &mut NameMap::new(function, &mut namer);
        param_decls = function.device_code_args()
            .map(|v| param_decl(v, name_map))
            .collect_vec().join(",\n  ");
        body = cfg(function, function.cfg(), name_map);
    }
    let var_decls = var_decls(&namer);
    format!(include_str!("template/device.c.template"),
            name = function.name,
            params = param_decls,
            var_decls = var_decls,
            body = body
           )
}
