use codegen;
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
        Type::PtrTo(..) => " uint8_t *",
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
    let name = namer.name_param(param.key());
    match param {
        ParamVal::External(par, par_type) => format!("{} {}", cpu_type(par_type), name),
        ParamVal::Size(_) => format!("uint32_t {}", name),
        ParamVal::GlobalMem(_, _, par_type) => format!("{} {}", cpu_type(par_type), name),
    }
    //format!(
    //    "{t} {name}",
    //    t = cpu_type(&param.t()),
    //    name = namer.name_param(param.key()),
    //    )
}

/// Prints an instruction.
fn inst(inst: &Instruction, namer: &mut NameMap, fun: &Function) -> String {
    //let assignement = format!("{} =", namer.name_inst(inst).to_string());
    match *inst.operator() {
        op::Add(ref lhs, ref rhs, _) => {
            let assignement = format!("{} = ",namer.name_inst(inst));
            format!("{} {} + {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
            //format!("{} {} + {};", assignement, namer.indexed_op_name(lhs), namer.indexed_op_name(rhs))
        },
        op::Sub(ref lhs, ref rhs, _) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
            format!("{} {} - {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mul(ref lhs, ref rhs, _, return_type) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
            format!("{} {} * {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, _ing) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
            format!("{} {} * {} + {};", assignement, namer.name_op(mul_lhs), namer.name_op(mul_rhs), namer.name_op(add_rhs))
        },
        op::Div(ref lhs, ref rhs, _) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
            format!("{} {} / {};", assignement, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mov(ref op) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
            format!("{} {};", assignement, namer.name_op(op).to_string())
        },
        op::Ld(ld_type, ref addr, _) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
            //format!("{} *(uint8_t *){};", assignement, namer.name_op(addr))
            format!("{} ({})*{};", assignement, cpu_type(&ld_type), namer.name_op(addr))
        },
        op::St(ref addr, ref val, _,  _) => {
            format!("*(uint8_t *){} = (uint_8 *)&{};", 
                    namer.name_op(addr),
                    namer.name_op(val).to_string())
        },
        op::Cast(ref op, t) => {
            //let assignement = format!("{} = ",namer.gen_name(inst.operator().t()));
            let assignement = format!("{} = ",namer.name_inst(inst));
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
        match t {
            ir::Type::PtrTo(..) => String::new(),
            _ => {
                let prefix = Namer::gen_prefix(&t);
                let mut s = format!("{} ", cpu_type(&t));
                s.push_str(&(0..n).map(|i| format!("{}{}", prefix, i)).collect_vec().join(", "));
                s.push_str(";\n  ");
                s
            }
        }
    };
    let mut ptr_decl = String::from("uint8_t  ");
    ptr_decl.push_str(&(0..namer.num_glob_ptr).map( |i| format!("*ptr{}", i)).collect_vec().join(", "));
    ptr_decl.push_str(&";\n");
    let other_var_decl = namer.num_var.iter().map(print_decl).collect_vec().join("\n  ");
    ptr_decl.push_str(&other_var_decl);
    ptr_decl
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
    let (param_decls, body, ld_params);
    {
        let name_map = &mut NameMap::new(function, &mut namer);
        param_decls = function.device_code_args()
            .map(|v| param_decl(v, name_map))
            .collect_vec().join(",\n  ");
        body = cfg(function, function.cfg(), name_map);
        ld_params = function.device_code_args().map(|val| {
            format!("{var_name} = {name};",
                    var_name = name_map.name_param_val(val.key()),
                    name = name_map.name_param(val.key()))
        }).collect_vec().join("\n  ");
    }
    let var_decls = var_decls(&namer);
    format!(include_str!("template/device.c.template"),
            name = function.name,
            ld_params = ld_params,
            params = param_decls,
            var_decls = var_decls,
            body = body
           )
}

fn fun_params_cast(function: &Function) -> String {
    function.device_code_args()
        .enumerate()
        .map(|(i, v)| match v {
            ParamVal::External(_, par_type) => format!("{t} p{i} = *({t}*)*(args + {i})", 
                                                       t = cpu_type(par_type), i = i),
            ParamVal::Size(size) => format!("uint32_t p{i} = *(uint32_t*)*(args + {i})", i = i),
            // Are we sure we know the size at compile time ? I think we do
            ParamVal::GlobalMem(_, size, par_type) => format!("{t} p{i} = *({t}*)*(args + {i})", 
                                                              t = cpu_type(par_type), i = i)
        }
        )
        .collect_vec()
        .join(";\n  ")
} 

fn params_call(function: &Function) -> String {
    function.device_code_args()
        .enumerate().map(|x| x.0)
        .map(|i| format!("p{}", i))
        .collect_vec()
        .join(", ")
}

pub fn wrapper_function(func: &Function) -> String {
    let fun_str = function(func);
    let fun_params = params_call(func);
    format!(include_str!("template/host.c.template"),
            fun_name = func.name,
            fun_str = fun_str,
            fun_params_cast = fun_params_cast(func),
            fun_params = fun_params,
           )
}
