//! C and pthread Backend,
use codegen::*;
use ir;
use itertools::Itertools;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::traits::ToPrimitive;
use num;
use search_space::DimKind;
use std::{fmt, io};
use std::borrow::Cow;
use std::sync::atomic;
use utils::*;

#[derive(Default)]
struct MppaNamer {
    num_var: HashMap<ir::Type, usize>,
    num_pointers: usize,
}

impl MppaNamer {
    fn prefix(t: &ir::Type) -> &'static str {
        match *t {
            ir::Type::I(1) => "b",
            ir::Type::I(8) => "c",
            ir::Type::I(16) => "s",
            ir::Type::I(32) => "i",
            ir::Type::I(64) => "l",
            ir::Type::F(32) => "f",
            ir::Type::F(64) => "d",
            _ => panic!("non-printable type"),
        }
    }
}

impl Namer for MppaNamer {
    fn name(&mut self, t: ir::Type) -> String {
        match t {
            ir::Type::PtrTo(..) => {
                let name = format!("ptr_{}", self.num_pointers);
                self.num_pointers += 1;
                name
            },
            t => {
                let prefix = Self::prefix(&t);
                let entry = self.num_var.entry(t).or_insert(0);
                let name = format!("{}_{}", prefix, *entry);
                *entry += 1;
                name
            }
        }
    }

    fn name_float(&self, val: &Ratio<BigInt>, len: u16) -> String {
        let f = val.numer().to_f64().unwrap() / val.denom().to_f64().unwrap();
        let (mantissa, exponent, sign) = num::Float::integer_decode(f);
        let sign = if sign == -1 { "-" } else { "" };
        let suffix = if len == 32 { "f" } else { "" };
        format!("{}0x{:x}p{:+}{}", sign, mantissa , exponent, suffix)
    }

    fn name_int(&self, val: &BigInt, _: u16) -> String { val.to_string() }

    fn name_param(&mut self, p: &ir::Parameter) -> String {
        format!("arguments.{}", p.name)
    }
}

impl fmt::Display for MppaNamer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (t, &num) in &self.num_var {
            let prefix = Self::prefix(t);
            write!(f, "  {} {}_0", type_name(t), prefix)?;
            for i in 1..num { write!(f, ", {}_{}", prefix, i)?; }
            writeln!(f, ";")?;
        }
        if self.num_pointers != 0 {
            write!(f, "  void *ptr_0")?;
            for i in 1..self.num_pointers { write!(f, ", *ptr_{}", i)?; }
            writeln!(f, ";")?;
        }
        Ok(())
    }
}

/// Prints the C code for a candidate implementation.
pub fn print(fun: &Function, time: bool, out: &mut io::Write) -> io::Result<()> {
    lazy_static! { static ref ID: atomic::AtomicUsize = atomic::AtomicUsize::new(0); }

    let mut namer = MppaNamer::default();
    let mut body = String::new();
    let thread_idxs;
    {
        let ref mut name_map = NameMap::new(fun, &mut namer);
        // Prints precomputed induction var levels.
        for ind_var in fun.induction_vars() {
            for &(dim, ref level) in &ind_var.precomputed {
                parallel_induction_level(dim, level, name_map, &mut body).unwrap();
            }
        }
        cfg(fun.cfg(), name_map, fun, &mut body).unwrap();
        let mut inner_size = 1;
        thread_idxs = fun.thread_dims().iter().rev().format_with("\n", |dims, f| {
            let idx = name_map.name_index(dims[0].id());
            let stride = inner_size;
            let size = dims[0].size().as_int().unwrap();
            inner_size *= size;
            f(&format_args!("{} = (core_id/{}) % {};", idx, stride, size))
        }).to_string();
    }
    let kernel_args = fun.params.iter().map(|p| {
        format!("{} {}", type_name(&p.t), p.name)
    }).chain({
        if time { Some("uint64_t* __time_ptr".to_string()) } else { None }
    }).format(", ");
    let arg_struct_fields = fun.params().iter().format_with("\n", |(_, p), f| {
        f(&format_args!("{} {};", type_name(&p.t), p.name))
    });
    let arg_struct_build = fun.params().iter().format_with("\n", |(val, p), f| {
        f(&format_args!("arguments.{} = ", p.name))?;
        match *val {
            ParamVal::External(p) => f(&format_args!("{};", p.name)),
            ParamVal::Size(size) => f(&print_size(size)),
            ParamVal::GlobalMem(..) => unimplemented!(), // FIXME
            ParamVal::GlobalMemSize(block) => f(&print_size(block.size())),
        }
    });
    let id = if time {
        let id = ID.fetch_add(1, atomic::Ordering::Relaxed);
        format!("int __candidate_id = {};", id)
    } else { "".to_string() };
    write!(out, include_str!("template.c"),
        candidate_id = id, // Does not works on the final binary
        def_time = if time { "#define TIMING 1" } else { "" },
        name = fun.name,
        num_threads = fun.num_threads(),
        thread_idxs = thread_idxs,
        kernel_args = kernel_args,
        arg_struct_fields = arg_struct_fields,
        arg_struct_build = arg_struct_build,
        var_decls = namer,
        body = body,
    )
}

/// Prints a multiplicative induction var level.
fn parallel_induction_level(dim: ir::DimId, level: &InductionVarLevel,
                            name_map: &NameMap, out: &mut fmt::Write) -> fmt::Result {
    let index = name_map.name_index(dim);
    let var = name_map.name_induction_var(level.var, Some(dim));
    let step = name_map.name_size(&level.increment, &ir::Type::I(32));
    match induction_level_base(level, name_map)[..] {
        [] => write!(out, "{} = {} * {};", var, step, index),
        [ref base] => write!(out, "{} = {} * {} + {};", var, step, index, base),
        _ => panic!(),
    }
}

/// Prints a size.
fn print_size(size: &ir::Size) -> String {
    let dividend = size.dividend().iter().map(|p| format!("* {}", &p.name));
    format!("{}{}/{};", size.factor(), dividend.format(""), size.divisor())
}

/// Prints the OpenCL wrapper for a candidate implementation.
pub fn print_ocl_wrapper(signature: &ir::Signature, out: &mut io::Write)
        -> io::Result<()> {
    let arg_names = signature.params.iter().format_with("", |p, f| {
            f(&format_args!("{}, ", p.name))
    });
    let cl_arg_defs = signature.params.iter().format_with("", |p, f| {
        f(&format_args!("{} {},", cl_type_name(&p.t), p.name))
    }).to_string();
    write!(out, include_str!("ocl_wrapper_template.cl"),
        name = signature.name,
        arg_names = arg_names,
        cl_arg_defs = cl_arg_defs,
    )
}

/// Prints a CFG element.
fn cfg(cfg: &Cfg, name_map: &mut NameMap, fun: &Function, out: &mut fmt::Write)
        -> fmt::Result {
    match *cfg {
        Cfg::Root(ref children) |
        Cfg::Thread(_, ref children) => cfg_vec(children, name_map, fun, out),
        Cfg::Loop(dim, DimKind::LOOP, ref children) => {
            let idx = name_map.decl_index(dim.id());
            let size = name_map.name_size(dim.size(), &ir::Type::I(32)).to_string();
            // Initialize induction variables.
            for level in &fun.dimensions()[&dim.id()].induction_vars {
                let var = name_map.name_induction_var(level.var, Some(dim.id()));
                match induction_level_base(level, name_map)[..] {
                    [ref base] => writeln!(out, "{} = {};", var, base)?,
                    [ref lhs, ref rhs] => writeln!(out, "{} = {} + {}", var, lhs, rhs)?,
                    _ => panic!(),
                }
            }
            write!(out, "for({idx}=0; {idx}<{size}; ", idx=idx, size=size)?;
            // Increment induction variables
            for level in &fun.dimensions()[&dim.id()].induction_vars {
                let var = name_map.name_induction_var(level.var, Some(dim.id()));
                let step = name_map.name_size(&level.increment, &ir::Type::I(32));
                write!(out, "{} += {}, ", var, step)?;
            }
            writeln!(out, "++{idx}) {{", idx=idx)?;
            cfg_vec(children, name_map, fun, out)?;
            writeln!(out, "}}")
        },
        Cfg::Loop(dim, DimKind::UNROLL, ref children) => {
            for i in 0..dim.size().as_int().unwrap() {
                name_map.set_current_index(dim.id(), i);
                for level in &fun.dimensions()[&dim.id()].induction_vars {
                    let var = &fun.induction_vars()[level.var.0 as usize];
                    let var_name = name_map.name_induction_var(level.var, Some(dim.id()));
                    let base = match induction_level_base(level, name_map)[..] {
                        [ref base] => base.clone(),
                        [ref lhs, ref rhs] => {
                            let tmp = name_map.gen_name(var.t.clone());
                            writeln!(out, "{} = {} + {};", tmp, lhs, rhs)?;
                            Cow::Owned(tmp)
                        },
                        _ => panic!(),
                    };
                    if i == 0 {
                        writeln!(out, "{} = {};", var_name, base)?;
                    } else if let Some(step) = level.increment.as_int() {
                        writeln!(out, "{} = {} + {};", var_name, base, step*i)?;
                    } else {
                        let step = name_map.name_size(&level.increment, &ir::Type::I(32));
                        writeln!(out, "{} += {};", var_name, step)?;
                    }
                }
                cfg_vec(children, name_map, fun, out)?;
            }
            name_map.unset_current_index(dim.id());
            Ok(())
        },
        Cfg::Instruction(inst) => instruction(inst, name_map, out),
        Cfg::Barrier =>
            writeln!(out, "  assert(!pthread_barrier_wait(&barrier));"),
        Cfg::Loop(..) => panic!("invalid loop kind"),
    }
}

/// Prints a list of CFG elements.
fn cfg_vec(cfgs: &[Cfg], name_map: &mut NameMap, fun: &Function, out: &mut fmt::Write)
        -> fmt::Result {
    for c in cfgs { cfg(c, name_map, fun, out)?; }
    Ok(())
}

/// Prints an instruction.
fn instruction(inst: &ir::Instruction, name_map: &NameMap, out: &mut fmt::Write
               ) -> fmt::Result {
    if inst.t() != ir::Type::Void { write!(out, "  {} = ", name_map.name_inst(inst))?; }
    match *inst.operator() {
        ir::op::Add(ref lhs, ref rhs, rounding) => {
            assert!(check_rounding(rounding));
            write!(out, "{} + {}", name_map.name(lhs), name_map.name(rhs))
        },
        ir::op::Sub(ref lhs, ref rhs, rounding) => {
            assert!(check_rounding(rounding));
            write!(out, "{} - {}", name_map.name(lhs), name_map.name(rhs))
        },
        ir::op::Mul(ref lhs, ref rhs, rounding, ref return_type) => {
            assert!(check_rounding(rounding));
            let t = type_name(return_type);
            write!(out, "({}){} * ({}){}", t, name_map.name(lhs), t, name_map.name(rhs))
        },
        ir::op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, rounding) => {
            // FIXME: use FMA inst
            assert!(check_rounding(rounding));
            let t = type_name(&add_rhs.t());
            let mul_lhs = name_map.name(mul_lhs);
            let mul_rhs = name_map.name(mul_rhs);
            let add_rhs = name_map.name(add_rhs);
            write!(out, "({}){} * ({}){} + {}", t, mul_lhs, t, mul_rhs, add_rhs)
        },
        ir::op::Div(ref lhs, ref rhs, rounding) => {
            assert!(check_rounding(rounding));
            write!(out, "{}/{}", name_map.name(lhs), name_map.name(rhs))
        },
        ir::op::Mov(ref op) => write!(out, "{}", name_map.name(op)),
        ir::op::Ld(ref t, ref addr, _) =>
            write!(out, "*({}*){}", type_name(t), name_map.name(addr)),
        ir::op::St(ref addr, ref val, ..) => {
            let val_type = type_name(&val.t());
            let addr = name_map.name(addr);
            write!(out, "*(({}*){}) = {}", val_type, addr, name_map.name(val))
        },
        ir::op::Cast(ref op, ir::Type::PtrTo(_)) =>
            write!(out, "(void*)(intptr_t) {}", name_map.name(op)),
        ir::op::Cast(ref op, ref t) =>
            write!(out, "({}) {}", type_name(t), name_map.name(op)),
        ir::op::TmpLd(..) | ir::op::TmpSt(..) => panic!("non-printable instruction"),
    }?;
    writeln!(out, ";")
}

/// Ensures the rounding is supported for the C backend.
fn check_rounding(rounding: ir::op::Rounding) -> bool {
    match rounding {
        ir::op::Rounding::Exact | ir::op::Rounding::Nearest => true,
        _ => false,
    }
}

/// Returns the name of a type.
fn type_name(t: &ir::Type) -> &'static str {
    match *t {
        ir::Type::Void => "void",
        ir::Type::PtrTo(..) => "void*",
        ir::Type::F(32) => "float",
        ir::Type::F(64) => "double",
        ir::Type::I(1) => "bool",
        ir::Type::I(8) => "int8_t",
        ir::Type::I(16) => "int16_t",
        ir::Type::I(32) => "int32_t",
        ir::Type::I(64) => "int64_t",
        _ => panic!("non-printable type"),
    }
}

/// Returns the name of a type.
fn cl_type_name(t: &ir::Type) -> &'static str {
    match *t {
        ir::Type::PtrTo(..) => "__global void*",
        ir::Type::I(8) => "char",
        ir::Type::I(16) => "short",
        ir::Type::I(32) => "int",
        ir::Type::I(64) => "long",
        _ => type_name(t),
    }
}
