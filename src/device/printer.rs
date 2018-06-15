use codegen::*;
use std::io::prelude::*;
use std::collection::Hashmap;

use ir::{self, op, Type};

trait Printer {
    type Label;

    /// Get a proper string representation of an integer in target language
    fn get_int(n: u32) -> String;

    fn print_binop(&mut self, return_id: &str, op: op::BinOp, op1: &str, op2: &str) ;

    fn print_mul(&mut self, return_id: &str, round: op::Rounding, op1: &str, op2: &str) ;

    fn print_mad(&mut self, return_id: &str, round: op::Rounding, op1: &str, op2: &str, op3: &&str) ;

    fn print_mov(&mut self, return_id: &str, op: &str) ;

    fn print_ld(&mut self, return_id: &str, addr: &str) ;

    fn print_st(&mut self, op1: &str, op2: &str) ;

    fn print_cond_st(&mut self, addr: &str, val: &str, cond: &str) ;

    fn print_cast(&mut self, return_id: &str, op1: &str, t: &Type) ;

    fn print_label(&mut self, label_id: &str) ;

    fn print_and(&mut self, return_id: &str, op1: &str, op2: &str);

    fn print_or(&mut self, return_id: &str, op1: &str, op2: &str);

    fn print_equal(&mut self, return_id: &str, op1: &str, op2: &str);

    fn print_lt(&mut self, return_id: &str, op1: &str, op2: &str);

    fn print_gt(&mut self, return_id: &str, op1: &str, op2: &str);

    fn print_cond_jump(&mut self, label_id: &str, cond: &str) ;

    fn print_sync(&mut self) ;
}

fn cfg_vec(printer, fun: &Function, cfgs: &[Cfg], namer: &mut NameMap) {
    cfgs.iter().map(|c| cfg(printer, fun, c, namer));
}

/// Prints a cfg.
fn cfg<'a, T: Printer>(printer: &mut T, fun: &Function, c: &Cfg<'a>, namer: &mut NameMap) {
    match *c {
        Cfg::Root(ref cfgs) => cfg_vec(printer, fun, cfgs, namer),
        Cfg::Loop(ref dim, ref cfgs) => gen_loop(printer, fun, dim, cfgs, namer),
        Cfg::Threads(ref dims, ref ind_levels, ref inner) => {
                enable_threads(printer, fun, dims, namer);
                for level in ind_levels {
                    parallel_induction_level(printer, level, namer));
                }
                cfg_vec(printer, fun, inner, namer);
                printer.print_sync();
        }
        Cfg::Instruction(ref i) => inst(printer, i, namer, fun),
    }
}

/// Change the side-effect guards so that only the specified threads are enabled.
fn enable_threads<T: Printer>(printer: &mut T, fun: &Function, threads: &[bool], namer: &mut NameMap) {
    let mut guard = None;
    for (&is_active, dim) in threads.iter().zip_eq(fun.thread_dims().iter()) {
        if is_active { continue; }
        let new_guard = namer.gen_name(ir::Type::I(1));
        let index = namer.name_index(dim.id());
        printer.print_equal(new_guard, index, &"0");
        if let Some(ref guard) = guard {
            //unwrap!(writeln!(ops, "   {} = {} && {};", guard, guard, new_guard));
            printer.print_and(guard, guard, new_guard);
        } else {
            guard = Some(new_guard);
        };
    }
    namer.set_side_effect_guard(guard.map(RcStr::new));
}


fn gen_loop<T: Printer>(printer: &mut T, fun: &Function, dim: &Dimension, cfgs:
                        &[Cfg], namer: &mut NameMap)
{
    match dim.kind() {
        DimKind::LOOP => {
            standard_loop(printer,  fun, dim, cfgs, namer)
        }
        DimKind::UNROLL => {unroll_loop(printer, fun, dim, cfgs, namer)}
        DimKind::VECTOR => {unroll_loop(printer, fun, dim, cfgs, namer)}
        _ => { unimplemented!() }
    }
}

fn standard_loop<T: Printer>(printer: &T, out: &T::Out, fun: &Function, dim: &Dimension, cfgs:
                             &[Cfg], namer: &mut NameMap) {
    let idx = namer.name_index(dim.id()).to_string();
    let zero = printer.get_int(1);
    printer.print_mov(idx, &zero);
    let ind_var_vec = vec![];
    let loop_id = namer.gen_loop_id();
    printer.print_label(loop_id);
    let ind_levels = dim.induction_levels().iter();
    ind_levels.map(|level| {
        let dim_id = level.increment.map(|(dim, _)| dim);
        let ind_var = namer.name_induction_var(level.ind_var, dim_id);
        let base_components = level.base.components().map(|v| namer.name(v));
        let init = match base_components.collect_vec()[..] {
            //[ref base] => format!("{} = {};//Induction variable\n  ", ind_var, base),
            [ref base] => printer.print_mov(out, ind_var, base),
            [ref lhs, ref rhs] =>
                printer.print_binop(out, namer, op::BinOp::Add, ind_var, lhs, rhs),
                //format!(" {} = {} + {};\n  ", ind_var, lhs, rhs),
            _ => panic!(),
        };
        ind_var_vec.push(ind_var);
    });
    cfg_vec(printer, fun, cfgs, namer);
    ind_levels.rev().map(|level| {
      let ind_var = ind_var_vec.pop().unwrap();
        if let Some((_, increment)) = level.increment {
            let step = namer.name_size(increment, level.t());
            printer.print_binop(ind_var, op::BinOp::Add, ind_var, step);
        };
    });
    let one = printer.get_int(1);
    printer.print_add(idx, idx, &one);
    let gt_cond = "loop_test";
    let size = printer.get_int(namer.name_size(dim.size(), Type::I(32));
    printer.print_lt(&gt_cond, idx, size);
}

fn unroll_loop(printer: &T, out: &T::Out, fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap)-> String {
    //let mut body = Vec::new();
    let mut incr_levels = Vec::new();
    let mut ind_var_vec = vec![];
    for level in dim.induction_levels() {
        let t = cpu_type(&level.t());
        let dim_id = level.increment.map(|(dim, _)| dim);
        let ind_var = namer.name_induction_var(level.ind_var, dim_id).to_string();
        let base_components = level.base.components().map(|v| namer.name(v));
        let base = match base_components.collect_vec()[..] {
            [ref base] => base.to_string(),
            [ref lhs, ref rhs] => {
                let tmp = namer.gen_name(level.t());
                body.push(format!(" {} = {} + {};", tmp, lhs, rhs));
                tmp
            },
            _ => panic!(),
        };
        //body.push(format!("{} = {};", ind_var, base));
        if let Some((_, incr)) = level.increment {
            incr_levels.push((level, ind_var, t, incr, base));
        }
    }
    for i in 0..unwrap!(dim.size().as_int()) {
        namer.set_current_index(dim, i);
        if i > 0 {
            for &(level, ref ind_var, _, incr, ref base) in &incr_levels {
                let incr =  if let Some(step) = incr.as_int() {
                    format!(" {} = {} + {};", ind_var, step*i, base)
                } else {
                    let step = namer.name_size(incr, level.t());
                    format!(" {} = {} + {};", ind_var, step, ind_var)
                };
                body.push(incr);
            }
        }
        body.push(cfg_vec(fun, cfgs, namer));
    }
    namer.unset_current_index(dim);
    body.join("\n  ")
}

/// Prints an instruction.
fn inst<T: Printer>(printer: T, inst: &Instruction, namer: &mut NameMap ) {
    match *inst.operator() {
        op::BinOp(op, ref lhs, ref rhs, round) => {
            assert_eq!(round, op::Rounding::Nearest);
            printer.print_binop(namer.name_inst(inst), op, namer.name_op(lhs), namer.name_op(rhs))
        }
        op::Mul(ref lhs, ref rhs, round, _) => {
            assert_eq!(round, op::Rounding::Nearest);
            printer.print_mul(namer.name_inst(inst), round, namer.name_op(lhs), namer.name_op(rhs))
        },
        op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, round) => {
            assert_eq!(round, op::Rounding::Nearest);
            printer.print_mad(namer.name_inst(inst), round, namer.name_op(mul_lhs), namer.name_op(mul_rhs), namer.name_op(add_rhs))
        },
        op::Mov(ref op) => {

            printer.print_mov(namer.name_inst(inst), namer.name_op(op))
        },
        op::Ld(ld_type, ref addr, _) => {

            printer.print_ld(namer.name_inst(inst), cpu_type(&ld_type), namer.name_op(op))
        },
        op::St(ref addr, ref val, _,  _) => {
            let op_type = val.t();
            let guard = if inst.has_side_effects() {
                namer.side_effect_guard()
            } else { None };
            if let Some(ref pred) = guard {
                format!("if({}) ", pred);
                  printer.print_cond_st(namer.name_op(addr), namer.name_op(val), pred);
            } else {printer.print_st(namer.name_op(addr), namer.name_op(val));};
        },
        op::Cast(ref op, t) => {
            printer.print_mov(namer.name_inst(inst), cpu_type(&t), namer.name_op(op))
        },
        op::TmpLd(..) | op::TmpSt(..) => panic!("non-printable instruction")
    }
}
