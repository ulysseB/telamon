use codegen::*;
use utils::*;
use itertools::Itertools;

use ir::{self, op, Type};
use search_space::{Domain, DimKind};

pub trait Printer {

    /// Get a proper string representation of an integer in target language
    fn get_int(&self, n: u32) -> String;

    fn get_type(&mut self, t: &ir::Type) -> &'static str;

    fn print_binop(&mut self, return_id: &str, op_type: ir::BinOp, op1: &str, op2: &str) ;

    fn print_mul(&mut self, return_id: &str, round: op::Rounding, op1: &str, op2: &str) ;

    fn print_mad(&mut self, return_id: &str, round: op::Rounding, op1: &str, op2: &str, op3: &str) ;

    fn print_mov(&mut self, return_id: &str, op: &str) ;

    fn print_ld(&mut self, return_id: &str, cast_type: &str,  addr: &str) ;

    fn print_st(&mut self, addr: &str, val: &str) ;

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

pub fn cfg_vec<T: Printer>(printer: &mut T, fun: &Function, cfgs: &[Cfg], namer: &mut NameMap) {
  for c in cfgs.iter() {
    cfg(printer, fun, c, namer);
  }
}

/// Prints a cfg.
pub fn cfg<'a, T: Printer>(printer: &mut T, fun: &Function, c: &Cfg<'a>, namer: &mut NameMap) {
  match *c {
    Cfg::Root(ref cfgs) => cfg_vec(printer, fun, cfgs, namer),
    Cfg::Loop(ref dim, ref cfgs) => gen_loop(printer, fun, dim, cfgs, namer),
    Cfg::Threads(ref dims, ref ind_levels, ref inner) => {
      enable_threads(printer, fun, dims, namer);
      for level in ind_levels {
        parallel_induction_level(printer, level, namer);
      }
      cfg_vec(printer, fun, inner, namer);
      printer.print_sync();
    }
    Cfg::Instruction(ref i) => inst(printer, i, namer),
  }
}

/// Prints a multiplicative induction var level.
pub fn parallel_induction_level<T: Printer>(printer: &mut T, level: &InductionLevel, namer: &NameMap) {
    let dim_id = level.increment.map(|(dim, _)| dim);
    let ind_var = namer.name_induction_var(level.ind_var, dim_id);
    let base_components =  level.base.components().map(|v| namer.name(v)).collect_vec();
    if let Some((dim, increment)) = level.increment {
        let index = namer.name_index(dim);
        let step = namer.name_size(increment, Type::I(32));
        match base_components[..] {
            [] => printer.print_mul(&ind_var, op::Rounding::Nearest, &index, &step),
            [ref base] => printer.print_mad(&ind_var, op::Rounding::Nearest, &index, &step, &base),
            _ => panic!()
        }
    } else {
        match base_components[..] {
            //[] => format!("{} = 0;//Induction initialized",  ind_var),
            [] => {let zero = printer.get_int(0); printer.print_mov(&ind_var, &zero);}
            //[ref base] => format!(" {} = {};//Induction initialized", ind_var, base),
            [ref base] =>  printer.print_mov(&ind_var, &base),
            //[ref lhs, ref rhs] => format!("{} = {} + {};//Induction initialized", ind_var, lhs, rhs),
            [ref lhs, ref rhs] =>  printer.print_binop(&ind_var, ir::BinOp::Add, &lhs, &rhs),
            _ => panic!()
        }
    }
}

/// Change the side-effect guards so that only the specified threads are enabled.
pub fn enable_threads<T: Printer>(printer: &mut T, fun: &Function, threads: &[bool], namer: &mut NameMap) {
    let mut guard = None;
    for (&is_active, dim) in threads.iter().zip_eq(fun.thread_dims().iter()) {
        if is_active { continue; }
        let new_guard = namer.gen_name(ir::Type::I(1));
        let index = namer.name_index(dim.id());
        printer.print_equal(&new_guard, index, &"0");
        if let Some(ref guard) = guard {
            //unwrap!(writeln!(ops, "   {} = {} && {};", guard, guard, new_guard));
            printer.print_and(&guard as &str, &guard as &str, &new_guard);
        } else {
            guard = Some(new_guard);
        };
    }
    namer.set_side_effect_guard(guard.map(RcStr::new));
}


pub fn gen_loop<T: Printer>(printer: &mut T, fun: &Function, dim: &Dimension, cfgs:
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

pub fn standard_loop<T: Printer>(printer: &mut T,  fun: &Function, dim: &Dimension, cfgs:
                             &[Cfg], namer: &mut NameMap) {
  let idx = namer.name_index(dim.id()).to_string();
  let zero = printer.get_int(1);
  printer.print_mov(&idx, &zero);
  let mut ind_var_vec = vec![];
  let loop_id = namer.gen_loop_id();
  printer.print_label(&loop_id.to_string());
  let ind_levels = dim.induction_levels();
  for level in ind_levels.iter() {
    let dim_id = level.increment.map(|(dim, _)| dim);
    let ind_var = namer.name_induction_var(level.ind_var, dim_id);
    let base_components = level.base.components().map(|v| namer.name(v));
    match base_components.collect_vec()[..] {
      //[ref base] => format!("{} = {};//Induction variable\n  ", ind_var, base),
      [ref base] => printer.print_mov(&ind_var, &base),
      [ref lhs, ref rhs] =>
        printer.print_binop( &ind_var, ir::BinOp::Add, &lhs, &rhs),
        //format!(" {} = {} + {};\n  ", ind_var, lhs, rhs),
      _ => panic!(),
    };
    ind_var_vec.push(ind_var.into_owned());
  }
  cfg_vec(printer, fun, cfgs, namer);
  for level in ind_levels.iter() {
    let ind_var: String = ind_var_vec.pop().unwrap();
    if let Some((_, increment)) = level.increment {
      let step = namer.name_size(increment, level.t());
      printer.print_binop(&ind_var, ir::BinOp::Add, &ind_var, &step);
    };
  }
  let one = printer.get_int(1);
  printer.print_binop(&idx, op::BinOp::Add, &idx, &one);
  let gt_cond = "loop_test";
  printer.print_lt(&gt_cond, &idx, &namer.name_size(dim.size(), Type::I(32)));
  printer.print_cond_jump(&loop_id.to_string(), &gt_cond);
}

pub fn unroll_loop<T:Printer>(printer: &mut T, fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap) {
  //let mut body = Vec::new();
  let mut incr_levels = Vec::new();
  //let mut ind_var_vec = vec![];
  for level in dim.induction_levels() {
    let dim_id = level.increment.map(|(dim, _)| dim);
    let ind_var = namer.name_induction_var(level.ind_var, dim_id).to_string();
    let base_components = level.base.components().map(|v| namer.name(v));
    let base = match base_components.collect_vec()[..] {
      [ref base] => base.to_string(),
      [ref lhs, ref rhs] => {
        let tmp = namer.gen_name(level.t());
        //body.push(format!(" {} = {} + {};", tmp, lhs, rhs));
        printer.print_binop(&tmp, ir::BinOp::Add, lhs, rhs);
        tmp
      },
      _ => panic!(),
    };
    //body.push(format!("{} = {};", ind_var, base));
    printer.print_mov(&ind_var, &base);
    if let Some((_, incr)) = level.increment {
      incr_levels.push((level, ind_var, incr, base));
    }
  }
  for i in 0..unwrap!(dim.size().as_int()) {
    namer.set_current_index(dim, i);
    if i > 0 {
      for &(level, ref ind_var, incr, ref base) in &incr_levels {
        if let Some(step) = incr.as_int() {
          //format!(" {} = {} + {};", ind_var, step*i, base),
            let stepxi = printer.get_int(step * i);
          printer.print_binop(ind_var, ir::BinOp::Add, &stepxi, &base);
        } else {
          let step = namer.name_size(incr, level.t());
          //format!(" {} = {} + {};", ind_var, step, ind_var)
          printer.print_binop(&ind_var, ir::BinOp::Add, &step, &ind_var);
        };
      }
    }
    //body.push(cfg_vec(fun, cfgs, namer));
    cfg_vec(printer, fun, cfgs, namer);
  }
  namer.unset_current_index(dim);
}

pub fn privatise_global_block<T: Printer>(printer: &mut T, block: &InternalMemBlock,
                                      namer: &mut NameMap, fun: &Function) {
  if fun.block_dims().is_empty() { return ; }
  let addr = namer.name_addr(block.id());
  let size = namer.name_size(block.local_size(), Type::I(32));
  let d0 = namer.name_index(fun.block_dims()[0].id()).to_string();
  let var = fun.block_dims()[1..].iter()
    .fold(d0, |old_var, dim| {
      let var = namer.gen_name(Type::I(32));
      let size = namer.name_size(dim.size(), Type::I(32));
      let idx = namer.name_index(dim.id());
      printer.print_mad(&var, op::Rounding::Nearest, &old_var, &size, &idx);
      //insts.push(format!("{} = {} * {} + {};",
      //                  var, old_var, size, idx));
      var
    });
  printer.print_mad(&addr, op::Rounding::Nearest, &var, &size, &addr);
}

/// Prints an instruction.
pub fn inst<T: Printer>(printer: &mut T, inst: &Instruction, namer: &mut NameMap ) {
    match *inst.operator() {
        op::BinOp(op, ref lhs, ref rhs, round) => {
            assert_eq!(round, op::Rounding::Nearest);
            printer.print_binop(&namer.name_inst(inst), op, &namer.name_op(lhs), &namer.name_op(rhs))
        }
        op::Mul(ref lhs, ref rhs, round, _) => {
            assert_eq!(round, op::Rounding::Nearest);
            printer.print_mul(&namer.name_inst(inst), round, &namer.name_op(lhs), &namer.name_op(rhs))
        },
        op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, round) => {
            assert_eq!(round, op::Rounding::Nearest);
            printer.print_mad(&namer.name_inst(inst), round, &namer.name_op(mul_lhs), &namer.name_op(mul_rhs), &namer.name_op(add_rhs))
        },
        op::Mov(ref op) => {

            printer.print_mov(&namer.name_inst(inst), &namer.name_op(op))
        },
        op::Ld(ld_type, ref addr, _) => {

          let ld_type = printer.get_type(&ld_type);
          printer.print_ld(&namer.name_inst(inst), &ld_type, &namer.name_op(addr))
        },
        op::St(ref addr, ref val, _,  _) => {
            //let op_type = val.t();
            let guard = if inst.has_side_effects() {
                namer.side_effect_guard()
            } else { None };
            if let Some(ref pred) = guard {
                //format!("if({}) ", pred);
                  printer.print_cond_st(&namer.name_op(addr), &namer.name_op(val), pred);
            } else {printer.print_st(&namer.name_op(addr), &namer.name_op(val));};
        },
        op::Cast(ref op, t) => {
            printer.print_cast(&namer.name_inst(inst), &namer.name_op(op), &t)
        },
        op::TmpLd(..) | op::TmpSt(..) => panic!("non-printable instruction")
    }
}
