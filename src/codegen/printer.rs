use codegen::*;
use utils::*;
use itertools::Itertools;
use search_space::InstFlag;

use ir::{self, op, Type};
use search_space::DimKind;

pub trait Printer {

    /// Get a proper string representation of an integer in target language
    fn get_int(&self, n: u32) -> String;

    /// Get a proper string representation of an float in target language
    fn get_float(&self, f: f64) -> String;

    /// Print a type in the backend
    fn get_type(&self, t: Type) -> String;

    /// Print return_id = op1 op op2
    fn print_binop(&mut self, return_id: &str, op_type: ir::BinOp, op1: &str, op2: &str, return_type: Type, round: op::Rounding); 

    /// Print return_id = op1 * op2
    fn print_mul(&mut self, return_id: &str, round: op::Rounding, op1: &str, type_op1: Type, op2: &str, type_op2: Type, return_type: Type);

    /// Print return_id = mlhs * mrhs + arhs
    fn print_mad(&mut self, return_id: &str, round: op::Rounding, mlhs: &str, type_mlhs: Type, mrhs: &str, type_mrhs: Type, arhs: &str, type_arhs: Type, return_type: Type);

    /// Print return_id = op 
    fn print_mov(&mut self, return_id: &str, op: &str, return_type: Type);

    /// Print return_id = load [addr] 
    fn print_ld(&mut self, return_id: &str, cast_type: Type,  addr: &str, return_type: Type, flag: InstFlag);

    /// Print store val [addr] 
    fn print_st(&mut self, addr: &str, val: &str, val_type: &str, mem_flag: InstFlag);

    /// Print if (cond) store val [addr] 
    fn print_cond_st(&mut self, addr: &str, val: &str, cond: &str, val_type: &str, mem_flag: InstFlag);

    /// Print return_id = (val_type) val
    fn print_cast(&mut self, return_id: &str, op1: &str, t: Type, round: op::Rounding);

    /// print a label where to jump
    fn print_label(&mut self, label_id: &str);

    /// Print return_id = op1 && op2
    fn print_and(&mut self, return_id: &str, op1: &str, op2: &str);

    /// Print return_id = op1 || op2
    fn print_or(&mut self, return_id: &str, op1: &str, op2: &str);

    /// Print return_id = op1 == op2
    fn print_equal(&mut self, return_id: &str, op1: &str, op2: &str);

    /// Print return_id = op1 < op2
    fn print_lt(&mut self, return_id: &str, op1: &str, op2: &str);

    /// Print return_id = op1 > op2
    fn print_gt(&mut self, return_id: &str, op1: &str, op2: &str);

    /// Print if (cond) jump label(label_id)
    fn print_cond_jump(&mut self, label_id: &str, cond: &str);

    /// Print wait on all threads
    fn print_sync(&mut self);

    // TODO(cleanup): remove this function once values are preprocessed by codegen. If values
    // are preprocessed, types will be already lowered.
    fn lower_type(t: ir::Type, fun: &Function) -> ir::Type {
        unwrap!(fun.space().ir_instance().device().lower_type(t, fun.space()))
    }

    fn cfg_vec(&mut self, fun: &Function, cfgs: &[Cfg], namer: &mut NameMap) {
        for c in cfgs.iter() {
            self.cfg( fun, c, namer);
        }
    }

    /// Prints a cfg.
    fn cfg<'a>(&mut self, fun: &Function, c: &Cfg<'a>, namer: &mut NameMap) {
        match *c {
            Cfg::Root(ref cfgs) => self.cfg_vec( fun, cfgs, namer),
            Cfg::Loop(ref dim, ref cfgs) => self.gen_loop( fun, dim, cfgs, namer),
            Cfg::Threads(ref dims, ref ind_levels, ref inner) => {
                self.enable_threads( fun, dims, namer);
                for level in ind_levels {
                    self.parallel_induction_level( level, namer);
                }
                self.cfg_vec( fun, inner, namer);
                self.print_sync();
            }
            Cfg::Instruction(ref i) => self.inst( i, namer, fun),
        }
    }

    /// Prints a multiplicative induction var level.
    fn parallel_induction_level(&mut self, level: &InductionLevel, namer: &NameMap) {
        let dim_id = level.increment.map(|(dim, _)| dim);
        let ind_var = namer.name_induction_var(level.ind_var, dim_id);
        let base_components =  level.base.components().map(|v| namer.name(v)).collect_vec();
        if let Some((dim, increment)) = level.increment {
            let index = namer.name_index(dim);
            let step = namer.name_size(increment, Type::I(32));
            match base_components[..] {
                // TODO get a sensible type for index and step rather than just random I32
                [] => self.print_mul(&ind_var, op::Rounding::Exact, &index, Type::I(32), &step, Type::I(32), Type::I(32)),
                [ref base] => self.print_mad(&ind_var, op::Rounding::Exact, &index, Type::I(32), &step,Type::I(32), &base, Type::I(32), Type::I(32)),
                _ => panic!()
            }
        } else {
            match base_components[..] {
                [] => {let zero = self.get_int(0); self.print_mov(&ind_var, &zero, Type::I(32));}
                [ref base] =>  self.print_mov(&ind_var, &base, Type::I(32)),
                [ref lhs, ref rhs] =>  self.print_binop(&ind_var, ir::BinOp::Add, &lhs, &rhs, Type::I(32), op::Rounding::Exact),
                _ => panic!()
            }
        }
    }

    /// Change the side-effect guards so that only the specified threads are enabled.
    fn enable_threads(&mut self, fun: &Function, threads: &[bool], namer: &mut NameMap) {
        let mut guard = None;
        for (&is_active, dim) in threads.iter().zip_eq(fun.thread_dims().iter()) {
            if is_active { continue; }
            let new_guard = namer.gen_name(ir::Type::I(1));
            let index = namer.name_index(dim.id());
            self.print_equal(&new_guard, index, &"0");
            if let Some(ref guard) = guard {
                self.print_and(&guard as &str, &guard as &str, &new_guard);
            } else {
                guard = Some(new_guard);
            };
        }
        namer.set_side_effect_guard(guard.map(RcStr::new));
    }


    fn gen_loop(&mut self, fun: &Function, dim: &Dimension, cfgs:
                                &[Cfg], namer: &mut NameMap)
    {
        match dim.kind() {
            DimKind::LOOP => {
                self.standard_loop(fun, dim, cfgs, namer)
            }
            DimKind::UNROLL => {self.unroll_loop(fun, dim, cfgs, namer)}
            DimKind::VECTOR => {self.unroll_loop(fun, dim, cfgs, namer)}
            _ => { unimplemented!() }
        }
    }

    fn standard_loop(&mut self, fun: &Function, dim: &Dimension, cfgs:
                                     &[Cfg], namer: &mut NameMap) {
        let idx = namer.name_index(dim.id()).to_string();
        let zero = self.get_int(0);
        self.print_mov(&idx, &zero, Type::I(32));
        let mut ind_var_vec = vec![];
        let loop_id = namer.gen_loop_id();
        let ind_levels = dim.induction_levels();
        for level in ind_levels.iter() {
            let dim_id = level.increment.map(|(dim, _)| dim);
            let ind_var = namer.name_induction_var(level.ind_var, dim_id);
            let base_components = level.base.components().map(|v| namer.name(v));
            match base_components.collect_vec()[..] {
                [ref base] => self.print_mov(&ind_var, &base, Type::I(32)),
                [ref lhs, ref rhs] =>
                    self.print_binop( &ind_var, ir::BinOp::Add, &lhs, &rhs, Type::I(32), op::Rounding::Exact),
                _ => panic!(),
            };
            ind_var_vec.push(ind_var.into_owned());
        }
        self.print_label(&loop_id.to_string());
        self.cfg_vec(fun, cfgs, namer);
        for (level, ind_var) in ind_levels.iter().zip_eq(ind_var_vec) {
            if let Some((_, increment)) = level.increment {
                let step = namer.name_size(increment, level.t());
                self.print_binop(&ind_var, ir::BinOp::Add, &ind_var, &step, Type::I(32), op::Rounding::Exact);
            };
        }
        let one = self.get_int(1);
        self.print_binop(&idx, op::BinOp::Add, &idx, &one, Type::I(32), op::Rounding::Exact);
        let gt_cond = namer.gen_name(ir::Type::I(32));
        self.print_lt(&gt_cond, &idx, &namer.name_size(dim.size(), Type::I(32)));
        self.print_cond_jump(&loop_id.to_string(), &gt_cond);
    }

    fn unroll_loop(&mut self, fun: &Function, dim: &Dimension, cfgs: &[Cfg], namer: &mut NameMap) {
        //let mut body = Vec::new();
        let mut incr_levels = Vec::new();
        for level in dim.induction_levels() {
            let dim_id = level.increment.map(|(dim, _)| dim);
            let ind_var = namer.name_induction_var(level.ind_var, dim_id).to_string();
            let base_components = level.base.components().map(|v| namer.name(v));
            let base = match base_components.collect_vec()[..] {
                [ref base] => base.to_string(),
                [ref lhs, ref rhs] => {
                    let tmp = namer.gen_name(level.t());
                    self.print_binop(&tmp, ir::BinOp::Add, lhs, rhs, Type::I(32), op::Rounding::Exact);
                    tmp
                },
                _ => panic!(),
            };
            self.print_mov(&ind_var, &base, Type::I(32));
            if let Some((_, incr)) = level.increment {
                incr_levels.push((level, ind_var, incr, base));
            }
        }
        for i in 0..unwrap!(dim.size().as_int()) {
            namer.set_current_index(dim, i);
            if i > 0 {
                for &(level, ref ind_var, incr, ref base) in &incr_levels {
                    if let Some(step) = incr.as_int() {
                        let stepxi = self.get_int(step * i);
                        self.print_binop(ind_var, ir::BinOp::Add, &stepxi, &base, Type::I(32), op::Rounding::Exact);
                    } else {
                        let step = namer.name_size(incr, level.t());
                        self.print_binop(&ind_var, ir::BinOp::Add, &step, &ind_var, Type::I(32), op::Rounding::Exact);
                    };
                }
            }
            self.cfg_vec(fun, cfgs, namer);
        }
        namer.unset_current_index(dim);
    }

    fn privatise_global_block(&mut self, block: &InternalMemBlock,
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
                self.print_mad(&var, op::Rounding::Nearest, &old_var, Type::I(32), &size, Type::I(32), &idx, Type::I(32), Type::I(32));
                var
            });
        self.print_mad(&addr, op::Rounding::Nearest, &var, Type::I(32), &size, Type::I(32), &addr, Type::I(32), Type::I(32));
    }

    /// Prints an instruction.
    fn inst(&mut self, inst: &Instruction, namer: &mut NameMap, fun: &Function ) {
        match *inst.operator() {
            op::BinOp(op, ref lhs, ref rhs, round) => {
                self.print_binop(&namer.name_inst(inst), op, &namer.name_op(lhs), &namer.name_op(rhs), inst.t(), round)
            }
            op::Mul(ref lhs, ref rhs, round, _) => {
                self.print_mul(&namer.name_inst(inst), round, &namer.name_op(lhs), Self::lower_type(lhs.t(), fun), &namer.name_op(rhs), Self::lower_type(rhs.t(), fun), inst.t())
            },
            op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, round) => {
                self.print_mad(&namer.name_inst(inst), round, &namer.name_op(mul_lhs), mul_lhs.t(), &namer.name_op(mul_rhs), mul_rhs.t(), &namer.name_op(add_rhs), add_rhs.t(), inst.t())
            },
            op::Mov(ref op) => {

                self.print_mov(&namer.name_inst(inst), &namer.name_op(op), Self::lower_type(inst.t(), fun))
            },
            op::Ld(ld_type, ref addr, _) => {

                //let ld_type = self.get_type(ld_type);
                self.print_ld(&namer.name_inst(inst), ld_type, &namer.name_op(addr), inst.t(), unwrap!(inst.mem_flag()))
            },
            op::St(ref addr, ref val, _,  _) => {
                let op_type = val.t();
                let str_type = self.get_type(op_type);
                let guard = if inst.has_side_effects() {
                    namer.side_effect_guard()
                } else { None };
                if let Some(ref pred) = guard {
                    self.print_cond_st(&namer.name_op(addr), &namer.name_op(val), pred, &str_type, unwrap!(inst.mem_flag()));
                } else {self.print_st(&namer.name_op(addr), &namer.name_op(val), &str_type, unwrap!(inst.mem_flag()));};
            },
            op::Cast(ref op, t) => {
                let rounding = match (op.t(), t) {
                    (Type::F(_), Type::I(_)) => op::Rounding::Nearest,
                    (Type::I(_), Type::F(_)) => op::Rounding::Nearest,
                    (Type::F(x), Type::F(y)) => if x > y {op::Rounding::Nearest} else {  op::Rounding::Exact },
                    _ => op::Rounding::Exact,
                };
                self.print_cast(&namer.name_inst(inst), &namer.name_op(op), t, rounding)
            },
            op::TmpLd(..) | op::TmpSt(..) => panic!("non-printable instruction")
        }
    }
}
