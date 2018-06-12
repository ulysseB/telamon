use codegen::*;

use ir::{self, op, Type};

trait Printer {
    fn print_binop(op1: &op::Operand, op2: &op::Operand) -> String;

    fn print_mul(round: op::Rounding, op1: &op::Operand, op2: &op::Operand) -> String;

    fn print_mad(round: op::Rounding, op1: &op::Operand, op2: &op::Operand, op3: &op::Operand) -> String;

    fn print_mov(op: &op::Operand) -> String;

    fn print_ld(addr: &op::Operand, access: AccessPattern) -> String;

    fn print_st(op1: &op::Operand, op1: &op::Operand, b: bool, ) -> String;

    fn print_cast(op1: &op::Operand, t: &Type) -> String;
}
