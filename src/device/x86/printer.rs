use std::fs::File;
use std::io::Read;
use device::x86::cpu::Cpu;
use codegen::Function;
// TODO(cc_perf): avoid concatenating strings.

/// Prints a `Function`.
pub fn function(function: &Function, cpu: &Cpu) -> String {

    let mut f = File::open("template/hello_world.c").expect("File not found");
    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .expect("something went wrong reading");
    contents
}

