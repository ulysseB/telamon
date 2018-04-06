use std::process::Command;
use std::fs::File;
use std::time::Instant;
use libloading;

pub fn compile(libname: String, source_path: String, lib_path: String) {
    Command::new("gcc")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-o")
        .arg(format!("{}lib{}.so", lib_path, libname))
        .arg(source_path)
        .status()
        .expect("Could not gcc for reasons");
}

pub fn link_and_exec(lib_path: String, fun_name: String) -> f64 {
    let lib = libloading::Library::new(lib_path)
        .expect("Library not found");
    unsafe {
        let func : libloading::Symbol<unsafe extern fn() -> f64> = 
            lib.get(fun_name.as_bytes())
            .expect("Could not find symbol in library");
        let t0 = Instant::now();
        func();
        let t = Instant::now() - t0;
        t.subsec_nanos() as f64
    }
}
