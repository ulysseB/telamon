use std::process::{ExitStatus, Command};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;
use std;
use libloading;

pub fn compile(mut source_file: File, lib_path: &String) -> ExitStatus {
    source_file.seek(SeekFrom::Start(0));
    Command::new("gcc")
        .stdin(source_file)
        .arg("-shared")
        .arg("-fPIC")
        .arg("-o")
        .arg(lib_path)
        .arg("-xc")
        .arg("-")
        .status()
        .expect("Could not execute gcc")
}

pub fn link_and_exec(lib_path: &String, fun_name: &String) -> f64 {
    Command::new("readelf")
        .arg("-Ws")
        .arg(lib_path)
        .status();
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
