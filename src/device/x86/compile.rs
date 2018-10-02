use libc;
use libloading;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::process::{Command, ExitStatus};
use std::time::Instant;

pub fn compile(mut source_file: File, lib_path: &String) -> ExitStatus {
    unwrap!(source_file.seek(SeekFrom::Start(0)));
    Command::new("gcc")
        .stdin(source_file)
        .arg("-shared")
        .arg("-fPIC")
        .arg("-o")
        .arg(lib_path)
        .arg("-xc")
        .arg("-")
        .arg("-lpthread")
        .status()
        .expect("Could not execute gcc")
}

pub fn link_and_exec(
    lib_path: &String,
    fun_name: &String,
    mut args: Vec<*mut libc::c_void>,
) -> f64
{
    let lib = libloading::Library::new(lib_path).expect("Library not found");
    unsafe {
        let func: libloading::Symbol<
            unsafe extern "C" fn(*mut *mut libc::c_void),
        > = lib
            .get(fun_name.as_bytes())
            .expect("Could not find symbol in library");
        let t0 = Instant::now();
        func(args.as_mut_ptr());
        let t = Instant::now() - t0;
        t.subsec_nanos() as f64
    }
}
