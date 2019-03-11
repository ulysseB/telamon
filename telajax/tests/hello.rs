use std::ffi::CString;
use telajax;

const OCL_WRAPPER: &str = " void _hello_world();

__kernel void hello_world(){
    printf(\"Hello world from %d\\n\", get_global_id(0));
    _hello_world();
} ";

const KERNEL_CODE: &str = "
#include <stdio.h>

void _hello_world(){
}";

const CFLAGS: &str = "-std=c99";

const EMPTY: &str = "";

#[test]
fn hello_telajax() {
    let executor = telajax::Device::get();

    // Wrapper build
    let name = CString::new("hello_world").unwrap();
    let wrapper_code = CString::new(OCL_WRAPPER).unwrap();
    let wrapper = executor.build_wrapper(&name, &wrapper_code).unwrap();

    // Kernel build
    let kernel_code = CString::new(KERNEL_CODE).unwrap();
    let cflags = CString::new(CFLAGS).unwrap();
    let lflags = CString::new(EMPTY).unwrap();
    let mut kernel = executor
        .build_kernel(&kernel_code, &cflags, &lflags, &wrapper)
        .unwrap();

    // Executing
    executor.execute_kernel(&mut kernel).unwrap();
}
