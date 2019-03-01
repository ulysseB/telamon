use telajax;
use utils::unwrap;
use std::ffi::CString;

const OCL_WRAPPER: &str =
" void _hello_world();

__kernel void hello_world(){
    printf(\"Hello world from %d\\n\", get_global_id(0));
    _hello_world();
} ";


const KERNEL_CODE: &str = "
#include <stdio.h>

void _hello_world(){
}" ;

const CFLAGS: &str = "-std=c99" ;

const EMPTY: &str = "" ;

#[test]
fn hello_telajax() {
    let executor = telajax::Device::get();

    // Wrapper build
    let name = unwrap!(CString::new("hello_world"));
    let wrapper_code = unwrap!(CString::new(OCL_WRAPPER));
    let wrapper = unwrap!(executor.build_wrapper(&name, &wrapper_code));

    // Kernel build
    let kernel_code = unwrap!(CString::new(KERNEL_CODE));
    let cflags = unwrap!(CString::new(CFLAGS));
    let lflags = unwrap!(CString::new(EMPTY));
    let mut kernel = unwrap!(executor.build_kernel(&kernel_code, &cflags, &lflags, &wrapper));
    unwrap!(executor.execute_kernel(&mut kernel));
    assert!(true);
}
