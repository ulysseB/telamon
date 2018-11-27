use std;
use std::ffi::CStr;

use libc::{c_char, size_t};
use telamon::explorer::{self, Config};

use super::context::Context;
use super::error::TelamonStatus;
use super::search_space::SearchSpace;

/// Allocate a new explorer configuration object with suitable
/// defaults.
///
/// The resulting config object must be freed using
/// `telamon_config_free`.
#[no_mangle]
pub extern "C" fn telamon_explorer_config_new() -> *mut Config {
    Box::into_raw(Box::new(Config::default()))
}

/// Create an explorer configuration object from a JSON-encoded string.
///
/// The resulting object must be destroyed using `telamon_explorer_config_free`.
#[no_mangle]
pub unsafe extern "C" fn telamon_explorer_config_from_json(
    buf: *const c_char,
    len: size_t,
) -> *mut Config {
    let slice = std::slice::from_raw_parts(buf as *const u8, len);
    let config_str = unwrap_or_exit!(std::str::from_utf8(slice), null);
    Box::into_raw(Box::new(Config::from_json(config_str)))
}

/// Frees an explorer configuration.
#[no_mangle]
pub unsafe extern "C" fn telamon_explorer_config_free(config: *mut Config) {
    std::mem::drop(Box::from_raw(config))
}

/// Copy a C string pointer into a Rust String object. Use this to set
/// string-valued configuration options.
///
/// # Safety
///
/// `dst` must point to a valid Rust String object and `src` must
/// point to a NULL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn telamon_string_copy(
    dst: *mut String,
    src: *const c_char,
) -> TelamonStatus {
    *dst = unwrap_or_exit!(CStr::from_ptr(src).to_str()).to_string();
    TelamonStatus::Ok
}

/// Run the exploration according to the configuration.
///
/// Does not take ownership of any of its arguments. The caller is
/// responsible for freeing them after `telamon_explore_all` returns.
///
/// # Safety
///
///  * `config` and `context` must point to valid objects of their
///    respective types.
///  * `num_search_spaces` must be non-zero.
///  * `search_space` must point to a sequence of at least
///    `num_search_spaces` valid `SearchSpace` objects.
#[no_mangle]
pub unsafe extern "C" fn telamon_explore(
    config: *const Config,
    context: *const Context,
    num_search_spaces: usize,
    search_space: *const SearchSpace,
) -> *mut SearchSpace {
    let search_space = std::slice::from_raw_parts(search_space, num_search_spaces)
        .iter()
        .cloned()
        .map(|search_space| search_space.0)
        .collect();
    let best =
        explorer::find_best(&*config, (*context).as_inner().as_ref(), search_space);

    unwrap_or_exit!(
        best.map(SearchSpace)
            .map(Box::new)
            .map(Box::into_raw)
            .ok_or(()),
        null
    )
}
