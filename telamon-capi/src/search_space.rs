//! C API wrappers to work with a Telamon search space.

use std;

use super::error::TelamonStatus;
use super::ir::Function;
use telamon::search_space;

/// Opaque type that abstracts away the lifetime parameter of
/// `search_space::SearchSpace`.
#[derive(Clone)]
pub struct SearchSpace(pub(crate) search_space::SearchSpace<'static>);

/// Creates a new search space from an IR function. The caller stays
/// is responsible for freeing the instance and action pointers; the
/// created search space does not keep references to them.
///
/// Must be freed using `telamon_search_space_free`.
///
/// # Safety
///
///  * `ir_instance` must point to a valid `Function` value.
///  * `actions` must point to a sequence of at least `num_actions`
///    valid `Action` values, unless `num_actions` is 0 in which case
///    `actions` is not used.
#[no_mangle]
pub unsafe extern "C" fn telamon_search_space_new(
    ir_instance: *const Function,
    num_actions: usize,
    actions: *const search_space::Action,
) -> *mut SearchSpace {
    // std::slice::from_raw_parts require that the `actions`
    // pointer be non-null, so we have a special case in order to
    // allow C API users to pass in a null pointer in that case.
    let actions = if num_actions == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(actions, num_actions).to_vec()
    };
    let search_space = unwrap_or_exit!(
        search_space::SearchSpace::new((*ir_instance).clone().into(), actions),
        null
    );
    Box::into_raw(Box::new(SearchSpace(search_space)))
}

/// Apply a sequence of actions to a search space.
///
/// # Safety
///
///  * `search_space` must be a valid pointer containing a valid
///    `SearchSpace` value.
///  * `num_actions` must be non-zero.
///  * `actions` must point to a sequence of at least `num_actions`
///     valid `Action` values.
#[no_mangle]
pub unsafe extern "C" fn telamon_search_space_apply(
    search_space: *mut SearchSpace,
    num_actions: usize,
    actions: *const search_space::Action,
) -> TelamonStatus {
    unwrap_or_exit!(
        (*search_space)
            .0
            .apply_decisions(std::slice::from_raw_parts(actions, num_actions).to_vec())
    );
    TelamonStatus::TelamonStatusOk
}

/// Frees a search space instance allocated through
/// `telamon_search_space_new`.
///
/// # Safety
///
/// `search_space` must point to a `SearchSpace` object created by
/// `telamon_search_space_new` which has not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn telamon_search_space_free(
    search_space: *mut SearchSpace,
) -> TelamonStatus {
    std::mem::drop(Box::from_raw(search_space));
    TelamonStatus::TelamonStatusOk
}
