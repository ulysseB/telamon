mod cgen;

use telamon::device::cuda::{Context, Executor, Gpu, PerfCounter};
use telamon::ir;
use telamon::search_space::InstFlag;
use gen;
use itertools::Itertools;
use table::Table;

/// Cleans a perfcounter name, for a better display
fn clean_perfcounter_str(s: &str) -> String { //TODO(display bench) clean names
    s.replace("Subp", "").replace("Sector","Sct").replace("read", "Rd")
        .replace("write", "Wr").replace("_", "").replace("misses", "Ms")
        .replace("hit", "Hit").replace("fb", "").replace("global", "Gbl")
}

/// Creates an empty `Table` to hold the given performance counters.
fn create_table(parameters: &[&str], counters: &[PerfCounter]) -> Table<u64> {
    let header = parameters.iter().map(|x| x.to_string())
        .chain(counters.iter().map(|x| clean_perfcounter_str(&x.to_string())))
        .collect_vec();
    Table::new(header)
}

/// Generate raw data for a given benchmark.
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn gen_raw_data(gpu: &Gpu,
                    executor: &Executor,
                    bench_choice: i32,
                    ld_flag: InstFlag,
                    st_flag: InstFlag,
                    outer_loop_sizes: &[i32],
                    inner_loop_sizes: &[i32],
                    strides: &[i32],
                    perf_counters: &[PerfCounter]) -> Table<u64> {
    let scalar_args = [("n_outer", ir::Type::I(32)),
                       ("n", ir::Type::I(32)),
                       ("stride", ir::Type::I(32))];
    let (base, mem_ids) = gen::base(&scalar_args, &["tab"]);
    let max_stride = strides.iter().cloned().max().unwrap_or(0);
    let max_loop_size = inner_loop_sizes.iter().cloned().max().unwrap_or(0);
    let array_size = max_loop_size * (max_stride + 1) + 1;
    let mut context = Context::from_gpu(gpu.clone(), executor);
    gen::bind_array::<f32>("tab", array_size as usize, &mut context);
    let arg_ranges = [
        ("stride", &strides[..]),
        ("n", &inner_loop_sizes[..]),
        ("n_outer", &outer_loop_sizes[..])];

    let n_access_per_iter;
    let n_inst_per_iter;
    let fun = match bench_choice {
        0 => {
            n_access_per_iter = 2;
            n_inst_per_iter = 5;
            cgen::incr_array(&base, gpu, ld_flag, st_flag, "n_outer", "n", "stride",
                             mem_ids[0], "tab")
        },
        1 => {
            n_access_per_iter = 4;
            n_inst_per_iter = 8;
            cgen::add_successive(&base, gpu, ld_flag, st_flag, "n_outer", "n", "stride",
                                 mem_ids[0], "tab")
        },
        2 => {
            //TODO(bench) cleanup bench: acc
            //TODO(bench) insts counts for bench 2
            n_access_per_iter = 1;
            n_inst_per_iter = 4;
            cgen::acc_array(&base, gpu, ld_flag, "n_outer", "n", "stride", mem_ids[0], "tab")
        },
        3 => {
            n_access_per_iter = 1;
            n_inst_per_iter = 3;
            cgen::write_array(&base, gpu, st_flag, "n_outer", "n", "stride", mem_ids[0], "tab")
        },
        _ => panic!("Unknown benchmark"),
    };

    let counters = executor.create_perf_counter_set(perf_counters);
    let table_headers = ["n_access", "n_inst", "stride", "n", "n_outer"];
    let mut table = create_table(&table_headers, perf_counters);
    let prefix = [n_access_per_iter, n_inst_per_iter];
    gen::run(&mut context, &fun, &arg_ranges, &counters, &prefix, &mut table);
    table
}
