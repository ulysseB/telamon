//! GPU (micro)-archtecture characterization.
mod instruction;
mod gen;
mod gpu;
mod math;
mod table;

use self::table::Table;

use device::cuda;
use itertools::Itertools;
use std;
use xdg;

/// Retrieve the description of the GPU from the description file. Updates it if needed.
pub fn get_gpu_desc(executor: &cuda::Executor) -> cuda::Gpu {
    let config_path = get_config_path();
    // FIXME: try to parse the file
    // FIXME: check gpu name
    // FIXME: if GPU not found, characterize it and add it to the file
    unimplemented!() // FIXME return the GPU
}

/// Characterize a GPU.
pub fn characterize(executor: &cuda::Executor) -> cuda::Gpu {
    info!("gpu name: {}", executor.device_name());
    let mut gpu = gpu::functional_desc(executor);
    gpu::performance_desc(executor, &mut gpu);
    gpu
}

/// Creates an empty `Table` to hold the given performance counters.
fn create_table(parameters: &[&str], counters: &[cuda::PerfCounter]) -> Table<u64> {
    let header = parameters.iter().map(|x| x.to_string())
        .chain(counters.iter().map(|x| x.to_string())).collect_vec();
    Table::new(header)
}

/// Returns the name of the configuration file.
pub fn get_config_path() -> std::path::PathBuf {
    let xdg_dirs = unwrap!(xdg::BaseDirectories::with_prefix("telamon"));
    xdg_dirs.find_config_file("cuda_gpus.json").unwrap_or_else(|| {
        let path = xdg_dirs .place_config_file("cuda_gpus.json");
        unwrap!(path, "cannot create configuration directory")
    })
}
