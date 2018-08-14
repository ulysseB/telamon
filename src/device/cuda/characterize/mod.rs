//! GPU (micro)-archtecture characterization.
mod instruction;
mod gen;
mod gpu;
mod math;
mod table;

use self::table::Table;

use device::cuda;
use itertools::Itertools;
use serde_json;
use std;
use xdg;

/// Error raised will retrieveing the GPU description
#[derive(Debug, Fail)]
enum Error {
    #[fail(display="could not parse GPU description: {}", _0)]
    Parser(serde_json::Error),
    #[fail(display="found description for the wrong GPU: {}", _0)]
    WrongGpu(String),
}

/// Retrieve the description of the GPU from the description file. Updates it if needed.
pub fn get_gpu_desc(executor: &cuda::Executor) -> cuda::Gpu {
    let config_path = get_config_path();
    lazy_static! {
        // Ensure that at most one thread runs the characterization.
        static ref LOCK: std::sync::Mutex<()> = Default::default();
    }
    let lock = unwrap!(LOCK.lock());
    let gpu = serde_json::from_reader(&unwrap!(std::fs::File::open(&config_path)))
        .map_err(Error::Parser)
        .and_then(|gpu: cuda::Gpu| {
            let name = executor.device_name();
            if gpu.name == name { Ok(gpu) } else { Err(Error::WrongGpu(name)) }
        }).unwrap_or_else(|err| {
            println!("Could not read the GPU characterization file.");
            println!("Running GPU characterization, this can take several minutes.");
            warn!("{}. Running characterization.", err);
            let gpu = characterize(executor);
            let out = unwrap!(std::fs::File::create(&config_path));
            unwrap!(serde_json::to_writer_pretty(out, &gpu));
            println!("Characterization finished and written to {}",
                     unwrap!(config_path.to_str()));
            gpu
        });
    std::mem::drop(lock);
    gpu
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
    // We use `place_config_file` instead of `find_config_file` to avoid returning
    // a system-wide files (e.g. in /usr/share/telamon/cuda_gpus.json) on which the user
    // doesn't have write permissions.
    let path = xdg_dirs.place_config_file("cuda_gpus.json");
    unwrap!(path, "cannot create configuration directory")
}
