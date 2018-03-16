#![allow(dead_code)]
#![feature(conservative_impl_trait, box_syntax)]
extern crate env_logger;
extern crate telamon;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate prettytable;
extern crate telamon_utils as utils;

mod gen;
mod cache;
mod table;

// command line options
extern crate getopts;
use std::env;
use getopts::Options;

use telamon::device::cuda::{Executor, Gpu, PerfCounter};
use telamon::ir;
use telamon::search_space::InstFlag;
use itertools::Itertools;
use table::Table;

/// Gather performance counter values and analyse them.
fn bench(gpu: &Gpu,
         executor: &Executor,
         bench_choice: i32,
         ld_flag: InstFlag,
         st_flag: InstFlag,
         outer_loop_sizes: &[i32],
         inner_loop_sizes: &[i32],
         strides: &[i32]) -> Table<f64> {

    // Define which performance counters are measured
    let mut perf_counters =
        vec![
        // General
        PerfCounter::InstExecuted,
        PerfCounter::ElapsedCyclesSM,
        PerfCounter::GlobalStoreTransaction,
        PerfCounter::UncachedGlobalLoadTransaction,
        // L1
        PerfCounter::L1GlobalLoadHit,
        PerfCounter::L1GlobalLoadMiss,
        ];
    // According to the arch, the perfcounters are different
    perf_counters.append(&mut match gpu.sm_major {// index at 8
        2 => vec![
            // L2
            PerfCounter::L2Subp0ReadSectorMisses,
            PerfCounter::L2Subp1ReadSectorMisses,
            PerfCounter::L2Subp0WriteSectorMisses,
            PerfCounter::L2Subp1WriteSectorMisses,

            PerfCounter::L2Subp0TotalReadSectorQueries,
            PerfCounter::L2Subp1TotalReadSectorQueries,
            PerfCounter::L2Subp0TotalWriteSectorQueries,
            PerfCounter::L2Subp1TotalWriteSectorQueries,

            PerfCounter::L2Subp0ReadSectorQueries,
            PerfCounter::L2Subp1ReadSectorQueries,
            PerfCounter::L2Subp0WriteSectorQueries,
            PerfCounter::L2Subp1WriteSectorQueries,

            PerfCounter::L2Subp0ReadHitSectors,
            PerfCounter::L2Subp1ReadHitSectors,
            // Mem
            PerfCounter::Fb0Subp0ReadSectors,
            PerfCounter::Fb0Subp1ReadSectors,
            PerfCounter::Fb1Subp0ReadSectors,
            PerfCounter::Fb1Subp1ReadSectors,
            PerfCounter::Fb0Subp0WriteSectors,
            PerfCounter::Fb0Subp1WriteSectors,
            PerfCounter::Fb1Subp0WriteSectors,
            PerfCounter::Fb1Subp1WriteSectors,
        ],
        3 => vec![
            //L2
            PerfCounter::L2Subp0ReadSectorMisses,
            PerfCounter::L2Subp1ReadSectorMisses,
            PerfCounter::L2Subp2ReadSectorMisses,
            PerfCounter::L2Subp3ReadSectorMisses,
            PerfCounter::L2Subp0WriteSectorMisses,
            PerfCounter::L2Subp1WriteSectorMisses,
            PerfCounter::L2Subp2WriteSectorMisses,
            PerfCounter::L2Subp3WriteSectorMisses,

            PerfCounter::L2Subp0TotalReadSectorQueries,
            PerfCounter::L2Subp2TotalReadSectorQueries,
            PerfCounter::L2Subp1TotalReadSectorQueries,
            PerfCounter::L2Subp3TotalReadSectorQueries,
            PerfCounter::L2Subp0TotalWriteSectorQueries,
            PerfCounter::L2Subp1TotalWriteSectorQueries,
            PerfCounter::L2Subp2TotalWriteSectorQueries,
            PerfCounter::L2Subp3TotalWriteSectorQueries,

            PerfCounter::L2Subp0ReadL1SectorQueries,
            PerfCounter::L2Subp1ReadL1SectorQueries,
            PerfCounter::L2Subp2ReadL1SectorQueries,
            PerfCounter::L2Subp3ReadL1SectorQueries,
            PerfCounter::L2Subp0WriteL1SectorQueries,
            PerfCounter::L2Subp1WriteL1SectorQueries,
            PerfCounter::L2Subp2WriteL1SectorQueries,
            PerfCounter::L2Subp3WriteL1SectorQueries,

            PerfCounter::L2Subp0ReadL1HitSectors,
            PerfCounter::L2Subp1ReadL1HitSectors,
            PerfCounter::L2Subp2ReadL1HitSectors,
            PerfCounter::L2Subp3ReadL1HitSectors,
            // Mem
            PerfCounter::FbSubp0ReadSectors,
            PerfCounter::FbSubp1ReadSectors,
            PerfCounter::FbSubp0WriteSectors,
            PerfCounter::FbSubp1WriteSectors,
        ],
        _ => vec![]
    });
    // Counting the memory loads and stores
    let memory_instructions_counters = false;
    if memory_instructions_counters {
        perf_counters.append(&mut vec![
                             PerfCounter::GldInst8Bit,
                             PerfCounter::GldInst16Bit,
                             PerfCounter::GldInst32Bit,
                             PerfCounter::GldInst64Bit,
                             PerfCounter::GldInst128Bit,
                             PerfCounter::GstInst8Bit,
                             PerfCounter::GstInst16Bit,
                             PerfCounter::GstInst32Bit,
                             PerfCounter::GstInst64Bit,
                             PerfCounter::GstInst128Bit,
        ]);
    }

    // Storing raw perfcounters measures in a table
    let raw_table = cache::gen_raw_data(
        gpu, executor, bench_choice, ld_flag, st_flag, outer_loop_sizes,
        inner_loop_sizes, strides, &perf_counters);

    analyse_counters(gpu, raw_table)
}


/// Analyse raw data from benchmark into a table
fn analyse_counters(gpu: &Gpu, raw_table: Table<u64>) -> Table<f64>
{
    // Round a value v to precision 10^-n
    fn round_n(v: f64, n: i32) -> f64 {
        let prec = (10f64).powi(n);
        let w = prec*v;
        return w.round()/prec;
    };
    // Computes the division a/b of two unsigned integers without errors
    fn div(a: f64, b: f64) -> f64 {if b == 0f64 {-1f64} else {a/b}};

    // Test if two (integer perfcounter) values are close
    fn is_close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 5f64 || (a-b).abs()/(a+b) <= 0.05
    }

    // Checks if two values are close and print if they are not
    fn check_close(a: f64, b: f64, s: &str) {
        if !is_close(a, b) {
            print!("{}: Different values {} and {}\n", s, a, b)
        }
    }

    // Computes sum of values on some range of perfcounters
    fn get_sum(entry: &Vec<u64>, index: &mut u32, range: u32, debug_descr: &str) -> f64{
        let ind = *index as usize;
        let mut sum = 0f64;
        if range == 1 {
            sum = entry[ind] as f64;
        } else {
            // Compute the sum of the chosen related values
            for i in ind..(ind + range as usize) {
                sum += entry[i] as f64;
            }
            // Check if the values are uniform over the range
            let mut uniform = true;
            for i in ind..(ind + range as usize) {
                if !is_close(sum, (range as f64) * (entry[i as usize] as f64)) {
                    uniform = false;
                }
            }
            if !uniform {
                print!("Non uniformity of perfcounters {}: index {}, average {}, values: ",
                       debug_descr, ind, sum/(range as f64));
                for i in ind..(ind + range as usize) {
                    print!("{}, ", entry[i as usize]);
                }
                print!("\n");
            }
        }
        // Update *index
        *index += range;
        // Return the sum of the considered perf counters values
        sum as f64
    };

    // Computes average of values on some range of perfcounters
    fn get_average(entry: &Vec<u64>, index: &mut u32, range: u32, debug_descr: &str) -> f64 {
        get_sum(entry, index, range, debug_descr)/(range as f64)
    };

    // Define offsets for perfcounters indexes
    let subp_nb = gpu.num_smx;
    let subp_nb_mem = 2;
    let fb_nb = if gpu.sm_major == 2 { 2 } else { 1 };

    // Create a table for results of processing
    let mut processed_table = Table::new(["Size/L1", "Size/L2", "Stride","Insts", "Time",
      "% L1H ld ", "% L1H st", "% L2H ld ", "% L2H st"].iter()
      .map(|x| x.to_string()).collect_vec());

    let mut read_table = Table::new(["n_access", "l2_read_misses", "l2_total_read_q",
      "l2_read_l1_q", "l2_read_l1_hit", "queries - hit", "read in mem"].iter()
      .map(|x| x.to_string()).collect_vec());
    let mut write_table = Table::new(["n_access", "l2_write_misses", "l2_total_write_q",
      "l2_write_l1_q", "mem_write"].iter()
      .map(|x| x.to_string()).collect_vec());

    // Fill the new table with data extracted from the raw_table
    for entry in raw_table {
        // Extract parameters
        let n_access_per_iter = entry[0] as f64;
        let n_inst_per_iter = entry[1] as f64;
        let stride = entry[2] as f64;
        let loop_size = entry[3] as f64;
        let reuse_loop = entry[4] as f64;
        // Extract performance counters
        //let inst_executed = entry[5] as f64;
        //let cycles = (entry[6] as f64)/(gpu.num_smx as f64) as f64;
        let gbl_st_transaction = entry[7] as f64;
        let gbl_uncached_ld_transaction = entry[8] as f64;
        let l1_unit_to_sector_unit = 2f64; // should be 8, is 2
        let l1_gbl_load_hit = entry[9] as f64 * l1_unit_to_sector_unit; // 32 bits > 32 bytes accesses -> *8
        let l1_gbl_load_miss = entry[10] as f64 * l1_unit_to_sector_unit;

        // Define perfcounter index, correctly incremented by calls to get_sum
        let mut cur_index = 11;
        // Sum the percounters of differnet smx and fb
        let l2_read_misses = get_sum(&entry, &mut cur_index, subp_nb, "l2_read_misses");
        let l2_write_misses = get_sum(&entry, &mut cur_index, subp_nb, "l2_write_misses");
        let l2_total_read_queries = get_sum(&entry, &mut cur_index, subp_nb, "l2_total_read_q");
        let l2_total_write_queries = get_sum(&entry, &mut cur_index, subp_nb, "l2_total_write_q");
        let l2_read_l1_queries = get_sum(&entry, &mut cur_index, subp_nb, "l2_read_l1_q");
        let l2_write_l1_queries = get_sum(&entry, &mut cur_index, subp_nb, "l2_write_l1_q");
        let l2_read_l1_hit = get_sum(&entry, &mut cur_index, subp_nb, "l2_read_l1_hit");
        let mem_read = get_sum(&entry, &mut cur_index, fb_nb * subp_nb_mem, "mem_read");
        let mem_write = get_sum(&entry, &mut cur_index, fb_nb * subp_nb_mem, "mem_write");

        //TODO(checks) check equality of parameters
        //TODO(checks) check equality of in/out for L1, L2 and Mem
        check_close(l1_gbl_load_miss, l2_read_l1_queries, "L1>L2 load");
        // laptop: true with factor  2
        check_close(l1_gbl_load_miss + l1_gbl_load_hit, gbl_uncached_ld_transaction,
                    "L1 load, gbl uncached load");
        // laptop: false : second is 0 when L1 is used. or the other way around for CG
        check_close(l2_read_misses, mem_read, "L2>Mem load Load");
        //laptop: false
        check_close(l2_write_misses, mem_write, "L2>Mem write Store");
        //laptop: false
        check_close(l2_read_l1_hit + l2_read_misses, l2_read_l1_queries,
                    "L2 load hit + misses, load queries");
        // laptop: false
        check_close(l2_read_l1_queries, l2_total_read_queries, "L2 read: l1, total");
        // laptop: true
        check_close(l2_write_l1_queries, l2_total_write_queries, "L2 write: l1, total");
        //laptop: true


        // Compute the ratio between the data accessed and the size of caches
        let data_size = loop_size * (ir::Type::F(32).len_byte().unwrap() as f64) as f64;
        let loop_size_over_l1 = round_n(div(data_size as f64, gpu.l1_cache_size as f64), 5);
        let loop_size_over_l2 = round_n(div(data_size as f64, gpu.l2_cache_size as f64), 5);

        // Compute the proportion of hits and misses in L1 and L2
        let l1_ld_hit_percent = round_n(div(l1_gbl_load_hit,
                                            l1_gbl_load_hit + l1_gbl_load_miss), 5);
        let l1_st_hit_percent = round_n(div(gbl_st_transaction - l2_write_l1_queries,
                                            gbl_st_transaction), 5);
        let l2_ld_hit_percent = round_n(div(l2_read_l1_hit, l2_read_l1_queries), 5);
        let l2_st_hit_percent = round_n(div(l2_write_l1_queries - mem_write,
                                     l2_write_l1_queries), 5);
        // Store the previous values into the table
        let do_table_results = true;   // FIXME: not working
        if do_table_results {
            let entry2 = vec![loop_size_over_l1, loop_size_over_l2, stride as f64,
                              entry[5] as f64, (entry[6]) as f64, l1_ld_hit_percent,
                              l1_st_hit_percent, l2_ld_hit_percent, l2_st_hit_percent];
            processed_table.add_entry(entry2);
        }

        // Write bench results to stdout
        print!("LpSz:{:>6}, {:.1},{:.1}; R:{:>2}  [", loop_size, loop_size_over_l1,
               loop_size_over_l2, reuse_loop);

        // printing prediction of measures for checks
        let n_access = n_access_per_iter * loop_size * reuse_loop as f64;
        let n_inst = ((n_inst_per_iter  + 3f64) * loop_size + 3f64) * reuse_loop;
        print!("{:>7}; ", n_access);
        print!("{:>7}|| ", n_inst);
        // Raw display of first performance counters
        for x in &entry[3..11] {
            print!("{:>7}, ",*x/ (reuse_loop as u64));
        }
        print!("\n");
        for x in vec![l2_read_misses, l2_write_misses, l2_total_read_queries, l2_total_write_queries, l2_read_l1_queries, l2_write_l1_queries, l2_read_l1_hit, mem_read, mem_write].iter() {
            print!("{:>7}, ",x/ (reuse_loop as f64));
        }
        println!("]");

        read_table.add_entry(vec![n_access, l2_read_misses, l2_total_read_queries, l2_read_l1_queries, l2_read_l1_hit, l2_read_l1_queries- l2_read_l1_hit, mem_read]);
        write_table.add_entry(vec![n_access, l2_write_misses, l2_total_write_queries, l2_write_l1_queries, mem_write]);

    }
    //processed_table.pretty().printstd();
    read_table.pretty().printstd();
    write_table.pretty().printstd();
    processed_table
}

// Main function, gathering command line options
fn main() {
    let _ = env_logger::init();
    let executor = Executor::init();
    let gpu_name = executor.device_name();
    let gpu = Gpu::from_name(&gpu_name).unwrap();
    // Setup command line arguments.
    let mut opts = Options::new();
    opts.optopt("b", "", "set benchmark number", "NUMBER");
    opts.optflag("h", "help", "print this help menu");
    opts.optflag("r", "test-reuse", "test w/ reuse loop");
    opts.optflag("s", "test-strides", "test different access strides");
    opts.optopt("f", "flag", "flag to use for memory accesses (cs, ca, cg)", "FLAG");
    //opts.optflag("n", "normalize", "normalize the output");
    //opts.optflag("t", "test-threads", "test w/ threads");
    //opts.optflag("l", "test-blocks", "test w/ blocks");
    // Parse arguments.
    let args: Vec<String> = env::args().collect();
    let matches = opts.parse(&args[1..]).unwrap();
    if matches.opt_present("h") {
        let brief = opts.short_usage(&args[0]);
        println!("{}", opts.usage(&brief));
        return;
    }
    // Set params from command line arguments
    let bench_nb = matches.opt_str("b").map(|s| {
        s.trim().parse::<i32>().expect("Error with benchmark number")
    }).unwrap_or(0);
    let l1_size = gpu.l1_cache_size as i32/ir::Type::F(32).len_byte().unwrap() as i32;
    let l2_size = gpu.l2_cache_size as i32/ir::Type::F(32).len_byte().unwrap() as i32;
    let outer_sizes = if matches.opt_present("r") { vec![1, 4, 16] } else { vec![1] };
    let inner_sizes = [l1_size/2, 2* l1_size, l2_size/2, l2_size, 2*l2_size];
    //let inner_sizes = [l1_size/32, l1_size/16, l1_size/8, l1_size/2, l1_size, 4*l1_size,
        //1*l2_size/4, l2_size/2, 3*l2_size/4, 7*l2_size/8, 15*l2_size/16, l2_size,
        //9/17*l2_size/16, 5*l2_size/4, 3*l2_size/2, 2*l2_size, 4*l2_size, 8*l2_size,
        //16*l2_size];
    let strides = if matches.opt_present("s") { vec![4, 8, 16, 32, 64] } else { vec![4] };
    let (ld_flag, st_flag) = matches.opt_str("f").map(|s| match &s as &str {
        "ca" => (InstFlag::MEM_CA, InstFlag::MEM_CA),
        "cg" => (InstFlag::MEM_CG, InstFlag::MEM_CG),
        "cs" => (InstFlag::MEM_CS, InstFlag::MEM_CS),
        _ => panic!("Unrecognized flag: {}. Valid flags are ca, cg and cs.", s)
    }).unwrap_or((InstFlag::MEM_CG, InstFlag::MEM_CG));

    // Run the benchmark, and print the results.
    bench(&gpu, &executor, bench_nb, ld_flag, st_flag, &outer_sizes, &inner_sizes,
          &strides);
}

// FIXME: create thread and block loops if needed:
//  if params.test_blocks {
//      let size_block = builder.cst_size(4 as u32);;
//      builder.open_dim_ex(size_block, ir::dim::kind::BLOCK, true);
//  }
//  if params.test_threads {
//      let size_thread = builder.cst_size(32 as u32);
//      builder.open_dim_ex(size_thread, ir::dim::kind::THREAD, true);
//  }

// FIXME: normalized printing
//  for x in res.iter().take(2) {
//      print!("{:>7}, ",*x /(reuse_loop as u64));
//  }
//  // Set normalizing value for comparison with predicted number of accesses
//  if params.normalize {
//      normalize_value = nb_accesses as f64;
//  }
//  // Normalize a value for displaying
//  let normalize = |value: u64| -> f64 {
//      let mut displayed_value = (value as f64)/(reuse_loop as f64);
//      if params.normalize {
//          displayed_value /= (normalize_value as f64);
//      }
//      displayed_value
//  };
//  // Display the rest
//  for x in res.iter().skip(2) {
//      // Raw or normalized
//      if !params.normalize {
//          print!("{:>7}, ",*x);
//      } else {
//          print!("{:>7.*}, ", 1, 100. * normalize(*x));
//      }
//  }
