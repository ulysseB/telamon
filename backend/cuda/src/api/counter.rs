//! Allows the execution of kernels on the GPU.
use crate::api::wrapper::*;
use crate::api::Argument;
use fxhash::FxHashMap;
use itertools::Itertools;
use std::ffi::CString;
use std::fmt;
use utils::*;

/// A set of performance counter to monitor.
pub struct PerfCounterSet<'a> {
    num_event: usize,
    event_sets: *mut CuptiEventGroupSets,
    event_pos: FxHashMap<u32, usize>,
    context: &'a CudaContext,
}

impl<'a> PerfCounterSet<'a> {
    /// Creates a new set of performance counters.
    pub fn new(context: &'a CudaContext, counters: &[PerfCounter]) -> Self {
        let mut event_ids: Vec<u32> = Vec::with_capacity(counters.len());
        let event_names = counters
            .iter()
            .map(|x| unwrap!(CString::new(x.to_string())))
            .collect_vec();
        let event_name_ptrs = event_names.iter().map(|x| x.as_ptr()).collect_vec();
        let event_sets = unsafe {
            event_ids.set_len(counters.len());
            let names_ptr = event_name_ptrs.as_ptr();
            let len = counters.len() as u32;
            create_cuptiEventGroupSets(context, len, names_ptr, event_ids.as_mut_ptr())
        };
        let event_pos = event_ids
            .into_iter()
            .enumerate()
            .map(|(x, y)| (y, x))
            .collect();
        PerfCounterSet {
            num_event: counters.len(),
            event_sets,
            event_pos,
            context,
        }
    }

    /// Instrument a `CudaFunction`.
    pub fn instrument(
        &self,
        fun: &CudaFunction,
        blocks: &[u32],
        threads: &[u32],
        args: &[&dyn Argument],
    ) -> Vec<u64> {
        let mut event_ids: Vec<u32> = Vec::with_capacity(self.num_event);
        let mut event_values: Vec<u64> = Vec::with_capacity(self.num_event);
        let mut ordered_values: Vec<u64> = Vec::with_capacity(self.num_event);
        let arg_raw_ptrs = args.iter().map(|x| x.raw_ptr()).collect_vec();
        unsafe {
            event_ids.set_len(self.num_event);
            event_values.set_len(self.num_event);
            ordered_values.set_len(self.num_event);
            instrument_kernel(
                self.context,
                fun,
                blocks.as_ptr(),
                threads.as_ptr(),
                arg_raw_ptrs.as_ptr(),
                self.event_sets,
                event_ids.as_mut_ptr(),
                event_values.as_mut_ptr(),
            );
        }
        let event_pos = event_ids.iter().map(|x| self.event_pos[x]);
        for (pos, value) in event_pos.zip(event_values) {
            ordered_values[pos] = value;
        }
        ordered_values
    }
}

unsafe impl<'a> Sync for PerfCounterSet<'a> {}
unsafe impl<'a> Send for PerfCounterSet<'a> {}

impl<'a> Drop for PerfCounterSet<'a> {
    fn drop(&mut self) {
        unsafe {
            free_cuptiEventGroupSets(self.context, self.event_sets);
        }
    }
}

/// Name a performance counter.
// Some performance counters are not present in every architecture.
#[derive(Clone, Copy, Debug)]
pub enum PerfCounter {
    // common
    /// The number of wrap of instruction executed, does not include replays.
    InstExecuted,
    /// The number of cycles used for the execution on each SMX.
    ElapsedCyclesSM,
    /// Loads and Stores
    LocalLoad,
    LocalStore,
    SharedLoad,
    SharedStore,
    /// Replays
    GlobalLoadReplay,
    GlobalStoreReplay,
    SharedLoadReplay,
    SharedStoreReplay,
    /// Number of uncached global loads and global stores
    UncachedGlobalLoadTransaction,
    GlobalStoreTransaction,
    /// The numbers for l1 cache accesses: hits and misses
    L1LocalLoadHit,
    L1LocalLoadMiss,
    L1LocalStoreHit,
    L1LocalStoreMiss,
    L1GlobalLoadHit,
    L1GlobalLoadMiss,
    L1LocalSharedBankConflict,
    L2Subp0WriteSectorMisses,
    L2Subp1WriteSectorMisses,
    L2Subp0ReadSectorMisses,
    L2Subp1ReadSectorMisses,
    // --
    GldInst8Bit,
    GldInst16Bit,
    GldInst32Bit,
    GldInst64Bit,
    GldInst128Bit,
    GstInst8Bit,
    GstInst16Bit,
    GstInst32Bit,
    GstInst64Bit,
    GstInst128Bit,
    Fb0Subp0ReadSectors,
    Fb0Subp0WriteSectors,
    Fb1Subp0ReadSectors,
    Fb1Subp0WriteSectors,
    Fb0Subp1ReadSectors,
    Fb0Subp1WriteSectors,
    Fb1Subp1ReadSectors,
    Fb1Subp1WriteSectors,
    // Fermi
    // The numbers for l2 cache accesses: hits and misses per sector
    L2Subp0WriteSectorQueries,
    L2Subp1WriteSectorQueries,
    L2Subp0ReadSectorQueries,
    L2Subp1ReadSectorQueries,
    L2Subp0ReadTexSectorQueries,
    L2Subp1ReadTexSectorQueries,
    L2Subp0ReadHitSectors,
    L2Subp1ReadHitSectors,
    L2Subp0ReadTexHitSectors,
    L2Subp1ReadTexHitSectors,
    L2Subp0ReadSysmemSectorQueries,
    L2Subp1ReadSysmemSectorQueries,
    L2Subp0WriteSysmemSectorQueries,
    L2Subp1WriteSysmemSectorQueries,
    L2Subp0TotalReadSectorQueries,
    L2Subp1TotalReadSectorQueries,
    L2Subp0TotalWriteSectorQueries,
    L2Subp1TotalWriteSectorQueries,
    // Kepler
    L2Subp2TotalReadSectorQueries,
    L2Subp2TotalWriteSectorQueries,
    L2Subp3TotalReadSectorQueries,
    L2Subp3TotalWriteSectorQueries,
    FbSubp0ReadSectors,
    FbSubp0WriteSectors,
    FbSubp1ReadSectors,
    FbSubp1WriteSectors,
    // --
    L2Subp2ReadSectorMisses,
    L2Subp3ReadSectorMisses,
    L2Subp2WriteSectorMisses,
    L2Subp3WriteSectorMisses,
    L2Subp0WriteL1SectorQueries,
    L2Subp1WriteL1SectorQueries,
    L2Subp2WriteL1SectorQueries,
    L2Subp3WriteL1SectorQueries,
    L2Subp0ReadL1SectorQueries,
    L2Subp1ReadL1SectorQueries,
    L2Subp2ReadL1SectorQueries,
    L2Subp3ReadL1SectorQueries,
    L2Subp0ReadL1HitSectors,
    L2Subp1ReadL1HitSectors,
    L2Subp2ReadL1HitSectors,
    L2Subp3ReadL1HitSectors,
}

impl fmt::Display for PerfCounter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            // Common counters (Probable errors, unchecked)
            PerfCounter::InstExecuted => "inst_executed",
            PerfCounter::ElapsedCyclesSM => "elapsed_cycles_sm",
            PerfCounter::LocalLoad => "local_load",
            PerfCounter::LocalStore => "local_store",
            PerfCounter::SharedLoad => "shared_load",
            PerfCounter::SharedStore => "shared_store",
            PerfCounter::UncachedGlobalLoadTransaction => {
                "uncached_global_load_transaction"
            }
            PerfCounter::GlobalLoadReplay => "global_ld_mem_divergence_replays",
            PerfCounter::GlobalStoreReplay => "global_st_mem_divergence_replays",
            PerfCounter::SharedLoadReplay => "shared_load_replay",
            PerfCounter::SharedStoreReplay => "shared_store_replay",
            PerfCounter::GlobalStoreTransaction => "global_store_transaction",
            PerfCounter::L1LocalLoadHit => "l1_local_load_hit",
            PerfCounter::L1LocalLoadMiss => "l1_local_load_miss",
            PerfCounter::L1LocalStoreHit => "l1_local_store_hit",
            PerfCounter::L1LocalStoreMiss => "l1_local_store_miss",
            PerfCounter::L1GlobalLoadHit => "l1_global_load_hit",
            PerfCounter::L1GlobalLoadMiss => "l1_global_load_miss",
            PerfCounter::L1LocalSharedBankConflict => "l1_local_shared_bank_conflict",
            // Fermi Counters
            PerfCounter::L2Subp0WriteSectorMisses => "l2_subp0_write_sector_misses",
            PerfCounter::L2Subp1WriteSectorMisses => "l2_subp1_write_sector_misses",
            PerfCounter::L2Subp0ReadSectorMisses => "l2_subp0_read_sector_misses",
            PerfCounter::L2Subp1ReadSectorMisses => "l2_subp1_read_sector_misses",
            PerfCounter::L2Subp0WriteSectorQueries => "l2_subp0_write_sector_queries",
            PerfCounter::L2Subp1WriteSectorQueries => "l2_subp1_write_sector_queries",
            PerfCounter::L2Subp0ReadSectorQueries => "l2_subp0_read_sector_queries",
            PerfCounter::L2Subp1ReadSectorQueries => "l2_subp1_read_sector_queries",
            PerfCounter::L2Subp0ReadTexSectorQueries => {
                "l2_subp0_read_tex_sector_queries"
            }
            PerfCounter::L2Subp1ReadTexSectorQueries => {
                "l2_subp1_read_tex_sector_queries"
            }
            PerfCounter::L2Subp0ReadHitSectors => "l2_subp0_read_hit_sectors",
            PerfCounter::L2Subp1ReadHitSectors => "l2_subp1_read_hit_sectors",
            PerfCounter::L2Subp0ReadTexHitSectors => "l2_subp0_read_tex_hit_sectors",
            PerfCounter::L2Subp1ReadTexHitSectors => "l2_subp1_read_tex_hit_sectors",
            PerfCounter::L2Subp0ReadSysmemSectorQueries => {
                "l2_subp0_read_sysmem_sector_queries"
            }
            PerfCounter::L2Subp1ReadSysmemSectorQueries => {
                "l2_subp1_read_sysmem_sector_queries"
            }
            PerfCounter::L2Subp0WriteSysmemSectorQueries => {
                "l2_subp0_write_sysmem_sector_queries"
            }
            PerfCounter::L2Subp1WriteSysmemSectorQueries => {
                "l2_subp1_write_sysmem_sector_queries"
            }
            PerfCounter::L2Subp0TotalReadSectorQueries => {
                "l2_subp0_total_read_sector_queries"
            }
            PerfCounter::L2Subp1TotalReadSectorQueries => {
                "l2_subp1_total_read_sector_queries"
            }
            PerfCounter::L2Subp0TotalWriteSectorQueries => {
                "l2_subp0_total_write_sector_queries"
            }
            PerfCounter::L2Subp1TotalWriteSectorQueries => {
                "l2_subp1_total_write_sector_queries"
            }
            PerfCounter::Fb0Subp0ReadSectors => "fb0_subp0_read_sectors",
            PerfCounter::Fb0Subp0WriteSectors => "fb0_subp0_write_sectors",
            PerfCounter::Fb1Subp0ReadSectors => "fb1_subp0_read_sectors",
            PerfCounter::Fb1Subp0WriteSectors => "fb1_subp0_write_sectors",
            PerfCounter::Fb0Subp1ReadSectors => "fb0_subp1_read_sectors",
            PerfCounter::Fb0Subp1WriteSectors => "fb0_subp1_write_sectors",
            PerfCounter::Fb1Subp1ReadSectors => "fb1_subp1_read_sectors",
            PerfCounter::Fb1Subp1WriteSectors => "fb1_subp1_write_sectors",
            // Kepler counters
            PerfCounter::L2Subp2TotalReadSectorQueries => {
                "l2_subp2_total_read_sector_queries"
            }
            PerfCounter::L2Subp2TotalWriteSectorQueries => {
                "l2_subp2_total_write_sector_queries"
            }
            PerfCounter::L2Subp3TotalReadSectorQueries => {
                "l2_subp3_total_read_sector_queries"
            }
            PerfCounter::L2Subp3TotalWriteSectorQueries => {
                "l2_subp3_total_write_sector_queries"
            }
            PerfCounter::FbSubp0ReadSectors => "fb_subp0_read_sectors",
            PerfCounter::FbSubp0WriteSectors => "fb_subp0_write_sectors",
            PerfCounter::FbSubp1ReadSectors => "fb_subp1_read_sectors",
            PerfCounter::FbSubp1WriteSectors => "fb_subp1_write_sectors",
            PerfCounter::GldInst8Bit => "gld_inst_8bit",
            PerfCounter::GldInst16Bit => "gld_inst_16bit",
            PerfCounter::GldInst32Bit => "gld_inst_32bit",
            PerfCounter::GldInst64Bit => "gld_inst_64bit",
            PerfCounter::GldInst128Bit => "gld_inst_128bit",
            PerfCounter::GstInst8Bit => "gst_inst_8bit",
            PerfCounter::GstInst16Bit => "gst_inst_16bit",
            PerfCounter::GstInst32Bit => "gst_inst_32bit",
            PerfCounter::GstInst64Bit => "gst_inst_64bit",
            PerfCounter::GstInst128Bit => "gst_inst_128bit",
            PerfCounter::L2Subp2ReadSectorMisses => "l2_subp2_read_sector_misses",
            PerfCounter::L2Subp3ReadSectorMisses => "l2_subp3_read_sector_misses",
            PerfCounter::L2Subp2WriteSectorMisses => "l2_subp2_write_sector_misses",
            PerfCounter::L2Subp3WriteSectorMisses => "l2_subp3_write_sector_misses",
            PerfCounter::L2Subp0WriteL1SectorQueries => {
                "l2_subp0_write_l1_sector_queries"
            }
            PerfCounter::L2Subp1WriteL1SectorQueries => {
                "l2_subp1_write_l1_sector_queries"
            }
            PerfCounter::L2Subp2WriteL1SectorQueries => {
                "l2_subp2_write_l1_sector_queries"
            }
            PerfCounter::L2Subp3WriteL1SectorQueries => {
                "l2_subp3_write_l1_sector_queries"
            }
            PerfCounter::L2Subp0ReadL1SectorQueries => "l2_subp0_read_l1_sector_queries",
            PerfCounter::L2Subp1ReadL1SectorQueries => "l2_subp1_read_l1_sector_queries",
            PerfCounter::L2Subp2ReadL1SectorQueries => "l2_subp2_read_l1_sector_queries",
            PerfCounter::L2Subp3ReadL1SectorQueries => "l2_subp3_read_l1_sector_queries",
            PerfCounter::L2Subp0ReadL1HitSectors => "l2_subp0_read_l1_hit_sectors",
            PerfCounter::L2Subp1ReadL1HitSectors => "l2_subp1_read_l1_hit_sectors",
            PerfCounter::L2Subp2ReadL1HitSectors => "l2_subp2_read_l1_hit_sectors",
            PerfCounter::L2Subp3ReadL1HitSectors => "l2_subp3_read_l1_hit_sectors",
        };
        write!(f, "{}", s)
    }
}
