#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

pub mod library_types {
    include!(concat!(env!("OUT_DIR"), "/library_types.rs"));
}

pub mod vector_types {
    include!(concat!(env!("OUT_DIR"), "/vector_types.rs"));
}

pub mod driver_types {
    pub use crate::vector_types::*;

    include!(concat!(env!("OUT_DIR"), "/driver_types.rs"));
}

pub mod cuComplex {
    pub use crate::vector_types::*;

    include!(concat!(env!("OUT_DIR"), "/cuComplex.rs"));
}

#[cfg(any(feature = "cuda", feature = "cupti"))]
pub mod cuda_types {
    pub use crate::driver_types::CUuuid;

    include!(concat!(env!("OUT_DIR"), "/cuda_types.rs"));
}

#[cfg(feature = "curand")]
pub mod curand {
    pub use crate::driver_types::*;
    pub use crate::library_types::*;

    include!(concat!(env!("OUT_DIR"), "/curand.rs"));
}

#[cfg(feature = "cudnn")]
pub mod cudnn {
    pub use crate::driver_types::*;
    pub use crate::library_types::*;

    include!(concat!(env!("OUT_DIR"), "/cudnn.rs"));
}

#[cfg(feature = "cuda")]
pub mod cuda {
    pub use crate::cuda_types::*;

    include!(concat!(env!("OUT_DIR"), "/cuda.rs"));
}

#[cfg(feature = "cupti")]
pub mod cupti {
    pub use crate::cuda_types::*;

    include!(concat!(env!("OUT_DIR"), "/cupti.rs"));
}

#[cfg(any(feature = "cublas", feature = "cublas-lt"))]
pub mod cublas_types {
    pub use crate::cuComplex::*;
    pub use crate::driver_types::*;
    pub use crate::library_types::*;

    include!(concat!(env!("OUT_DIR"), "/cublas_types.rs"));
}

#[cfg(feature = "cublas")]
pub mod cublas {
    pub use crate::cublas_types::*;

    include!(concat!(env!("OUT_DIR"), "/cublas.rs"));
}

#[cfg(feature = "cublas")]
pub mod cublas_v2 {
    pub use crate::cublas_types::*;

    include!(concat!(env!("OUT_DIR"), "/cublas_v2.rs"));

    pub use self::cublasCreate_v2 as cublasCreate;
    pub use self::cublasDestroy_v2 as cublasDestroy;
    pub use self::cublasGetPointerMode_v2 as cublasGetPointerMode;
    pub use self::cublasGetStream_v2 as cublasGetStream;
    pub use self::cublasGetVersion_v2 as cublasGetVersion;
    pub use self::cublasSetPointerMode_v2 as cublasSetPointerMode;
    pub use self::cublasSetStream_v2 as cublasSetStream;

    /* Blas3 Routines   */

    pub use self::cublasDnrm2_v2 as cublasDnrm2;
    pub use self::cublasDznrm2_v2 as cublasDznrm2;
    pub use self::cublasScnrm2_v2 as cublasScnrm2;
    pub use self::cublasSnrm2_v2 as cublasSnrm2;

    pub use self::cublasCdotc_v2 as cublasCdotc;
    pub use self::cublasCdotu_v2 as cublasCdotu;
    pub use self::cublasDdot_v2 as cublasDdot;
    pub use self::cublasSdot_v2 as cublasSdot;
    pub use self::cublasZdotc_v2 as cublasZdotc;
    pub use self::cublasZdotu_v2 as cublasZdotu;

    pub use self::cublasCscal_v2 as cublasCscal;
    pub use self::cublasCsscal_v2 as cublasCsscal;
    pub use self::cublasDscal_v2 as cublasDscal;
    pub use self::cublasSscal_v2 as cublasSscal;
    pub use self::cublasZdscal_v2 as cublasZdscal;
    pub use self::cublasZscal_v2 as cublasZscal;

    pub use self::cublasCaxpy_v2 as cublasCaxpy;
    pub use self::cublasDaxpy_v2 as cublasDaxpy;
    pub use self::cublasSaxpy_v2 as cublasSaxpy;
    pub use self::cublasZaxpy_v2 as cublasZaxpy;

    pub use self::cublasCcopy_v2 as cublasCcopy;
    pub use self::cublasDcopy_v2 as cublasDcopy;
    pub use self::cublasScopy_v2 as cublasScopy;
    pub use self::cublasZcopy_v2 as cublasZcopy;

    pub use self::cublasCswap_v2 as cublasCswap;
    pub use self::cublasDswap_v2 as cublasDswap;
    pub use self::cublasSswap_v2 as cublasSswap;
    pub use self::cublasZswap_v2 as cublasZswap;

    pub use self::cublasIcamax_v2 as cublasIcamax;
    pub use self::cublasIdamax_v2 as cublasIdamax;
    pub use self::cublasIsamax_v2 as cublasIsamax;
    pub use self::cublasIzamax_v2 as cublasIzamax;

    pub use self::cublasIcamin_v2 as cublasIcamin;
    pub use self::cublasIdamin_v2 as cublasIdamin;
    pub use self::cublasIsamin_v2 as cublasIsamin;
    pub use self::cublasIzamin_v2 as cublasIzamin;

    pub use self::cublasDasum_v2 as cublasDasum;
    pub use self::cublasDzasum_v2 as cublasDzasum;
    pub use self::cublasSasum_v2 as cublasSasum;
    pub use self::cublasScasum_v2 as cublasScasum;

    pub use self::cublasCrot_v2 as cublasCrot;
    pub use self::cublasCsrot_v2 as cublasCsrot;
    pub use self::cublasDrot_v2 as cublasDrot;
    pub use self::cublasSrot_v2 as cublasSrot;
    pub use self::cublasZdrot_v2 as cublasZdrot;
    pub use self::cublasZrot_v2 as cublasZrot;

    pub use self::cublasCrotg_v2 as cublasCrotg;
    pub use self::cublasDrotg_v2 as cublasDrotg;
    pub use self::cublasSrotg_v2 as cublasSrotg;
    pub use self::cublasZrotg_v2 as cublasZrotg;

    pub use self::cublasDrotm_v2 as cublasDrotm;
    pub use self::cublasSrotm_v2 as cublasSrotm;

    pub use self::cublasDrotmg_v2 as cublasDrotmg;
    pub use self::cublasSrotmg_v2 as cublasSrotmg;

    /* Blas2 Routines */

    pub use self::cublasCgemv_v2 as cublasCgemv;
    pub use self::cublasDgemv_v2 as cublasDgemv;
    pub use self::cublasSgemv_v2 as cublasSgemv;
    pub use self::cublasZgemv_v2 as cublasZgemv;

    pub use self::cublasCgbmv_v2 as cublasCgbmv;
    pub use self::cublasDgbmv_v2 as cublasDgbmv;
    pub use self::cublasSgbmv_v2 as cublasSgbmv;
    pub use self::cublasZgbmv_v2 as cublasZgbmv;

    pub use self::cublasCtrmv_v2 as cublasCtrmv;
    pub use self::cublasDtrmv_v2 as cublasDtrmv;
    pub use self::cublasStrmv_v2 as cublasStrmv;
    pub use self::cublasZtrmv_v2 as cublasZtrmv;

    pub use self::cublasCtbmv_v2 as cublasCtbmv;
    pub use self::cublasDtbmv_v2 as cublasDtbmv;
    pub use self::cublasStbmv_v2 as cublasStbmv;
    pub use self::cublasZtbmv_v2 as cublasZtbmv;

    pub use self::cublasCtpmv_v2 as cublasCtpmv;
    pub use self::cublasDtpmv_v2 as cublasDtpmv;
    pub use self::cublasStpmv_v2 as cublasStpmv;
    pub use self::cublasZtpmv_v2 as cublasZtpmv;

    pub use self::cublasCtrsv_v2 as cublasCtrsv;
    pub use self::cublasDtrsv_v2 as cublasDtrsv;
    pub use self::cublasStrsv_v2 as cublasStrsv;
    pub use self::cublasZtrsv_v2 as cublasZtrsv;

    pub use self::cublasCtpsv_v2 as cublasCtpsv;
    pub use self::cublasDtpsv_v2 as cublasDtpsv;
    pub use self::cublasStpsv_v2 as cublasStpsv;
    pub use self::cublasZtpsv_v2 as cublasZtpsv;

    pub use self::cublasCtbsv_v2 as cublasCtbsv;
    pub use self::cublasDtbsv_v2 as cublasDtbsv;
    pub use self::cublasStbsv_v2 as cublasStbsv;
    pub use self::cublasZtbsv_v2 as cublasZtbsv;

    pub use self::cublasChemv_v2 as cublasChemv;
    pub use self::cublasCsymv_v2 as cublasCsymv;
    pub use self::cublasDsymv_v2 as cublasDsymv;
    pub use self::cublasSsymv_v2 as cublasSsymv;
    pub use self::cublasZhemv_v2 as cublasZhemv;
    pub use self::cublasZsymv_v2 as cublasZsymv;

    pub use self::cublasChbmv_v2 as cublasChbmv;
    pub use self::cublasDsbmv_v2 as cublasDsbmv;
    pub use self::cublasSsbmv_v2 as cublasSsbmv;
    pub use self::cublasZhbmv_v2 as cublasZhbmv;

    pub use self::cublasChpmv_v2 as cublasChpmv;
    pub use self::cublasDspmv_v2 as cublasDspmv;
    pub use self::cublasSspmv_v2 as cublasSspmv;
    pub use self::cublasZhpmv_v2 as cublasZhpmv;

    pub use self::cublasCgerc_v2 as cublasCgerc;
    pub use self::cublasCgeru_v2 as cublasCgeru;
    pub use self::cublasDger_v2 as cublasDger;
    pub use self::cublasSger_v2 as cublasSger;
    pub use self::cublasZgerc_v2 as cublasZgerc;
    pub use self::cublasZgeru_v2 as cublasZgeru;

    pub use self::cublasCher_v2 as cublasCher;
    pub use self::cublasCsyr_v2 as cublasCsyr;
    pub use self::cublasDsyr_v2 as cublasDsyr;
    pub use self::cublasSsyr_v2 as cublasSsyr;
    pub use self::cublasZher_v2 as cublasZher;
    pub use self::cublasZsyr_v2 as cublasZsyr;

    pub use self::cublasChpr_v2 as cublasChpr;
    pub use self::cublasDspr_v2 as cublasDspr;
    pub use self::cublasSspr_v2 as cublasSspr;
    pub use self::cublasZhpr_v2 as cublasZhpr;

    pub use self::cublasCher2_v2 as cublasCher2;
    pub use self::cublasCsyr2_v2 as cublasCsyr2;
    pub use self::cublasDsyr2_v2 as cublasDsyr2;
    pub use self::cublasSsyr2_v2 as cublasSsyr2;
    pub use self::cublasZher2_v2 as cublasZher2;
    pub use self::cublasZsyr2_v2 as cublasZsyr2;

    pub use self::cublasChpr2_v2 as cublasChpr2;
    pub use self::cublasDspr2_v2 as cublasDspr2;
    pub use self::cublasSspr2_v2 as cublasSspr2;
    pub use self::cublasZhpr2_v2 as cublasZhpr2;

    /* Blas3 Routines   */

    pub use self::cublasCgemm_v2 as cublasCgemm;
    pub use self::cublasDgemm_v2 as cublasDgemm;
    pub use self::cublasSgemm_v2 as cublasSgemm;
    pub use self::cublasZgemm_v2 as cublasZgemm;

    pub use self::cublasCherk_v2 as cublasCherk;
    pub use self::cublasCsyrk_v2 as cublasCsyrk;
    pub use self::cublasDsyrk_v2 as cublasDsyrk;
    pub use self::cublasSsyrk_v2 as cublasSsyrk;
    pub use self::cublasZherk_v2 as cublasZherk;
    pub use self::cublasZsyrk_v2 as cublasZsyrk;

    pub use self::cublasCher2k_v2 as cublasCher2k;
    pub use self::cublasCsyr2k_v2 as cublasCsyr2k;
    pub use self::cublasDsyr2k_v2 as cublasDsyr2k;
    pub use self::cublasSsyr2k_v2 as cublasSsyr2k;
    pub use self::cublasZher2k_v2 as cublasZher2k;
    pub use self::cublasZsyr2k_v2 as cublasZsyr2k;

    pub use self::cublasChemm_v2 as cublasChemm;
    pub use self::cublasCsymm_v2 as cublasCsymm;
    pub use self::cublasDsymm_v2 as cublasDsymm;
    pub use self::cublasSsymm_v2 as cublasSsymm;
    pub use self::cublasZhemm_v2 as cublasZhemm;
    pub use self::cublasZsymm_v2 as cublasZsymm;

    pub use self::cublasCtrsm_v2 as cublasCtrsm;
    pub use self::cublasDtrsm_v2 as cublasDtrsm;
    pub use self::cublasStrsm_v2 as cublasStrsm;
    pub use self::cublasZtrsm_v2 as cublasZtrsm;

    pub use self::cublasCtrmm_v2 as cublasCtrmm;
    pub use self::cublasDtrmm_v2 as cublasDtrmm;
    pub use self::cublasStrmm_v2 as cublasStrmm;
    pub use self::cublasZtrmm_v2 as cublasZtrmm;
}

#[cfg(feature = "cublas-lt")]
pub mod cublasLt {
    pub use crate::cublas_types::*;

    include!(concat!(env!("OUT_DIR"), "/cublasLt.rs"));
}
