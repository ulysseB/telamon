use std::os::raw::c_void;
use std::{error, fmt};

use cuda_sys::cuda::*;
use cuda_sys::cudnn::*;

#[derive(Copy, Clone)]
pub enum CudnnErrorKind {
    NotInitialized,
    AllocFailed,
    BadParam,
    InternalError,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    NotSupported,
    LicenseError,
    RuntimePrerequisiteMissing,
    RuntimeInProgress,
    RuntimeFpOverflow,
    Other,
}

pub struct CudnnError {
    pub status: cudnnStatus_t,
}

pub type Result<T> = std::result::Result<T, CudnnError>;

impl fmt::Debug for CudnnError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}", unsafe {
            std::ffi::CStr::from_ptr(cudnnGetErrorString(self.status)).to_string_lossy()
        })
    }
}

impl fmt::Display for CudnnError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}", unsafe {
            std::ffi::CStr::from_ptr(cudnnGetErrorString(self.status)).to_string_lossy()
        })
    }
}

impl error::Error for CudnnError {}

impl CudnnError {
    pub fn kind(&self) -> CudnnErrorKind {
        use CudnnErrorKind::*;

        match self.status {
            CUDNN_STATUS_NOT_INITIALIZED => NotInitialized,
            CUDNN_STATUS_ALLOC_FAILED => AllocFailed,
            CUDNN_STATUS_BAD_PARAM => BadParam,
            CUDNN_STATUS_INTERNAL_ERROR => InternalError,
            CUDNN_STATUS_INVALID_VALUE => InvalidValue,
            CUDNN_STATUS_ARCH_MISMATCH => ArchMismatch,
            CUDNN_STATUS_MAPPING_ERROR => MappingError,
            CUDNN_STATUS_EXECUTION_FAILED => ExecutionFailed,
            CUDNN_STATUS_NOT_SUPPORTED => NotSupported,
            CUDNN_STATUS_LICENSE_ERROR => LicenseError,
            CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => RuntimePrerequisiteMissing,
            CUDNN_STATUS_RUNTIME_IN_PROGRESS => RuntimeInProgress,
            CUDNN_STATUS_RUNTIME_FP_OVERFLOW => RuntimeFpOverflow,
            _ => Other,
        }
    }
}

#[macro_export]
macro_rules! cudnn_call {
    ($e:expr) => {{
        let status: ::cuda_sys::cudnn::cudnnStatus_t = $e;
        if status == ::cuda_sys::cudnn::CUDNN_STATUS_SUCCESS {
            Ok(())
        } else {
            Err($crate::CudnnError { status })
        }
    }};
}

macro_rules! cudnn_create {
    ($create:ident ( $($pre:expr, )* _ $(: $t:ty)? $(, $post:expr)* )) => {{
        let mut slot $(: ::std::mem::MaybeUninit<$t>)? = ::std::mem::MaybeUninit::uninit();
        match cudnn_call!($create($($pre, )* slot.as_mut_ptr() $(, $post)*)) {
            Ok(()) => Ok(slot.assume_init()),
            Err(err) => Err(err),
        }
    }};
}

macro_rules! cudnn_destroy {
    ($destroy:ident, $handle:expr) => {{
        let handle = ::std::mem::replace($handle, ::std::ptr::null_mut());
        if !handle.is_null() {
            cudnn_call!($destroy(handle))
        } else {
            Ok(())
        }
    }};
}

pub struct ConvolutionForwardParams<'a> {
    handle: &'a CudnnHandle,
    x_desc: &'a TensorDescriptor,
    w_desc: &'a FilterDescriptor,
    descriptor: &'a ConvolutionDescriptor,
    y_desc: &'a TensorDescriptor,
    algo: ConvolutionForwardAlgo,
    // Workspace
    ws_size: usize,
    ws_data: CUdeviceptr,
}

impl<'a> ConvolutionForwardParams<'a> {
    pub unsafe fn call(
        &mut self,
        alpha: *const c_void,
        x: *const c_void,
        w: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
    ) -> Result<()> {
        cudnn_call!(cudnnConvolutionForward(
            self.handle.as_raw(),
            alpha,
            self.x_desc.as_raw(),
            x,
            self.w_desc.as_raw(),
            w,
            self.descriptor.as_raw(),
            self.algo.as_raw(),
            self.ws_data as *mut c_void,
            self.ws_size,
            beta,
            self.y_desc.as_raw(),
            y
        ))
    }
}

impl<'a> Drop for ConvolutionForwardParams<'a> {
    fn drop(&mut self) {
        if std::mem::replace(&mut self.ws_size, 0) > 0 {
            unsafe {
                assert_eq!(cuMemFree_v2(self.ws_data), CUDA_SUCCESS);
            }
        }
    }
}

pub struct CudnnHandle {
    handle: cudnnHandle_t,
}

impl CudnnHandle {
    pub fn new() -> Result<Self> {
        unsafe { cudnn_create!(cudnnCreate(_)) }.map(|handle| CudnnHandle { handle })
    }

    pub fn as_raw(&self) -> cudnnHandle_t {
        self.handle
    }

    pub fn convolution_forward<'a>(
        &'a self,
        x_desc: &'a TensorDescriptor,
        w_desc: &'a FilterDescriptor,
        descriptor: &'a ConvolutionDescriptor,
        y_desc: &'a TensorDescriptor,
        algo: ConvolutionForwardAlgo,
    ) -> Result<ConvolutionForwardParams<'a>> {
        unsafe {
            let ws_size = cudnn_create!(cudnnGetConvolutionForwardWorkspaceSize(
                self.handle,
                x_desc.as_raw(),
                w_desc.as_raw(),
                descriptor.as_raw(),
                y_desc.as_raw(),
                algo.as_raw(),
                _: usize))?;

            let ws_data = {
                let mut ws_data: CUdeviceptr = 0;
                if ws_size > 0 {
                    assert_eq!(cuMemAlloc_v2(&mut ws_data, ws_size), CUDA_SUCCESS);
                }
                ws_data
            };

            Ok(ConvolutionForwardParams {
                handle: self,
                x_desc,
                w_desc,
                descriptor,
                y_desc,
                algo,
                ws_size,
                ws_data,
            })
        }
    }

    pub unsafe fn convolution_forward_implicit_gemm(
        &self,
        alpha: *const c_void,
        x_desc: &TensorDescriptor,
        x: *const c_void,
        w_desc: &FilterDescriptor,
        w: *const c_void,
        descriptor: &ConvolutionDescriptor,
        beta: *const c_void,
        y_desc: &TensorDescriptor,
        y: *mut c_void,
    ) -> Result<()> {
        self.convolution_forward(
            x_desc,
            w_desc,
            descriptor,
            y_desc,
            ConvolutionForwardAlgo::ImplicitGemm,
        )?
        .call(alpha, x, w, beta, y)
    }

    pub unsafe fn convolution_forward_implicit_precomp_gemm(
        &self,
        alpha: *const c_void,
        x_desc: &TensorDescriptor,
        x: *const c_void,
        w_desc: &FilterDescriptor,
        w: *const c_void,
        descriptor: &ConvolutionDescriptor,
        beta: *const c_void,
        y_desc: &TensorDescriptor,
        y: *mut c_void,
    ) -> Result<()> {
        self.convolution_forward(
            x_desc,
            w_desc,
            descriptor,
            y_desc,
            ConvolutionForwardAlgo::ImplicitPrecompGemm,
        )?
        .call(alpha, x, w, beta, y)
    }

    pub fn destroy(&mut self) -> Result<()> {
        unsafe { cudnn_destroy!(cudnnDestroy, &mut self.handle) }
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        let _r = self.destroy();
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TensorFormat {
    Nchw,
    Nhwc,
    NchwVectC,
}

impl TensorFormat {
    pub fn as_raw(self) -> cudnnTensorFormat_t {
        use TensorFormat::*;

        match self {
            Nchw => CUDNN_TENSOR_NCHW,
            Nhwc => CUDNN_TENSOR_NHWC,
            NchwVectC => CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DataType {
    Float,
    Double,
    Half,
    Int8,
    Int32,
    Int8x4,
    Int8x32,
    // Since 7.1
    Uint8,
    Uint8x4,
}

impl DataType {
    pub fn as_raw(self) -> cudnnDataType_t {
        use DataType::*;

        match self {
            Float => CUDNN_DATA_FLOAT,
            Double => CUDNN_DATA_DOUBLE,
            Half => CUDNN_DATA_HALF,
            Int8 => CUDNN_DATA_INT8,
            Int32 => CUDNN_DATA_INT32,
            Int8x4 => CUDNN_DATA_INT8x4,
            Int8x32 => CUDNN_DATA_INT8x32,
            Uint8 => CUDNN_DATA_UINT8,
            Uint8x4 => CUDNN_DATA_UINT8x4,
        }
    }
}

macro_rules! cudnn_descriptor {
    ($v:vis struct $name:ident[$create:ident, $destroy:ident] { $descriptor:ident: $t:ty, }) => {
        $v struct $name { $descriptor: $t }

        impl $name {
            fn create() -> Result<$t> {
                unsafe { cudnn_create!($create(_)) }
            }

            pub fn as_raw(&self) -> $t {
                self.$descriptor
            }

            pub fn destroy(&mut self) -> Result<()> {
                unsafe { cudnn_destroy!($destroy, &mut self.$descriptor) }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                let _r = self.destroy();
            }
        }
    };
}

cudnn_descriptor! {
    pub struct TensorDescriptor[cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor] {
        descriptor: cudnnTensorDescriptor_t,
    }
}

impl TensorDescriptor {
    pub fn new_4d(
        format: TensorFormat,
        data_type: DataType,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<Self> {
        let descriptor = Self::create()?;
        unsafe {
            cudnn_call!(cudnnSetTensor4dDescriptor(
                descriptor,
                format.as_raw(),
                data_type.as_raw(),
                n,
                c,
                h,
                w
            ))?;
        }
        Ok(TensorDescriptor { descriptor })
    }
}

cudnn_descriptor! {
    pub struct FilterDescriptor[cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor] {
        descriptor: cudnnFilterDescriptor_t,
    }
}

impl FilterDescriptor {
    pub fn new_4d(
        data_type: DataType,
        format: TensorFormat,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<Self> {
        let descriptor = Self::create()?;
        unsafe {
            cudnn_call!(cudnnSetFilter4dDescriptor(
                descriptor,
                data_type.as_raw(),
                format.as_raw(),
                k,
                c,
                h,
                w
            ))?;
        }
        Ok(FilterDescriptor { descriptor })
    }
}

cudnn_descriptor! {
    pub struct ConvolutionDescriptor[cudnnCreateConvolutionDescriptor, cudnnDestroyConvolutionDescriptor] {
        descriptor: cudnnConvolutionDescriptor_t,
    }
}

impl ConvolutionDescriptor {
    pub fn new_2d(
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: ConvolutionMode,
        compute_type: DataType,
    ) -> Result<Self> {
        let descriptor = Self::create()?;
        unsafe {
            cudnn_call!(cudnnSetConvolution2dDescriptor(
                descriptor,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                mode.as_raw(),
                compute_type.as_raw()
            ))?;
        }
        Ok(ConvolutionDescriptor { descriptor })
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ConvolutionForwardAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonfused,
}

impl fmt::Display for ConvolutionForwardAlgo {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            ConvolutionForwardAlgo::ImplicitGemm => "_IMPLICIT_GEMM",
            ConvolutionForwardAlgo::ImplicitPrecompGemm => "_IMPLICIT_PRECOMP_GEMM",
            ConvolutionForwardAlgo::Gemm => "_GEMM",
            ConvolutionForwardAlgo::Direct => "_DIRECT",
            ConvolutionForwardAlgo::Fft => "_FFT",
            ConvolutionForwardAlgo::FftTiling => "_FFT_TILING",
            ConvolutionForwardAlgo::Winograd => "_WINOGRAD",
            ConvolutionForwardAlgo::WinogradNonfused => "_WINOGRAD_NONFUSED",
        })
    }
}

impl ConvolutionForwardAlgo {
    pub fn as_raw(self) -> cudnnConvolutionFwdAlgo_t {
        use ConvolutionForwardAlgo::*;

        match self {
            ImplicitGemm => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            ImplicitPrecompGemm => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            Gemm => CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            Direct => CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            Fft => CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            FftTiling => CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            Winograd => CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            WinogradNonfused => CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ConvolutionMode {
    Convolution,
    CrossCorrelation,
}

impl ConvolutionMode {
    pub fn as_raw(self) -> cudnnConvolutionMode_t {
        use ConvolutionMode::*;

        match self {
            Convolution => CUDNN_CONVOLUTION,
            CrossCorrelation => CUDNN_CROSS_CORRELATION,
        }
    }
}
