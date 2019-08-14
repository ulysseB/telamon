use std::os::raw::c_void;
use std::ptr;
use std::{error, fmt};

use cudnn_sys::*;

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

#[derive(Debug)]
pub struct CudnnError {
    status: cudnnStatus_t,
}

pub type Result<T> = std::result::Result<T, CudnnError>;

impl fmt::Display for CudnnError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            write!(
                fmt,
                "{}",
                std::ffi::CStr::from_ptr(cudnnGetErrorString(self.status))
                    .to_string_lossy()
            )
        }
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

macro_rules! cudnn_call {
    ($e:expr => $v:expr) => {
        {
            let status: cudnnStatus_t = $e;
            if status == CUDNN_STATUS_SUCCESS {
                Ok($v)
            } else {
                Err(CudnnError { status })
            }
        }
    };
    ($e:expr) => {
        cudnn_call!($e => ())
    };
}

pub struct CudnnHandle {
    handle: cudnnHandle_t,
}

impl CudnnHandle {
    pub fn new() -> Result<Self> {
        let mut handle = CudnnHandle {
            handle: ptr::null_mut(),
        };
        unsafe {
            cudnn_call!(cudnnCreate(&mut handle.handle))?;
        }
        Ok(handle)
    }

    pub fn as_raw(&self) -> cudnnHandle_t {
        self.handle
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
        cudnn_call!(cudnnConvolutionForward(
            self.handle,
            alpha,
            x_desc.as_raw(),
            x,
            w_desc.as_raw(),
            w,
            descriptor.as_raw(),
            ConvolutionForwardAlgo::ImplicitGemm.as_raw(),
            ptr::null_mut(),
            0,
            beta,
            y_desc.as_raw(),
            y
        ))
    }

    pub fn destroy(&mut self) -> Result<()> {
        unsafe {
            cudnn_call!(cudnnDestroy(std::mem::replace(
                &mut self.handle,
                ptr::null_mut()
            )))
        }
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, ptr::null_mut());

        if !handle.is_null() {
            let _r = unsafe { cudnn_call!(cudnnDestroy(handle)) };
        }
    }
}

#[derive(Debug, Copy, Clone)]
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

#[derive(Debug, Copy, Clone)]
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
                use ::std::mem::MaybeUninit;

                let mut descriptor: MaybeUninit<$t> = MaybeUninit::uninit();
                unsafe {
                    cudnn_call!($create(descriptor.as_mut_ptr()))?;
                    Ok(descriptor.assume_init())
                }
            }

            pub fn as_raw(&self) -> $t {
                self.$descriptor
            }

            pub fn destroy(&mut self) -> Result<()> {
                let descriptor = ::std::mem::replace(
                    &mut self.$descriptor, ::std::ptr::null_mut());

                unsafe {
                    cudnn_call!($destroy(descriptor))
                }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.$descriptor.is_null() {
                    let _r = self.destroy();
                }
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
