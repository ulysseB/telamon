import abc
import contextlib
import functools
import json
import weakref

import numpy as np
from telamon._capi import ffi, lib

# Initialize Telamon as early as possible.
lib.telamon_init()


class TelamonError(Exception):
    """Exception wrapper for Telamon errors."""

    def __init__(self):
        strerror = lib.telamon_strerror()
        if strerror == ffi.NULL:
            strerror = "<unknown>"
        else:
            strerror = ffi.gc(strerror, lib.telamon_str_free)
            strerror = ffi.string(strerror).decode('utf-8', 'replace')

        super().__init__(strerror)

def _objptr(arg):
    """Helper function to automatically unwrap the `_objptr` field of wrapped
    cffi objects."""

    if isinstance(arg, tuple):
        return tuple(_objptr(arg) for arg in arg)

    return getattr(arg, '_objptr', arg)

def managed(new, delete=None):
    """Decorator for creating functions creating owned objects.

    This returns a function behaving like `new`, except for two points:

     - When `new` returns `NULL`, a `TelamonError` is raised
     - Otherwise, `delete` is registered as a cffi destructor on the returned
       pointer with `ffi.gc`
    """

    @functools.wraps(new)
    def wrapped(*args):
        objptr = new(*_objptr(args))
        if objptr == ffi.NULL:
            raise TelamonError()
        if delete is not None:
            objptr = ffi.gc(objptr, delete)
        return objptr

    return wrapped

def checked(func):
    """Decorator for creating checked functions.

    This returns a function behaving like `func`, except that it raises a
    `TelamonError` if `func` returns a non-`Ok` value.
    """

    @functools.wraps(func)
    def wrapped(*args):
        if func(*_objptr(args)) != lib.TelamonStatus_Ok:
            raise TelamonError()

    return wrapped

_DEPS = weakref.WeakKeyDictionary()

def _record_dependencies(src, *dst):
    assert src not in _DEPS
    _DEPS[src] = dst

class Device(abc.ABC):
    """An abstract base class representing Telamon devices."""

    @abc.abstractmethod
    def _new_context(self):
        raise NotImplementedError


class X86Device(Device):
    """An x86 device."""

    _ffi_context_new = staticmethod(managed(
        lib.telamon_x86_context_new, lib.telamon_context_free))

    def _new_context(self):
        return self._ffi_context_new()


class CudaDevice(Device):
    """A CUDA device."""

    _ffi_executor_new = staticmethod(managed(
        lib.telamon_cuda_executor_new, lib.telamon_cuda_executor_free))
    _ffi_context_new = staticmethod(managed(
        lib.telamon_cuda_context_new, lib.telamon_context_free))

    def __init__(self):
        self._executor = self._ffi_executor_new()

    def _new_context(self):
        context = self._ffi_context_new(self._executor)

        # Record the context -> executor dependency to ensure objects get
        # garbage collected in the right order.
        _record_dependencies(context, self._executor)

        return context

class Context:
    """A Telamon context."""

    def __init__(self, objptr):
        self._objptr = objptr


class ExplorerConfig:
    """Telamon Explorer configuration."""

    _ffi_from_json = staticmethod(
        managed(
            lib.telamon_explorer_config_from_json,
            lib.telamon_explorer_config_free))

    def __init__(self, config=None):
        config_buf = ffi.from_buffer(json.dumps(config or {}).encode())
        self._objptr = self._ffi_from_json(config_buf, len(config_buf))


class SignedKernel:
    """A built kernel with associated signature."""

    _ffi_benchmark = staticmethod(
        checked(lib.telamon_signed_kernel_benchmark))

    def __init__(self, objptr, *, context: Context):
        self._objptr = objptr
        self.context = context

    def benchmark(
            self,
            num_samples: int,
            *,
            context: Context = None,
            config: ExplorerConfig = None,
    ):
        """Benchmark the kernel."""

        if context is None:
            context = self.context

        if not isinstance(config, ExplorerConfig):
            config = ExplorerConfig(config)

        out = np.ndarray(num_samples, dtype=np.float64)
        self._ffi_benchmark(
            self,
            context,
            config,
            num_samples,
            ffi.cast('double *', out.ctypes.data))
        return out


class Kernel:
    """Base class for Python objects representing Telamon kernels."""

    _ffi_build = staticmethod(checked(lib.telamon_kernel_build))

    def build(self, device: Device) -> (SignedKernel, Context):
        # pylint:disable=protected-access
        context = device._new_context()
        # pylint:enable=protected-access

        context_ref = ffi.new("Context*")
        signed_kernel = ffi.new("SignedKernel*")

        self._ffi_build(self, context, signed_kernel, context_ref)

        # The SignedKernel is owned and may contain a reference to the context.
        signed_kernel = ffi.gc(signed_kernel, lib.telamon_signed_kernel_free)
        _record_dependencies(signed_kernel, context)

        # The context_ref is a reference to the context
        _record_dependencies(context_ref, context)

        return SignedKernel(
            signed_kernel,
            context=Context(context_ref),
        )

    def optimize(self, device: Device, config=None):
        return self.build(device).benchmark(0, config=config)


class _Tiling:
    """Helper to convert Python tilings into cffi compatible objects.

    Attributes:
        data_ptr: A pointer to the tiling data that can be used by the C API.
            This pointer points inside a NumPy array that the `_Tiling` object
            has a reference to, but no other reference is guaranteed to exist:
            hence it should not be accessed past the lifetime of the `_Tiling`
            object.
    """

    def __init__(self, tiles, *, copy: bool = True):
        """Initializes a new tiling.

        Args:
            tiles: The tiling specification. If `None`, the tiling is
                unspecified; otherwise, it should be a sequence of integers
                defining each tile.
            copy: If `False` and `tiles` is already a `uint32` NumPy array,
                `_Tiling` will use the `tiles` array directly instead of making
                a copy. In all other cases a copy will be made regardless of
                this argument.
        """
        if tiles is None:
            self._tiles = None
        else:
            self._tiles = np.array(tiles, dtype=np.uint32, copy=copy)

    @property
    def data_ptr(self):
        if self._tiles is None:
            return ffi.NULL

        return ffi.cast("uint32_t *", self._tiles.ctypes.data)

    def __getitem__(self, index):
        if self._tiles is None:
            raise IndexError("cannot index into an unspecified tiling")

        return self._tiles[index]

    def __len__(self):
        if self._tiles is None:
            return 0

        return len(self._tiles)


class MatMul(Kernel):
    """A Matrix Multiply kernel."""

    _ffi_new = staticmethod(managed(
        lib.telamon_kernel_matmul_new, lib.telamon_kernel_free))

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        *,
        a_stride: int = 1,
        transpose_a: bool = False,
        transpose_b: bool = False,
        generic: bool = True,
        m_tiles=None,
        n_tiles=None,
        k_tiles=None
    ):
        """Initializes a new Matrix Multiply kernel."""

        if a_stride < 1:
            raise ValueError("a_stride should be a positive integer.")

        # The X_tiles variable have references to NumPy arrays that are used by
        # the kernel_matmul_new call and must thus outlive it.
        m_tiles = _Tiling(m_tiles, copy=False)
        n_tiles = _Tiling(n_tiles, copy=False)
        k_tiles = _Tiling(k_tiles, copy=False)

        super().__init__(
            self._ffi_new(
                m,
                n,
                k,
                a_stride,
                int(transpose_a),
                int(transpose_b),
                int(generic),
                m_tiles.data_ptr,
                len(m_tiles),
                n_tiles.data_ptr,
                len(n_tiles),
                k_tiles.data_ptr,
                len(k_tiles),
            )
        )
