import contextlib
import json
from telamon._capi import ffi, lib

# Initialize Rust logger early.
lib.env_logger_try_init()

class TelamonError(Exception):
    """Base error class for Telamon errors."""

# FIXME: The device stack should be a thread-local variable.
_DEVICE_STACK = []

@contextlib.contextmanager
def device(device_spec: str):
    """A context manager setting the device to execute kernels on.

    .. code-block::
        # Executed on the default device
        tl.MatMul(1024, 1024, 1024).optimize()

        # Executed on the GPU (if available)
        with tl.device('GPU'):
            tl.MatMul(1024, 1024, 1024).optimize()

    Args:
        device_spec: The device specification to use. One of "CPU" or "GPU".
    """

    if device_spec.upper() == 'CPU':
        device_id = lib.X86
    elif device_spec.upper() == 'GPU':
        device_id = lib.Cuda
    else:
        raise ValueError(
            'Invalid device specification: {}; expected "CPU" or "GPU"'.format(
                device_spec))

    _DEVICE_STACK.append(device_id)
    try:
        yield
    finally:
        popped_id = _DEVICE_STACK.pop()
        assert popped_id == device_id

def _get_current_device_id():
    return lib.X86 if not _DEVICE_STACK else _DEVICE_STACK[-1]

class RustObject:
    """Thin wrapper around a Rust object."""

    __slots__ = ('_objptr', )

    # The deallocation function. This should be defined by subclasses, but we
    # allow it to be left as `None` for non-allocated types that are moved out
    # of the C API directly (e.g. integers or booleans).
    _dealloc_ = None

    def __init__(self, objptr):
        self._objptr = objptr

    @property
    def objptr(self):
        assert self._objptr is not None
        return self._objptr

    def __del__(self):
        # This should not happen as __del__ usually shouldn't be called more
        # than once, but we may as well be extra careful and not send a null
        # pointer to the deallocation function.
        if self._objptr is None:
            return

        dealloc = self.__class__._dealloc_
        if dealloc is not None:
            dealloc(self._objptr)

        # Prevent double free/use after free.
        self._objptr = None

class Kernel(RustObject):
    """Base class for Python objects representing Telamon kernels."""

    _dealloc_ = lib.kernel_free

    def optimize(self, config=None):
        config_bytes = json.dumps(config or {}).encode()

        if not lib.kernel_optimize(
                self.objptr,
                _get_current_device_id(),
                ffi.new("char[]", config_bytes),
                len(config_bytes)):
            raise TelamonError(
                'Optimization failed.')

def _ffi_tiling(tiles):
    """Helper to convert Python tilings into a cffi compatible object.

    Args:
        tiles: The tiles ton convert. Must be either `None` (allow all tilings
            across this axis) or a single tile definition.

    Returns:
        A cffi object representing a newly allocated tiling object.
    """
    if tiles is None:
        return ffi.NULL

    c_tiles = ffi.new('Tiling *')
    c_tiles.length = len(tiles)
    c_tiles.data = ffi.new('unsigned int *', len(tiles))
    c_tiles.data[0:len(tiles)] = tiles
    return c_tiles

class MatMul(Kernel):
    """A Matrix Multiply kernel."""

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
            k_tiles=None):
        """Initializes a new Matrix Multiply kernel."""

        if a_stride < 1:
            raise ValueError(
                'a_stride should be a positive integer.')

        super().__init__(
            lib.kernel_matmul_new(
                m, n, k,
                a_stride, int(transpose_a), int(transpose_b), int(generic),
                _ffi_tiling(m_tiles),
                _ffi_tiling(n_tiles),
                _ffi_tiling(k_tiles)))
