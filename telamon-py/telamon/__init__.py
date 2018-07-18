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

class _Tiling:
    """Helper to convert Python tilings into a cffi compatible object.

    _Tiling instances wrap a pointer to a cffi-allocated copy of the
    tiling data as well as the tiles length. It must outlive the usage
    of the underlying data pointer.
    """

    def __init__(self, tiles):
        """Initializes a new _Tiling wrapper.

        Args:
            tiles: The tiles ton convert. Must be either `None` (allow all tilings
                across this axis) or a single tile definition.
        """

        if tiles is None:
            self.data = ffi.NULL
            self.length = 0

        else:
            cdata = ffi.new('unsigned int *', len(tiles))
            cdata[0:len(tiles)] = tiles
            self.data = cdata
            self.length = len(tiles)


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

        # We need to store the _Tiling objects into variables in order
        # for the data pointers to stay alive until after the call
        # below.
        m_tiles = _Tiling(m_tiles)
        n_tiles = _Tiling(n_tiles)
        k_tiles = _Tiling(k_tiles)

        super().__init__(
            lib.kernel_matmul_new(
                m, n, k,
                a_stride, transpose_a, transpose_b, generic,
                m_tiles.data, m_tiles.length,
                n_tiles.data, n_tiles.length,
                k_tiles.data, k_tiles.length))
