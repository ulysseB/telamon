import contextlib
import json
import numpy as np
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

    if device_spec.upper() == "CPU":
        device_id = lib.DeviceId_X86
    elif device_spec.upper() == "GPU":
        device_id = lib.DeviceId_Cuda
    else:
        raise ValueError(
            'Invalid device specification: {}; expected "CPU" or "GPU"'.format(
                device_spec
            )
        )

    _DEVICE_STACK.append(device_id)
    try:
        yield
    finally:
        popped_id = _DEVICE_STACK.pop()
        assert popped_id == device_id


def _get_current_device_id():
    return lib.DeviceId_X86 if not _DEVICE_STACK else _DEVICE_STACK[-1]


class RustObject:
    """Thin wrapper around a Rust object."""

    __slots__ = ("_objptr",)

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
            len(config_bytes),
        ):
            raise TelamonError("Optimization failed.")


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
            lib.kernel_matmul_new(
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
