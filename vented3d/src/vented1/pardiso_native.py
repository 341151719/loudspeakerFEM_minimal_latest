"""Own a oneMKL PARDISO handle through the wheel's exported 64-bit C API.

The 0.0.6 Python wrapper releases its handle on any iparm change, including
solve-only iparm[11]. This adapter keeps solve-mode changes local to our own
handle. It never accesses or modifies another wrapper's private memory.
"""
from __future__ import annotations
import ctypes as ct
import os
import numpy as np


class PardisoSolver:
    def __init__(self, matrix):
        # A one-equation core needs no sparse factorization. The installed MKL
        # runtime intermittently corrupts the heap on this degenerate path in
        # sphere controls; avoid that path with the exact scalar inverse.
        self._scalar = None
        if matrix.shape == (1, 1):
            value = complex(matrix[0, 0])
            if value == 0 or not np.isfinite(value):
                raise ValueError('Scalar core must be finite and nonsingular')
            self.n = 1
            self.iparm = np.zeros(64, dtype=np.int64)
            self._scalar = value
            self._released = False
            self.reported_threads = 1
            self.backend = 'exact scalar inverse'
            return
        try:
            runtime = os.environ.get('VENTED_MKL_RT')
            if runtime:
                self.library = ct.CDLL(runtime)
                self.library.MKL_Set_Num_Threads.argtypes = [ct.c_int]
                self.library.MKL_Set_Num_Threads.restype = None
                self.library.MKL_Set_Dynamic.argtypes = [ct.c_int]
                self.library.MKL_Set_Dynamic.restype = None
                self.library.MKL_Get_Max_Threads.restype = ct.c_int
                self.library.MKL_Set_Dynamic(0)
                self.library.MKL_Set_Num_Threads(int(os.environ.get('MKL_NUM_THREADS','1')))
                self.reported_threads = self.library.MKL_Get_Max_Threads()
                self.backend = runtime
            else:
                from pymklpardiso import _mkl_pardiso
                self.library = ct.CDLL(_mkl_pardiso.__file__)
                self.reported_threads = 1
                self.backend = _mkl_pardiso.__file__
            self.function = self.library.pardiso_64
        except (ImportError, AttributeError, OSError) as exc:
            raise ImportError("The MKL wheel must expose the pardiso_64 C symbol") from exc
        self.function.argtypes = [ct.c_void_p]*16
        self.function.restype = None
        self.handle = (ct.c_void_p*64)()
        self.iparm = np.zeros(64, dtype=np.int64)
        self.iparm[0] = 1
        self.iparm[1] = 2  # METIS fill-reducing ordering
        self.iparm[7] = 2  # iterative refinement steps
        self.iparm[9] = 13 # nonsymmetric pivot perturbation exponent
        self.iparm[10] = 1 # nonsymmetric scaling
        self.iparm[12] = 1 # weighted matching
        self.iparm[34] = 1 # zero-based CSR
        self.n = matrix.shape[0]
        self.ia = np.ascontiguousarray(matrix.indptr, dtype=np.int64)
        self.ja = np.ascontiguousarray(matrix.indices, dtype=np.int64)
        self.values = np.ascontiguousarray(matrix.data, dtype=np.complex128)
        self.perm = np.zeros(self.n, dtype=np.int64)
        self._released = False
        try:
            self._call(12)  # analysis and factorization
        except BaseException:
            self.release()
            raise

    def _call(self, phase, b=None, x=None):
        if self._released:
            raise RuntimeError("PARDISO handle has been released")
        values = [ct.c_int64(v) for v in
                  (1, 1, 13, phase, self.n, 1 if b is None or b.ndim == 1 else b.shape[1], 0, 0)]
        maxfct, mnum, mtype, phase_arg, n, nrhs, msglvl, error = values
        dummy = np.zeros(1, dtype=np.complex128)
        self.function(self.handle, ct.byref(maxfct), ct.byref(mnum),
                      ct.byref(mtype), ct.byref(phase_arg), ct.byref(n),
                      self.values.ctypes.data, self.ia.ctypes.data,
                      self.ja.ctypes.data, self.perm.ctypes.data, ct.byref(nrhs),
                      self.iparm.ctypes.data, ct.byref(msglvl),
                      dummy.ctypes.data if b is None else b.ctypes.data,
                      dummy.ctypes.data if x is None else x.ctypes.data,
                      ct.byref(error))
        if error.value:
            raise RuntimeError(f"oneMKL PARDISO phase {phase}: error {error.value}")

    def solve(self, b):
        if self._released:
            raise RuntimeError("PARDISO handle has been released")
        b = np.array(b, dtype=np.complex128, order='F', copy=True)
        if b.ndim not in (1, 2) or b.shape[0] != self.n:
            raise ValueError("RHS must have shape (n,) or (n, nrhs)")
        if self._scalar is not None:
            value = self._scalar.conjugate() if self.iparm[11] == 1 else self._scalar
            return b/value
        x = np.empty_like(b, order='F')
        self._call(33, b, x)
        return x

    def get_iparm_value(self, index):
        return int(self.iparm[index])

    def set_iparm(self, index, value):
        if index != 11 or value not in (0, 1, 2):
            raise ValueError("Only solve-only iparm[11] changes are supported")
        self.iparm[index] = value

    def release(self):
        if not self._released:
            if self._scalar is not None:
                self._released = True
                return
            try:
                self._call(-1)
            finally:
                self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
