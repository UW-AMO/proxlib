# cython: infer_types=True
cimport numpy as cnp
cimport cython
from scipy.optimize.cython_optimize cimport brentq


ctypedef struct params:
    int n
    double h
    double l
    double u
    double* y


# user-defined callback
cdef double f(double x, void *args):
    cdef params *p = <params *> args
    cdef double s = 0

    for i in range(p.n):
        s += max(min(p.y[i] - x, p.u), p.l)

    return s - p.h


@cython.boundscheck(False)
@cython.wraparound(False)
def proj_csimplex(double[:, ::1] y_view,
                  double h,
                  double l,
                  double u,
                  xtol=1e-5,
                  rtol=1e-5,
                  mitr=20):
    cdef int m = y_view.shape[0]
    cdef int n = y_view.shape[1]
    cdef params p = {"n": n, "h": h, "l": l, "u": u, "y": &y_view[0][0]}
    cdef double xa
    cdef double xb

    for i in range(m):
        p.y = &y_view[i][0]
        xa = min(y_view[i]) - u
        xb = max(y_view[i]) - l
        x = brentq(f, xa, xb, <params *> &p, xtol, rtol, mitr, NULL)
        for j in range(n):
            y_view[i][j] = max(min(y_view[i][j] - x, u), l) 