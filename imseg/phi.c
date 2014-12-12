#include "phi.h"

static PyMethodDef phiMethods[] = {
    {"phi", phi, METH_VARARGS, "plabels, n, m, factors"},
    {"phi_blanket", phi_blanket, METH_VARARGS, "plabels, n, m, factors"},
    {"phi_all", phi_all, METH_VARARGS, "plabels, factors"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initphi(void) {
    Py_InitModule("phi", phiMethods);
}

static PyObject *phi(PyObject *self, PyObject *args) {

    Py_buffer b_plabels, b_factors;
    int64_t *plabels;
    int n, m, N, M, phi_key, label, comp;
    double *factors;

    if (!PyArg_ParseTuple(args, "w*iiw*", &b_plabels, &n, &m, &b_factors))
        return NULL;

    // Check plabels
    if (b_plabels.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "expected 2d plabels");
        return NULL;
    }
    plabels = b_plabels.buf;
    N = b_plabels.shape[0] - 2; // shape of original image
    M = b_plabels.shape[1] - 2; // shape of original image
    if (n < 0 || n > N-1) {
        PyErr_SetString(PyExc_ValueError, "incorrect index n");
        return NULL;
    }
    if (m < 0 || m > M-1) {
        PyErr_SetString(PyExc_ValueError, "incorrect index m");
        return NULL;
    }

    // Check factors
    if (b_factors.ndim != 1 && b_factors.shape[0] != 256) {
        PyErr_SetString(PyExc_ValueError, "expected factors of shape (256,)");
        return NULL;
    }
    factors = b_factors.buf;

   // Compute
    phi_key = 0;
    label = plabels[(n+1)*(M+2) + m+1];
    phi_key += (int)(plabels[(n+0)*(M+2) + m+0] == label) << 0;
    phi_key += (int)(plabels[(n+1)*(M+2) + m+0] == label) << 1;
    phi_key += (int)(plabels[(n+2)*(M+2) + m+0] == label) << 2;
    phi_key += (int)(plabels[(n+0)*(M+2) + m+1] == label) << 3;
    phi_key += (int)(plabels[(n+2)*(M+2) + m+1] == label) << 4;
    phi_key += (int)(plabels[(n+0)*(M+2) + m+2] == label) << 5;
    phi_key += (int)(plabels[(n+1)*(M+2) + m+2] == label) << 6;
    phi_key += (int)(plabels[(n+2)*(M+2) + m+2] == label) << 7;
    return PyFloat_FromDouble(-1.*factors[phi_key]);

}

static PyObject *phi_blanket(PyObject *self, PyObject *args) {

    Py_buffer b_plabels, b_factors;
    int64_t *plabels;
    int n, m, N, M, phi_key, label, i, j, comp;
    double *factors, result;

    if (!PyArg_ParseTuple(args, "w*iiw*", &b_plabels, &n, &m, &b_factors))
        return NULL;

    // Check plabels
    if (b_plabels.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "expected 2d plabels");
        return NULL;
    }
    plabels = b_plabels.buf;
    N = b_plabels.shape[0] - 2;
    M = b_plabels.shape[1] - 2;
    if (n < 0 || n > N-1) {
        PyErr_SetString(PyExc_ValueError, "incorrect index n");
        return NULL;
    }
    if (m < 0 || m > M-1) {
        PyErr_SetString(PyExc_ValueError, "incorrect index m");
        return NULL;
    }

    // Check factors
    if (b_factors.ndim != 1 && b_factors.shape[0] != 256) {
        PyErr_SetString(PyExc_ValueError, "expected factors of shape (256,)");
        return NULL;
    }
    factors = b_factors.buf;

    // Compute
    result = 0.;
    for (i=max(n-1, 0); i<min(n+2, N); i++) {
        for (j=max(m-1, 0); j<min(m+2, M); j++) {
            phi_key = 0;
            label = plabels[(i+1)*(M+2) + j+1];
            phi_key += (int)(plabels[(i+0)*(M+2) + j+0] == label) << 0;
            phi_key += (int)(plabels[(i+1)*(M+2) + j+0] == label) << 1;
            phi_key += (int)(plabels[(i+2)*(M+2) + j+0] == label) << 2;
            phi_key += (int)(plabels[(i+0)*(M+2) + j+1] == label) << 3;
            phi_key += (int)(plabels[(i+2)*(M+2) + j+1] == label) << 4;
            phi_key += (int)(plabels[(i+0)*(M+2) + j+2] == label) << 5;
            phi_key += (int)(plabels[(i+1)*(M+2) + j+2] == label) << 6;
            phi_key += (int)(plabels[(i+2)*(M+2) + j+2] == label) << 7;
            result -= factors[phi_key];
        }
    }
    return PyFloat_FromDouble(result);
}

static PyObject *phi_all(PyObject *self, PyObject *args) {

    Py_buffer b_plabels, b_factors;
    int64_t *plabels;
    int N, M, phi_key, label, i, j, comp;
    double *factors, result;

    if (!PyArg_ParseTuple(args, "w*w*", &b_plabels, &b_factors))
        return NULL;

    // Check plabels
    if (b_plabels.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "expected 2d plabels");
        return NULL;
    }
    plabels = b_plabels.buf;
    N = b_plabels.shape[0] - 2;
    M = b_plabels.shape[1] - 2;

    // Check factors
    if (b_factors.ndim != 1 && b_factors.shape[0] != 256) {
        PyErr_SetString(PyExc_ValueError, "expected factors of shape (256,)");
        return NULL;
    }
    factors = b_factors.buf;

    // Compute
    result = 0.;
    for (i=0; i<N; i++) {
        for (j=0; j<M; j++) {
            phi_key = 0;
            label = plabels[(i+1)*(M+2) + j+1];
            phi_key += (int)(plabels[(i+0)*(M+2) + j+0] == label) << 0;
            phi_key += (int)(plabels[(i+1)*(M+2) + j+0] == label) << 1;
            phi_key += (int)(plabels[(i+2)*(M+2) + j+0] == label) << 2;
            phi_key += (int)(plabels[(i+0)*(M+2) + j+1] == label) << 3;
            phi_key += (int)(plabels[(i+2)*(M+2) + j+1] == label) << 4;
            phi_key += (int)(plabels[(i+0)*(M+2) + j+2] == label) << 5;
            phi_key += (int)(plabels[(i+1)*(M+2) + j+2] == label) << 6;
            phi_key += (int)(plabels[(i+2)*(M+2) + j+2] == label) << 7;
            result -= factors[phi_key];
        }
    }
    return PyFloat_FromDouble(result);
}


