#include <Python.h>
#include <stdint.h>

inline int max(int a, int b) { return a > b ? a : b;}
inline int min(int a, int b) { return a < b ? a : b;}

static PyObject *phi(PyObject *self, PyObject *args);
static PyObject *phi_blanket(PyObject *self, PyObject *args);
static PyObject *phi_all(PyObject *self, PyObject *args);
