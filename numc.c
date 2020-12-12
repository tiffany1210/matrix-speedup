#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    PyObject *mat = args;
    if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be of type numc.Matrix!");
        return NULL;
    }
    Matrix61c *mat61c = (Matrix61c*) args;
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    if (rows != mat61c->mat->rows || cols != mat61c->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match!");
        return NULL;
    }
    matrix *sum = NULL;
    int allocate_failed = allocate_matrix(&sum, self->mat->rows, self->mat->cols);
    if (allocate_failed) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation for result failed!");
        return NULL;
    }
    int add_failed = add_matrix(sum, self->mat, mat61c->mat);
    if (add_failed) {
        return NULL;
    }
    Matrix61c *rmat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    rmat->mat = sum;
    rmat->shape = get_shape(sum->rows, sum->cols);
    return (PyObject*)rmat;
}

/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    PyObject *mat = args;
    if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be of type numc.Matrix!");
        return NULL;
    }
    Matrix61c *mat61c = (Matrix61c*) mat;
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    if (rows != mat61c->mat->rows || cols != mat61c->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match!");
        return NULL;
    }
    matrix *sub = NULL;
    int allocate_failed = allocate_matrix(&sub, self->mat->rows, self->mat->cols);
    if (allocate_failed) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation for result failed!");
        return NULL;
    }
    int sub_failed = sub_matrix(sub, self->mat, mat61c->mat);
    if (sub_failed) {
        return NULL;
    }
    Matrix61c *rmat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    rmat->mat = sub;
    rmat->shape = get_shape(sub->rows, sub->cols);
    return (PyObject*) rmat;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    PyObject *mat = args;
    if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be of type numc.Matrix!");
        return NULL;
    }
    Matrix61c *mat61c = (Matrix61c *) mat;
    int rows = self->mat->rows;
    int cols = mat61c->mat->cols;
    if (self->mat->cols != mat61c->mat->rows) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match!");
        return NULL;
    }
    matrix *product = NULL;
    int allocate_failed = allocate_matrix(&product, rows, cols);
    if (allocate_failed) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation for result failed");
        return NULL;
    }
    int mul_failed = mul_matrix(product, self->mat, mat61c->mat);
    if (mul_failed) {
        PyErr_SetString(PyExc_TypeError, "dimensions do not match");
        return NULL;
    }
    Matrix61c *rmat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    rmat->mat = product;
    rmat->shape = get_shape(product->rows, product->cols);
    return (PyObject*)rmat; 
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    matrix *neg = NULL;
    int allocate_failed = allocate_matrix(&neg, self->mat->rows, self->mat->cols);
    if (allocate_failed) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation for result failed");
        return NULL;
    }
    int neg_failed = neg_matrix(neg, self->mat);
    if (neg_failed) {
        PyErr_SetString(PyExc_TypeError, "dimensions do not match");
        return NULL;
    }
    Matrix61c *rmat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    rmat->mat = neg;
    rmat->shape = get_shape(neg->rows, neg->cols);
    return (PyObject*)rmat;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    matrix *abs = NULL;
    int allocate_failed = allocate_matrix(&abs, self->mat->rows, self->mat->cols);
    if (allocate_failed) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation for the result failed");
        return NULL;
    }
    int abs_failed = abs_matrix(abs, self->mat);
    if (abs_failed) {
        PyErr_SetString(PyExc_TypeError, "dimensions do not match");
        return NULL;
    } 
    Matrix61c *rmat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    rmat->mat = abs;
    rmat->shape = get_shape(abs->rows, abs->cols);
    return (PyObject*)rmat;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    if (!PyLong_Check(pow)) {
        PyErr_SetString(PyExc_TypeError, "pow should be an integer!");
        return NULL;
    }
    if (self->mat->rows != self->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match, must be a square matrix!");
        return NULL;
    }
    int power = PyLong_AsLong(pow);
    if (power < 0) {
        PyErr_SetString(PyExc_ValueError, "power must be nonnegative.");
        return NULL;
    }
    matrix *pow_mat = NULL;
    int allocate_failed = allocate_matrix(&pow_mat, self->mat->rows, self->mat->cols);
    if (allocate_failed) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation for result failed!");
        return NULL;
    }
    int pow_failed = pow_matrix(pow_mat, self->mat, power);
    if (pow_failed) {
        PyErr_SetString(PyExc_TypeError, "dimensions do not match");
        return NULL;
    } 
    Matrix61c *rmat = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    rmat->mat = pow_mat;
    rmat->shape = get_shape(pow_mat->rows, pow_mat->cols);
    return (PyObject*) rmat;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    .nb_add = (binaryfunc) &Matrix61c_add,
    .nb_subtract = (binaryfunc) &Matrix61c_sub,
    .nb_multiply = (binaryfunc) &Matrix61c_multiply,
    .nb_power = (ternaryfunc) &Matrix61c_pow,
    .nb_negative = (unaryfunc) &Matrix61c_neg,
    .nb_absolute = (unaryfunc) &Matrix61c_abs,
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 3, 3, &arg1, &arg2, &arg3)) {
        if (!PyLong_Check(arg1) || !PyLong_Check(arg2)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments for row and col");
            return NULL;
        }
        int row = PyLong_AsLong(arg1);
        int col = PyLong_AsLong(arg2);
        if (row < 0 || row >= self->mat->rows || col < 0 || col >= self->mat->cols) {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            return NULL;
        }
        double val;
        if (PyFloat_Check(arg3)) {
            val = PyFloat_AsDouble(arg3);
        } else if (PyLong_Check(arg3)) {
            val = PyLong_AsDouble(arg3);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid argument for val");
            return NULL;
        }
        set(self->mat, row, col, val);
        return Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    if (PyArg_UnpackTuple(args, "args", 2, 2, &arg1, &arg2)) {
        if (!PyLong_Check(arg1) || !PyLong_Check(arg2)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return NULL;
        }
        int row = PyLong_AsLong(arg1);
        int col = PyLong_AsLong(arg2);
        if (row < 0 || row >= self->mat->rows || col < 0 || col >= self->mat->cols) {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            return NULL;
        } 
        return PyFloat_FromDouble(get(self->mat, row, col));
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    } 
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    {"get", (PyCFunction) Matrix61c_get_value, METH_VARARGS, "returns value at a given position"},
    {"set", (PyCFunction) Matrix61c_set_value, METH_VARARGS, "sets value at a given position"},
    {NULL, NULL, 0, NULL}
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    /* if 1-d matrix */
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t slicelength;
    if (self->mat->is_1d) {
        Py_ssize_t size = self->mat->rows * self->mat->cols;
        if (PyLong_Check(key)) {
            int index = PyLong_AsLong(key);
            if (index >= size || index < 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of range!");
                return NULL;
            } else {
                return PyFloat_FromDouble(self->mat->data_1[index]);
            } 
        } else if (PySlice_Check(key)) {
            PySlice_GetIndicesEx(key, size, &start, &stop, &step, &slicelength);
            if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "invalid slice!");
                return NULL;
            } else if (slicelength == 1) {
                return PyFloat_FromDouble(self->mat->data_1[start]);
            } else {
                Matrix61c* mat1 = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                if (self->mat->cols == 1) {
                    allocate_matrix_ref(&(mat1->mat), self->mat, start, 0, slicelength, 1);
                } else {
                    allocate_matrix_ref(&(mat1->mat), self->mat, 0, start, 1, slicelength);
                }
                mat1->shape = get_shape(slicelength, 1);
                return (PyObject*)mat1;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments for 1d matrix");
            return NULL;
        }
    } else {
        PyObject *arg1 = NULL;
        PyObject *arg2 = NULL;
        int size = self->mat->rows;
        Matrix61c* mat2 = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
        if (PyLong_Check(key)) {
            int index = PyLong_AsLong(key);
            if (index >= size || index < 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of range!");
                return NULL;
            } else {
                allocate_matrix_ref(&(mat2->mat), self->mat, index, 0, 1, self->mat->cols);
                mat2->shape = get_shape(1, self->mat->cols);
                return (PyObject*) mat2;
            }
        } else if (PySlice_Check(key)) {
            PySlice_GetIndicesEx(key, size, &start, &stop, &step, &slicelength);
            if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "invalid slice!");
                return NULL;
            } else if (slicelength == 1) {
                return PyFloat_FromDouble(*(self->mat->data[start]));
            } else {
                allocate_matrix_ref(&(mat2->mat), self->mat, start, 0, slicelength, self->mat->cols);
                mat2->shape = get_shape(slicelength, self->mat->cols);
                return (PyObject*)mat2;
            }
        } else if (PyArg_UnpackTuple(key, "key", 2, 2, &arg1, &arg2)) {
            int size1 = self->mat->rows;
            int size2 = self->mat->cols;
            /*if arg1 and arg2 = integers*/
            if (PyLong_Check(arg1) && PyLong_Check(arg2)) {
                int row = PyLong_AsLong(arg1);
                int col = PyLong_AsLong(arg2);
                if (row < 0 || row >= self->mat->rows || col < 0 || col >= self->mat->cols) {
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return NULL;
                } 
                return PyFloat_FromDouble(get(self->mat, row, col));
            } else if (PyLong_Check(arg1) && PySlice_Check(arg2)) {
                int index = PyLong_AsLong(arg1);
                if (index >= size1 || index < 0) {
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return NULL;
                } else {
                    PySlice_GetIndicesEx(arg2, size2, &start, &stop, &step, &slicelength);
                    if (slicelength < 1 || step != 1) {
                        PyErr_SetString(PyExc_ValueError, "invalid slice!");
                        return NULL;
                    } else if (slicelength == 1){
                        return PyFloat_FromDouble(self->mat->data[index][start]);
                    } else {
                        allocate_matrix_ref(&(mat2->mat), self->mat, index, start, 1, slicelength);
                        mat2->shape = get_shape(1, slicelength);
                        return (PyObject*)mat2;
                    }     
                }              
            } else if (PySlice_Check(arg1) && PyLong_Check(arg2)) {
                int index = PyLong_AsLong(arg2);
                if (index >= size2 || index < 0) {
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return NULL;
                }
                PySlice_GetIndicesEx(arg1, size1, &start, &stop, &step, &slicelength);
                if (slicelength < 1 || step != 1) {
                    PyErr_SetString(PyExc_ValueError, "Invalid slice!");
                    return NULL;
                } else if (slicelength == 1) {
                    return PyFloat_FromDouble(self->mat->data[start][index]);
                } else {
                    allocate_matrix_ref(&(mat2->mat), self->mat, start, index, slicelength, 1);
                    mat2->shape = get_shape(slicelength, 1);
                    return (PyObject*)mat2;
                }
            } else if (PySlice_Check(arg1) && PySlice_Check(arg2)) {
                PySlice_GetIndicesEx(arg1, size1, &start, &stop, &step, &slicelength);
                Py_ssize_t start2;
                Py_ssize_t stop2;
                Py_ssize_t step2;
                Py_ssize_t slicelength2;
                PySlice_GetIndicesEx(arg2, size2, &start2, &stop2, &step2, &slicelength2);
                if (slicelength < 1 || step != 1 || slicelength2 < 1 || step2 != 1) {
                    PyErr_SetString(PyExc_ValueError, "Invalid slice!"); 
                    return NULL; 
                } else if (slicelength == 1 && slicelength2 == 1) {
                    return PyFloat_FromDouble(self->mat->data[start][start2]);
                } else {
                    allocate_matrix_ref(&(mat2->mat), self->mat, start, start2, slicelength, slicelength2);
                    mat2->shape = get_shape(slicelength, slicelength2);
                    return (PyObject*) mat2;
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "Invalid arguments for 2d matrix");
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments!");
            return NULL;
        }
    }
    return NULL;
}

int int_int(Matrix61c* self, PyObject *arg1, PyObject *arg2, PyObject *v) {
    int row = PyLong_AsLong(arg1);
    int col = PyLong_AsLong(arg2);
    if (row < 0 || row >= self->mat->rows || col < 0 || col >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    } 
    double val;
    if (PyFloat_Check(v)) {
        val = PyFloat_AsDouble(v);
    } else if (PyLong_Check(v)) {
        val = PyLong_AsLong(v);
    } else {
        PyErr_SetString(PyExc_TypeError, "v should be a float or an int");
        return -1;
    }
    self->mat->data[row][col] = val;
    return 0;
}

int int_slice(Matrix61c* self, PyObject *arg1, PyObject *arg2, PyObject *v) {
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t slicelength;
    int size1 = self->mat->rows;
    int size2 = self->mat->cols;
    int index = PyLong_AsLong(arg1);
    if (index >= size1 || index < 0) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    } 
    PySlice_GetIndicesEx(arg2, size2, &start, &stop, &step, &slicelength);
    if (slicelength < 1 || step != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid slice!");
        return -1;
    }
    if (slicelength == 1) {
        double val;
        if (PyFloat_Check(v)) {
            val = PyFloat_AsDouble(v);
        } else if (PyLong_Check(v)) {
            val = PyLong_AsLong(v);
        } else {
            PyErr_SetString(PyExc_TypeError, "v should be a float or an int");
            return -1;
        }
        self->mat->data[index][start] = val;
        return 0;
    }
    if (!PyList_Check(v)) {
        PyErr_SetString(PyExc_TypeError, "v should be a list");
        return -1;
    } 
    if (PyList_Size(v) != slicelength) {
        PyErr_SetString(PyExc_ValueError, "ths size of v does not match slicelength");
        return -1;
    }
    int j = 0;
    for (int i = start; i < stop; i++) {
        PyObject* item = PyList_GetItem(v, j);
        if (!PyFloat_Check(item) && !PyLong_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "Wrong value");
            return -1;
        }  
        self->mat->data[index][i] = PyFloat_AsDouble(item); 
        j++;
    }
    return 0;
}

int slice_int(Matrix61c* self, PyObject *arg1, PyObject *arg2, PyObject *v) {
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t slicelength;
    int size1 = self->mat->rows;
    int size2 = self->mat->cols;
    int index = PyLong_AsLong(arg2);
    if (index >= size2 || index < 0) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    }
    PySlice_GetIndicesEx(arg1, size1, &start, &stop, &step, &slicelength);
    if (slicelength < 1 || step != 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid slice!");
        return -1;
    } 
    if (slicelength == 1) {
        double val;
        if (PyFloat_Check(v)) {
            val = PyFloat_AsDouble(v);
        } else if (PyLong_Check(v)) {
            val = PyLong_AsLong(v);
        } else {
            PyErr_SetString(PyExc_TypeError, "v should be a float or an int");
            return -1;
        }
        self->mat->data[start][index] = val;
        return 0;
    }
    if (!PyList_Check(v)) {
        PyErr_SetString(PyExc_TypeError, "v should be a list");
        return -1;
    } 
    if (PyList_Size(v) != slicelength) {
        PyErr_SetString(PyExc_ValueError, "ths size of v does not match slicelength");
        return -1;
    }
    int j = 0;
    for (int i = start; i < stop; i++) {
        PyObject* item = PyList_GetItem(v, j);
        if (!PyFloat_Check(item) && !PyLong_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "Wrong value");
            return -1;
        }  
        self->mat->data[i][index] = PyFloat_AsDouble(item); 
        j++;
    }
    return 0;
}

int slice_slice(Matrix61c* self, PyObject *arg1, PyObject *arg2, PyObject *v) {
    int size1 = self->mat->rows;
    int size2 = self->mat->cols;
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t slicelength;
    PySlice_GetIndicesEx(arg1, size1, &start, &stop, &step, &slicelength);
    Py_ssize_t start2;
    Py_ssize_t stop2;
    Py_ssize_t step2;
    Py_ssize_t slicelength2;
    PySlice_GetIndicesEx(arg2, size2, &start2, &stop2, &step2, &slicelength2);
    if (slicelength < 1 || step != 1 || slicelength2 < 1 || step2 != 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid slice!");
        return -1;
    } 
    if (slicelength == 1 && slicelength2 == 1) {
        double val;
        if (PyFloat_Check(v)) {
            val = PyFloat_AsDouble(v);
        } else if (PyLong_Check(v)) {
            val = PyLong_AsLong(v);
        } else {
            PyErr_SetString(PyExc_TypeError, "v should be a float or an int");
            return -1;
        }
        self->mat->data[start][start2] = val;
        return 0;
    }
    if (!PyList_Check(v)) {
        PyErr_SetString(PyExc_TypeError, "v should be a list");
        return -1;
    }
    if (slicelength == 1) {
        if (PyList_Size(v) != slicelength2) {
            PyErr_SetString(PyExc_ValueError, "the size of v does not match slicelength");
            return -1;
        }
        int j = 0;
        for (int i = start2; i < stop2; i++) {
            PyObject* item = PyList_GetItem(v, j);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_ValueError, "wrong value");
                return -1;
            }
            self->mat->data[start][i] = PyFloat_AsDouble(item);
            j++;
        }
        return 0;
    }
    if (slicelength2 == 1) {
        if (PyList_Size(v) != slicelength) {
            PyErr_SetString(PyExc_ValueError, "the size of v does not match slicelength");
            return -1;
        }
        int j = 0;
        for (int i = start; i < stop; i++) {
            PyObject* item = PyList_GetItem(v, j);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_ValueError, "wrong value");
                return -1;
            }
            self->mat->data[i][start2] = PyFloat_AsDouble(item);
            j++;
        }
        return 0;
    }
    if (PyList_Size(v) != slicelength) {
        PyErr_SetString(PyExc_ValueError, "the size of v does not match slicelength");
        return -1;
    }
    int k = 0;
    for (int i = start; i < stop; i++) {
        PyObject* list = PyList_GetItem(v, k);
        if (!PyList_Check(list)) {
            PyErr_SetString(PyExc_TypeError, "v should be a list");
            return -1;
        }
        if (PyList_Size(v) != slicelength2) {
            PyErr_SetString(PyExc_ValueError, "the size of v does not match slicelength");
            return -1;
        }
        int s = 0;
        for (int j = start2; j < stop2; j++) {
            PyObject* item = PyList_GetItem(list, s);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_ValueError, "Wrong value");
                return -1;
            }  
            self->mat->data[i][j] = PyFloat_AsDouble(item); 
            s++;
        }
        k++;
    }
    return 0;
}
/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t slicelength;
    if (self->mat->is_1d) {
        int size = self->mat->rows * self->mat->cols;
        double val;
        if (PyLong_Check(key)) {
            if (PyFloat_Check(v)) {
                val = PyFloat_AsDouble(v);
            } else if (PyLong_Check(v)) {
                val = PyLong_AsLong(v);
            } else {
                PyErr_SetString(PyExc_TypeError, "v should be a float or an int");
                return -1;
            }
            int index = PyLong_AsLong(key);
            if (index >= size || index < 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of range!");
                return -1;
            } else {
                (self->mat->data_1[index]) = val;
                return 0;
            } 
        } else if (PySlice_Check(key)) {
            PySlice_GetIndicesEx(key, size , &start, &stop, &step, &slicelength);
            if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "invalid slice!");
                return -1;
            } else if (slicelength == 1) {
                double val;
                if (PyFloat_Check(v)) {
                    val = PyFloat_AsDouble(v);
                } else if (PyLong_Check(v)) {
                    val = PyLong_AsLong(v);
                } else {
                    PyErr_SetString(PyExc_TypeError, "v should be a float or an int");
                    return -1;
                }
                (self->mat->data_1[start]) = val;
                return 0;
            } else {
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "v should be a list");
                    return -1;
                } 
                if (slicelength != PyList_Size(v)) {
                    PyErr_SetString(PyExc_ValueError, "ths size of v does not match slicelength");
                    return -1;
                }
                int j = 0;
                for (int i = start; i < stop; i++) {
                    PyObject* item = PyList_GetItem(v, j);
                    if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                        PyErr_SetString(PyExc_ValueError, "Wrong value");
                        return -1;
                    }
                    (self->mat->data_1[i]) = PyFloat_AsDouble(item); 
                    j++;
                }
                return 0;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments for 1d matrix");
            return -1;
        }
    } else {
        int size = self->mat->rows;
        PyObject *arg1 = NULL;
        PyObject *arg2 = NULL;
        if (PyLong_Check(key)) {
            int index = PyLong_AsLong(key);
            if (index >= size || index < 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of range!");
                return -1;
            } 
            if (!PyList_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "v should be a list");
                return -1;
            }
            if (self->mat->cols != PyList_Size(v)) {
                PyErr_SetString(PyExc_ValueError, "ths size of v does not match length");
                return -1;
            }
            for (int i = 0; i < self->mat->cols; i++) {
                PyObject* item = PyList_GetItem(v, i);
                if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                    PyErr_SetString(PyExc_ValueError, "Wrong value");
                    return -1;
                }  
                self->mat->data[index][i] = PyFloat_AsDouble(item); 
            }
            return 0;
        } else if (PySlice_Check(key)) {
            PySlice_GetIndicesEx(key, size, &start, &stop, &step, &slicelength);
            if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "invalid slice!");
                return -1;
            } 
            if (!PyList_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "v should be a list");
                return -1;
            }
            if (slicelength == 1) {
                if (PyList_Size(v) != self->mat->cols) {
                    PyErr_SetString(PyExc_ValueError, "ths size of v should be a columnlength");
                    return -1;
                }
                for (int i = 0; i < self->mat->cols; i++) {
                    PyObject* item = PyList_GetItem(v, i);
                    if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                        PyErr_SetString(PyExc_ValueError, "value should be a float or an int");
                        return -1;
                    }
                    self->mat->data[start][i] = PyFloat_AsDouble(item);
                }
                return 0;
            }
            if (slicelength != PyList_Size(v)) {
                PyErr_SetString(PyExc_ValueError, "ths size of v does not match slicelength");
                return -1;
            }
            int k = 0;
            for (int i = start; i < stop; i++) {
                PyObject* list = PyList_GetItem(v, k);
                if (!PyList_Check(list)) {
                    PyErr_SetString(PyExc_TypeError, "elements of v should be a list");
                    return -1;
                }
                if (PyList_Size(list) != self->mat->cols) {
                    PyErr_SetString(PyExc_ValueError, "ths size of list does not match columnlength");
                    return -1;
                }
                for (int j = 0; j < self->mat->cols; j++) {
                    PyObject* item = PyList_GetItem(list, j);
                    if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                        PyErr_SetString(PyExc_ValueError, "Wrong value");
                        return -1;
                    }  
                    self->mat->data[i][j] = PyFloat_AsDouble(item); 
                }
                k++;   
            } 
        } else if (PyArg_UnpackTuple(key, "key", 2, 2, &arg1, &arg2)) {
            /*if arg1 and arg2 = integers*/
            if (PyLong_Check(arg1) && PyLong_Check(arg2)) {
                return int_int(self, arg1, arg2, v);
            } else if (PyLong_Check(arg1) && PySlice_Check(arg2)) {
                return int_slice(self, arg1, arg2, v);
            } else if (PySlice_Check(arg1) && PyLong_Check(arg2)) {
                return slice_int(self, arg1, arg2, v);
            } else if (PySlice_Check(arg1) && PySlice_Check(arg2)) {
                return slice_slice(self, arg1, arg2, v);
            }
        }
    }
    return 0;
}



PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}