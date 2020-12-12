#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        PyErr_SetString(PyExc_TypeError, "Invalid dimensions!");
        return -1;
    }
    *mat = malloc(sizeof(struct matrix));
    if (*mat == NULL) {
        return -1;
    }
    if (rows == 1 || cols == 1) {
        (*mat)->is_1d = 1;
    } else {
        (*mat)->is_1d = 0;
    }

    (*mat)->data = (double **) calloc(rows, sizeof(double *));
    if ((*mat)->data == NULL) {
        return -1;
    }

    (*mat)->data_1 = (double *) calloc(rows *cols, sizeof(double));
    if ((*mat)->data_1 == NULL) {
        free(mat);
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        (*mat)->data[i] = (*mat)->data_1 + (i * cols);
    }

    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    if (rows <= 0 || cols <= 0 || from == NULL) {
        return -1;
    }
    *mat = (matrix *) malloc(sizeof(matrix));
    if (*mat == NULL) {
        free(*mat);
        return -1;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    if (rows == 1 || cols == 1) {
        (*mat)->is_1d = 1;
    } else {
        (*mat)->is_1d = 0;
    }

    (*mat)->data = (double **) calloc(rows, sizeof(double *));
    if ((*mat)->data == NULL) {
        return -1;
    }

    (*mat)->data_1 = (double *) calloc(rows *cols, sizeof(double));
    if ((*mat)->data_1 == NULL) {
        return -1;
    }

    for (int i = 0; i < rows; i++) { 
        for (int j = 0; j < cols; j++) {
            (*mat)->data_1[i * cols + j] = from->data_1[(from->cols * (row_offset + i)) + col_offset + j];
        }
        (*mat)->data[i] = (*mat)->data_1 + (i * cols);
    }
    from->ref_cnt += 1;
    (*mat)->parent = from;
    (*mat)->ref_cnt = 1;
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL) {
       return;
     } else {
         if (mat->parent == NULL) {
           (mat->ref_cnt)--;
           if (mat->ref_cnt == 0) {
              free(mat->data_1);
             free(mat->data);
             free(mat);
           }
         } else {
           (mat->parent->ref_cnt)--;
           if(mat->parent->ref_cnt == 0) {
            free(mat->parent->data_1);
             free(mat->parent->data);
             free(mat->parent);            
           }
           free(mat);
         }
       }
    return;
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    int length = mat->rows * mat->cols;
    __m256d value = _mm256_set1_pd(val);
    #pragma omp parallel for
    for (int i = 0; i < length / 16 * 16; i+=16) {
      // double* mat_addr = mat->data + i;
      _mm256_storeu_pd(mat->data_1 + i, value);
      _mm256_storeu_pd(mat->data_1 + i+4, value);
      _mm256_storeu_pd(mat->data_1 + i+8, value);
      _mm256_storeu_pd(mat->data_1 + i+12, value);
    }
    for (int i = length / 16 * 16; i < length; i++) {
        mat->data_1[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (result->rows == mat1->rows && result->cols == mat1->cols && result->rows == mat2->rows && result->cols == mat2->cols)
    {
        int size = result->rows * result->cols;
        double *result_d = result->data_1;
        double *mat1_d = mat1->data_1;
        double *mat2_d = mat2->data_1;
        __m256d d1;
        __m256d d2;
        __m256d res1;
        __m256d res2;
        __m256d res3;
        __m256d res4;
        int unroll = 4;
        int stride = unroll * 4;
        omp_set_num_threads(8);
        int ompSize = size / stride * stride - stride;
#pragma omp parallel private(d1, d2, res1, res2, res3, res4)
        {
#pragma omp for
            for (int i = 0; i <= ompSize; i += stride)
            {
                d1 = _mm256_loadu_pd(&mat1_d[i]);
                d2 = _mm256_loadu_pd(&mat2_d[i]);
                res1 = _mm256_add_pd(d1, d2);

                d1 = _mm256_loadu_pd(&mat1_d[i + 4]);
                d2 = _mm256_loadu_pd(&mat2_d[i + 4]);
                res2 = _mm256_add_pd(d1, d2);

                d1 = _mm256_loadu_pd(&mat1_d[i + 8]);
                d2 = _mm256_loadu_pd(&mat2_d[i + 8]);
                res3 = _mm256_add_pd(d1, d2);

                d1 = _mm256_loadu_pd(&mat1_d[i + 12]);
                d2 = _mm256_loadu_pd(&mat2_d[i + 12]);
                res4 = _mm256_add_pd(d1, d2);

                _mm256_storeu_pd(&result_d[i], res1);
                _mm256_storeu_pd(&result_d[i + 4], res2);
                _mm256_storeu_pd(&result_d[i + 8], res3);
                _mm256_storeu_pd(&result_d[i + 12], res4);
            }
        }
        for (int k = size / stride * stride; k < size; ++k)
        {
            result_d[k] = mat1_d[k] + mat2_d[k];
        }
        return 0;
    }
    return -1;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != result->rows || mat2->cols != result-> cols || mat2->rows != result->rows || mat2->cols != result->cols || result->rows <= 0 || result->cols <= 0) {
      return -1;
    } else {
        int length = result->rows * result->cols;
        double* result_addr;
        double* mat1_addr;
        double* mat2_addr;
        __m256d res0;
        __m256d res1;
        __m256d res2;
        __m256d res3;
        #pragma omp parallel for private(res0, res1, res2, res3)
        for (int i = 0; i < length / 16 * 16; i+=16) {
            result_addr = result->data_1 + i;
            mat1_addr = mat1->data_1 + i;
            mat2_addr = mat2->data_1 + i;
            res0 = _mm256_sub_pd(_mm256_loadu_pd(mat1_addr), _mm256_loadu_pd(mat2_addr));
            res1 = _mm256_sub_pd(_mm256_loadu_pd(mat1_addr+4), _mm256_loadu_pd(mat2_addr+4));
            res2 = _mm256_sub_pd(_mm256_loadu_pd(mat1_addr+8), _mm256_loadu_pd(mat2_addr+8));
            res3 = _mm256_sub_pd(_mm256_loadu_pd(mat1_addr+12), _mm256_loadu_pd(mat2_addr+12));
            _mm256_storeu_pd(result_addr, res0);
            _mm256_storeu_pd(result_addr+4, res1);
            _mm256_storeu_pd(result_addr+8, res2);
            _mm256_storeu_pd(result_addr+12, res3);
        }
        for (int i = length / 16 * 16; i < length; i++) {
            result->data_1[i] = *(mat1->data_1 + i) - *(mat2->data_1 + i);
        }
        return 0;
    }
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != result->rows || mat2->cols != result->cols || mat1->cols != mat2->rows) {
     PyErr_SetString(PyExc_TypeError, "Dimensions do not match");
     return -1;
   } else {
     __m256d vects0[mat2->cols]; // 1D __m256d vector for storing
     __m256d vects1[mat2->cols];
     __m256d vects2[mat2->cols];
     __m256d vects3[mat2->cols];

     __m256d mat1_in0_k0, mat1_in1_k0, mat1_in2_k0, mat1_in3_k0;
     __m256d mat1_in0_k1, mat1_in1_k1, mat1_in2_k1, mat1_in3_k1;
     
     __m256d set0, set1, set2, set3;
     __m256d set4, set5, set6, set7;
     
     __m256d sum00, sum01, sum02, sum03, sum10, sum11, sum12, sum13, sum20, sum21, sum22, sum23, sum30, sum31, sum32, sum33;
     __m256d sum04, sum05, sum06, sum07, sum14, sum15, sum16, sum17, sum24, sum25, sum26, sum27, sum34, sum35, sum36, sum37;
     
     int i, j, k, l;
     
     __m256d tail_set0, tail_set1;
     
      __m256d vval00, vval01, vval02, vval03, vval10, vval11, vval12, vval13, vval20, vval21, vval22, vval23, vval30, vval31, vval32, vval33;
     double dval00, dval01, dval02, dval03, dval10, dval11, dval12, dval13, dval20, dval21, dval22, dval23, dval30, dval31, dval32, dval33;
          
     double tail_vals0[mat2->cols];
     double tail_vals1[mat2->cols];
     double tail_vals2[mat2->cols];
     double tail_vals3[mat2->cols];
     
     double* m1 = mat1->data_1;
     double* m2 = mat2->data_1;
     double* mr = result->data_1;
     int m1_cols = mat1->cols;
     int m2_cols = mat2->cols;
     int mr_cols = result->cols;
     
     #pragma omp parallel for private(i, j, k, l, vects0, vects1, vects2, vects3, tail_vals0, tail_vals1, tail_vals2, tail_vals3, mat1_in0_k0, mat1_in1_k0, mat1_in2_k0, mat1_in3_k0, mat1_in0_k1, mat1_in1_k1, mat1_in2_k1, mat1_in3_k1, set0, set1, set2, set3, set4, set5, set6, set7, tail_set0, tail_set1, sum00, sum01, sum02, sum03, sum10, sum11, sum12, sum13, sum20, sum21, sum22, sum23, sum30, sum31, sum32, sum33, sum04, sum05, sum06, sum07, sum14, sum15, sum16, sum17, sum24, sum25, sum26, sum27, sum34, sum35, sum36, sum37, vval00, vval01, vval02, vval03, vval10, vval11, vval12, vval13, vval20, vval21, vval22, vval23, vval30, vval31, vval32, vval33, dval00, dval01, dval02, dval03, dval10, dval11, dval12, dval13, dval20, dval21, dval22, dval23, dval30, dval31, dval32, dval33)
     for (i = 0; i < mat1->rows / 4 * 4; i+=4) {

       //initialize vectors[] to __m256(0) for each entry of vectors[] for each loop of i, same with dvals and tail_vals
       for (l = 0; l < m2_cols; l++) {
         vects0[l] = _mm256_setzero_pd();
         vects1[l] = _mm256_setzero_pd();
         vects2[l] = _mm256_setzero_pd();
         vects3[l] = _mm256_setzero_pd();
         tail_vals0[l] = 0;
         tail_vals1[l] = 0;
         tail_vals2[l] = 0;
         tail_vals3[l] = 0;
       }
      for (k = 0; k < m1_cols / 8 * 8; k+=8) {
        mat1_in0_k0 = _mm256_loadu_pd(m1 + (i * m1_cols) + k);
        mat1_in0_k1 = _mm256_loadu_pd(m1 + (i * m1_cols) + (k+4));
        mat1_in1_k0 = _mm256_loadu_pd(m1 + ((i+1) * m1_cols) + k);
        mat1_in1_k1 = _mm256_loadu_pd(m1 + ((i+1) * m1_cols) + (k+4));
        mat1_in2_k0 = _mm256_loadu_pd(m1 + ((i+2) * m1_cols) + k);
        mat1_in2_k1 = _mm256_loadu_pd(m1 + ((i+2) * m1_cols) + (k+4));
        mat1_in3_k0 = _mm256_loadu_pd(m1 + ((i+3) * m1_cols) + k);
        mat1_in3_k1 = _mm256_loadu_pd(m1 + ((i+3) * m1_cols) + (k+4));

         for (j = 0; j < m2_cols / 8 * 8; j+=8) {
           sum00 = vects0[j];
           sum01 = vects0[j+1];
           sum02 = vects0[j+2];
           sum03 = vects0[j+3];
           sum04 = vects0[j+4];
           sum05 = vects0[j+5];
           sum06 = vects0[j+6];
           sum07 = vects0[j+7];
          
           sum10 = vects1[j];
           sum11 = vects1[j+1];
           sum12 = vects1[j+2];
           sum13 = vects1[j+3];
           sum14 = vects1[j+4];
           sum15 = vects1[j+5];
           sum16 = vects1[j+6];
           sum17 = vects1[j+7];
           
           sum20 = vects2[j];
           sum21 = vects2[j+1];
           sum22 = vects2[j+2];
           sum23 = vects2[j+3];
           sum24 = vects2[j+4];
           sum25 = vects2[j+5];
           sum26 = vects2[j+6];
           sum27 = vects2[j+7];
           
           sum30 = vects3[j];
           sum31 = vects3[j+1];
           sum32 = vects3[j+2];
           sum33 = vects3[j+3];
           sum34 = vects3[j+4];
           sum35 = vects3[j+5];
           sum36 = vects3[j+6];
           sum37 = vects3[j+7];
           
           set0 = _mm256_set_pd(*(m2 + j + ((k+3) * m2_cols)), *(m2 + j + ((k+2) * m2_cols)), *(m2 + j + ((k+1) * m2_cols)), *(m2 + j + (k * m2_cols)));
           set1 = _mm256_set_pd(*(m2 + (j+1) + ((k+3) * m2_cols)), *(m2 + (j+1) + ((k+2) * m2_cols)), *(m2 + (j+1) + ((k+1) * m2_cols)), *(m2 + (j+1) + (k * m2_cols)));
           set2 = _mm256_set_pd(*(m2 + (j+2) + ((k+3) * m2_cols)), *(m2 + (j+2) + ((k+2) * m2_cols)), *(m2 + (j+2) + ((k+1) * m2_cols)), *(m2 + (j+2) + (k * m2_cols)));
           set3 = _mm256_set_pd(*(m2 + (j+3) + ((k+3) * m2_cols)), *(m2 + (j+3) + ((k+2) * m2_cols)), *(m2 + (j+3) + ((k+1) * m2_cols)), *(m2 + (j+3) + (k * m2_cols)));
           
           sum00 = _mm256_fmadd_pd(mat1_in0_k0, set0, sum00);
           sum01 = _mm256_fmadd_pd(mat1_in0_k0, set1, sum01);
           sum02 = _mm256_fmadd_pd(mat1_in0_k0, set2, sum02);
           sum03 = _mm256_fmadd_pd(mat1_in0_k0, set3, sum03);
           
           sum10 = _mm256_fmadd_pd(mat1_in1_k0, set0, sum10);
           sum11 = _mm256_fmadd_pd(mat1_in1_k0, set1, sum11);
           sum12 = _mm256_fmadd_pd(mat1_in1_k0, set2, sum12);
           sum13 = _mm256_fmadd_pd(mat1_in1_k0, set3, sum13);
           
           sum20 = _mm256_fmadd_pd(mat1_in2_k0, set0, sum20);
           sum21 = _mm256_fmadd_pd(mat1_in2_k0, set1, sum21);
           sum22 = _mm256_fmadd_pd(mat1_in2_k0, set2, sum22);
           sum23 = _mm256_fmadd_pd(mat1_in2_k0, set3, sum23);
           
           sum30 = _mm256_fmadd_pd(mat1_in3_k0, set0, sum30);
           sum31 = _mm256_fmadd_pd(mat1_in3_k0, set1, sum31);
           sum32 = _mm256_fmadd_pd(mat1_in3_k0, set2, sum32);
           sum33 = _mm256_fmadd_pd(mat1_in3_k0, set3, sum33);
           
           set0 = _mm256_set_pd(*(m2 + j + ((k+7) * m2_cols)), *(m2 + j + ((k+6) * m2_cols)), *(m2 + j + ((k+5) * m2_cols)), *(m2 + j + ((k+4) * m2_cols)));
           set1 = _mm256_set_pd(*(m2 + (j+1) + ((k+7) * m2_cols)), *(m2 + (j+1) + ((k+6) * m2_cols)), *(m2 + (j+1) + ((k+5) * m2_cols)), *(m2 + (j+1) + ((k+4) * m2_cols)));
           set2 = _mm256_set_pd(*(m2 + (j+2) + ((k+7) * m2_cols)), *(m2 + (j+2) + ((k+6) * m2_cols)), *(m2 + (j+2) + ((k+5) * m2_cols)), *(m2 + (j+2) + ((k+4) * m2_cols)));
           set3 = _mm256_set_pd(*(m2 + (j+3) + ((k+7) * m2_cols)), *(m2 + (j+3) + ((k+6) * m2_cols)), *(m2 + (j+3) + ((k+5) * m2_cols)), *(m2 + (j+3) + ((k+4) * m2_cols)));
           
           vects0[j] = _mm256_fmadd_pd(mat1_in0_k1, set0, sum00);
           vects0[j+1] = _mm256_fmadd_pd(mat1_in0_k1, set1, sum01);
           vects0[j+2] = _mm256_fmadd_pd(mat1_in0_k1, set2, sum02);
           vects0[j+3] = _mm256_fmadd_pd(mat1_in0_k1, set3, sum03);
           
           vects1[j] = _mm256_fmadd_pd(mat1_in1_k1, set0, sum10);
           vects1[j+1] = _mm256_fmadd_pd(mat1_in1_k1, set1, sum11);
           vects1[j+2] = _mm256_fmadd_pd(mat1_in1_k1, set2, sum12);
           vects1[j+3] = _mm256_fmadd_pd(mat1_in1_k1, set3, sum13);
           
           vects2[j] = _mm256_fmadd_pd(mat1_in2_k1, set0, sum20);
           vects2[j+1] = _mm256_fmadd_pd(mat1_in2_k1, set1, sum21);
           vects2[j+2] = _mm256_fmadd_pd(mat1_in2_k1, set2, sum22);
           vects2[j+3] = _mm256_fmadd_pd(mat1_in2_k1, set3, sum23);
           
           vects3[j] = _mm256_fmadd_pd(mat1_in3_k1, set0, sum30);
           vects3[j+1] = _mm256_fmadd_pd(mat1_in3_k1, set1, sum31);
           vects3[j+2] = _mm256_fmadd_pd(mat1_in3_k1, set2, sum32);
           vects3[j+3] = _mm256_fmadd_pd(mat1_in3_k1, set3, sum33);
           
           set4 = _mm256_set_pd(*(m2 + (j+4) + ((k+3) * m2_cols)), *(m2 + (j+4) + ((k+2) * m2_cols)), *(m2 + (j+4) + ((k+1) * m2_cols)), *(m2 + (j+4) + (k * m2_cols)));
           set5 = _mm256_set_pd(*(m2 + (j+5) + ((k+3) * m2_cols)), *(m2 + (j+5) + ((k+2) * m2_cols)), *(m2 + (j+5) + ((k+1) * m2_cols)), *(m2 + (j+5) + (k * m2_cols)));
           set6 = _mm256_set_pd(*(m2 + (j+6) + ((k+3) * m2_cols)), *(m2 + (j+6) + ((k+2) * m2_cols)), *(m2 + (j+6) + ((k+1) * m2_cols)), *(m2 + (j+6) + (k * m2_cols)));
           set7 = _mm256_set_pd(*(m2 + (j+7) + ((k+3) * m2_cols)), *(m2 + (j+7) + ((k+2) * m2_cols)), *(m2 + (j+7) + ((k+1) * m2_cols)), *(m2 + (j+7) + (k * m2_cols)));
           
           sum04 = _mm256_fmadd_pd(mat1_in0_k0, set4, sum04);
           sum05 = _mm256_fmadd_pd(mat1_in0_k0, set5, sum05);
           sum06 = _mm256_fmadd_pd(mat1_in0_k0, set6, sum06);
           sum07 = _mm256_fmadd_pd(mat1_in0_k0, set7, sum07);
           
           sum14 = _mm256_fmadd_pd(mat1_in1_k0, set4, sum14);
           sum15 = _mm256_fmadd_pd(mat1_in1_k0, set5, sum15);
           sum16 = _mm256_fmadd_pd(mat1_in1_k0, set6, sum16);
           sum17 = _mm256_fmadd_pd(mat1_in1_k0, set7, sum17);
           
           sum24 = _mm256_fmadd_pd(mat1_in2_k0, set4, sum24);
           sum25 = _mm256_fmadd_pd(mat1_in2_k0, set5, sum25);
           sum26 = _mm256_fmadd_pd(mat1_in2_k0, set6, sum26);
           sum27 = _mm256_fmadd_pd(mat1_in2_k0, set7, sum27);
           
           sum34 = _mm256_fmadd_pd(mat1_in3_k0, set4, sum34);
           sum35 = _mm256_fmadd_pd(mat1_in3_k0, set5, sum35);
           sum36 = _mm256_fmadd_pd(mat1_in3_k0, set6, sum36);
           sum37 = _mm256_fmadd_pd(mat1_in3_k0, set7, sum37);
           
           set4 = _mm256_set_pd(*(m2 + (j+4) + ((k+7) * m2_cols)), *(m2 + (j+4) + ((k+6) * m2_cols)), *(m2 + (j+4) + ((k+5) * m2_cols)), *(m2 + (j+4) + ((k+4) * m2_cols)));
           set5 = _mm256_set_pd(*(m2 + (j+5) + ((k+7) * m2_cols)), *(m2 + (j+5) + ((k+6) * m2_cols)), *(m2 + (j+5) + ((k+5) * m2_cols)), *(m2 + (j+5) + ((k+4) * m2_cols)));
           set6 = _mm256_set_pd(*(m2 + (j+6) + ((k+7) * m2_cols)), *(m2 + (j+6) + ((k+6) * m2_cols)), *(m2 + (j+6) + ((k+5) * m2_cols)), *(m2 + (j+6) + ((k+4) * m2_cols)));
           set7 = _mm256_set_pd(*(m2 + (j+7) + ((k+7) * m2_cols)), *(m2 + (j+7) + ((k+6) * m2_cols)), *(m2 + (j+7) + ((k+5) * m2_cols)), *(m2 + (j+7) + ((k+4) * m2_cols)));
          
           vects0[j+4] = _mm256_fmadd_pd(mat1_in0_k1, set4, sum04);
           vects0[j+5] = _mm256_fmadd_pd(mat1_in0_k1, set5, sum05);
           vects0[j+6] = _mm256_fmadd_pd(mat1_in0_k1, set6, sum06);
           vects0[j+7] = _mm256_fmadd_pd(mat1_in0_k1, set7, sum07);
           
           vects1[j+4] = _mm256_fmadd_pd(mat1_in1_k1, set4, sum14);
           vects1[j+5] = _mm256_fmadd_pd(mat1_in1_k1, set5, sum15);
           vects1[j+6] = _mm256_fmadd_pd(mat1_in1_k1, set6, sum16);
           vects1[j+7] = _mm256_fmadd_pd(mat1_in1_k1, set7, sum17);
           
           vects2[j+4] = _mm256_fmadd_pd(mat1_in2_k1, set4, sum24);
           vects2[j+5] = _mm256_fmadd_pd(mat1_in2_k1, set5, sum25);
           vects2[j+6] = _mm256_fmadd_pd(mat1_in2_k1, set6, sum26);
           vects2[j+7] = _mm256_fmadd_pd(mat1_in2_k1, set7, sum27);
           
           vects3[j+4] = _mm256_fmadd_pd(mat1_in3_k1, set4, sum34);
           vects3[j+5] = _mm256_fmadd_pd(mat1_in3_k1, set5, sum35);
           vects3[j+6] = _mm256_fmadd_pd(mat1_in3_k1, set6, sum36);
           vects3[j+7] = _mm256_fmadd_pd(mat1_in3_k1, set7, sum37);
         }
     
         /* j LOOP TAIL CASE */
         for (j = m2_cols / 8 * 8; j < m2_cols; j++) {
           tail_set0 = _mm256_set_pd(*(m2 + j + ((k+3) * m2_cols)), *(m2 + j + ((k+2) * m2_cols)), *(m2 + j + ((k+1) * m2_cols)), *(m2 + j + (k * m2_cols)));
           tail_set1 = _mm256_set_pd(*(m2 + j + ((k+7) * m2_cols)), *(m2 + j + ((k+6) * m2_cols)), *(m2 + j + ((k+5) * m2_cols)), *(m2 + j + ((k+4) * m2_cols)));

           vects0[j] = _mm256_fmadd_pd(mat1_in0_k0, tail_set0, vects0[j]);
           vects1[j] = _mm256_fmadd_pd(mat1_in1_k0, tail_set0, vects1[j]);
           vects2[j] = _mm256_fmadd_pd(mat1_in2_k0, tail_set0, vects2[j]);
           vects3[j] = _mm256_fmadd_pd(mat1_in3_k0, tail_set0, vects3[j]);

           vects0[j] = _mm256_fmadd_pd(mat1_in0_k1, tail_set1, vects0[j]);
           vects1[j] = _mm256_fmadd_pd(mat1_in1_k1, tail_set1, vects1[j]);
           vects2[j] = _mm256_fmadd_pd(mat1_in2_k1, tail_set1, vects2[j]);
           vects3[j] = _mm256_fmadd_pd(mat1_in3_k1, tail_set1, vects3[j]);
         }
       }
     
       /* k LOOP TAIL CASE */
       for (k = m1_cols / 8 * 8; k < m1_cols; k++) { //m1_cols == mat2->rows FYI
         for (j = 0; j < m2_cols; j++) {
           double tail_val0 = 0;
           tail_val0 = *(m1 + (i * m1_cols) + k) * *(m2 + j + (k * m2_cols));
           tail_vals0[j] += tail_val0;
           
           double tail_val1 = 0;
           tail_val1 = *(m1 + ((i+1) * m1_cols) + k) * *(m2 + j + (k * m2_cols));
           tail_vals1[j] += tail_val1;
           
           double tail_val2 = 0;
           tail_val2 = *(m1 + ((i+2) * m1_cols) + k) * *(m2 + j + (k * m2_cols));
           tail_vals2[j] += tail_val2;
           
           double tail_val3 = 0;
           tail_val3 = *(m1 + ((i+3) * m1_cols) + k) * *(m2 + j + (k * m2_cols));
           tail_vals3[j] += tail_val3;
         } 
       }
     
       /* adds each __m256d vector into dval and then stores */
       for (l = 0; l < m2_cols / 4 * 4; l+=4) {         
         dval00 = 0;
         double arr00[4];
         vval00 = vects0[l];
       
         dval01 = 0;
         double arr01[4];
         vval01 = vects0[l+1];
       
         dval02 = 0;
         double arr02[4];
         vval02 = vects0[l+2];
       
         dval03 = 0;
         double arr03[4];
         vval03 = vects0[l+3];
       
         dval10 = 0;
         double arr10[4];
         vval10 = vects1[l];
       
         dval11 = 0;
         double arr11[4];
         vval11 = vects1[l+1];
       
         dval12 = 0;
         double arr12[4];
         vval12 = vects1[l+2];
       
         dval13 = 0;
         double arr13[4];
         vval13 = vects1[l+3];
         
         dval20 = 0;
         double arr20[4];
         vval20 = vects2[l];
       
         dval21 = 0;
         double arr21[4];
         vval21 = vects2[l+1];
       
         dval22 = 0;
         double arr22[4];
         vval22 = vects2[l+2];
       
         dval23 = 0;
         double arr23[4];
         vval23 = vects2[l+3];
         
         dval30 = 0;
         double arr30[4];
         vval30 = vects3[l];
       
         dval31 = 0;
         double arr31[4];
         vval31 = vects3[l+1];
       
         dval32 = 0;
         double arr32[4];
         vval32 = vects3[l+2];
       
         dval33 = 0;
         double arr33[4];
         vval33 = vects3[l+3];
       
         _mm256_storeu_pd(arr00, vval00);
         _mm256_storeu_pd(arr01, vval01);
         _mm256_storeu_pd(arr02, vval02);
         _mm256_storeu_pd(arr03, vval03);
         _mm256_storeu_pd(arr10, vval10);
         _mm256_storeu_pd(arr11, vval11);
         _mm256_storeu_pd(arr12, vval12);
         _mm256_storeu_pd(arr13, vval13);
         _mm256_storeu_pd(arr20, vval20);
         _mm256_storeu_pd(arr21, vval21);
         _mm256_storeu_pd(arr22, vval22);
         _mm256_storeu_pd(arr23, vval23);
         _mm256_storeu_pd(arr30, vval30);
         _mm256_storeu_pd(arr31, vval31);
         _mm256_storeu_pd(arr32, vval32);
         _mm256_storeu_pd(arr33, vval33);
         
         for (int m = 0; m < 4; m++) {
           dval00 += arr00[m];
           dval01 += arr01[m];
           dval02 += arr02[m];
           dval03 += arr03[m];
           dval10 += arr10[m];
           dval11 += arr11[m];
           dval12 += arr12[m];
           dval13 += arr13[m];
           dval20 += arr20[m];
           dval21 += arr21[m];
           dval22 += arr22[m];
           dval23 += arr23[m];
           dval30 += arr30[m];
           dval31 += arr31[m];
           dval32 += arr32[m];
           dval33 += arr33[m];
         }
         *(mr + (i * mr_cols) + l) = dval00 + tail_vals0[l];
         *(mr + (i * mr_cols) + (l+1)) = dval01 + tail_vals0[l+1];
         *(mr + (i * mr_cols) + (l+2)) = dval02 + tail_vals0[l+2];
         *(mr + (i * mr_cols) + (l+3)) = dval03 + tail_vals0[l+3];
         
         *(mr + ((i+1) * mr_cols) + l) = dval10 + tail_vals1[l];
         *(mr + ((i+1) * mr_cols) + (l+1)) = dval11 + tail_vals1[l+1];
         *(mr + ((i+1) * mr_cols) + (l+2)) = dval12 + tail_vals1[l+2];
         *(mr + ((i+1) * mr_cols) + (l+3)) = dval13 + tail_vals1[l+3];
         
         *(mr + ((i+2) * mr_cols) + l) = dval20 + tail_vals2[l];
         *(mr + ((i+2) * mr_cols) + (l+1)) = dval21 + tail_vals2[l+1];
         *(mr + ((i+2) * mr_cols) + (l+2)) = dval22 + tail_vals2[l+2];
         *(mr + ((i+2) * mr_cols) + (l+3)) = dval23 + tail_vals2[l+3];
         
         *(mr + ((i+3) * mr_cols) + l) = dval30 + tail_vals3[l];
         *(mr + ((i+3) * mr_cols) + (l+1)) = dval31 + tail_vals3[l+1];
         *(mr + ((i+3) * mr_cols) + (l+2)) = dval32 + tail_vals3[l+2];
         *(mr + ((i+3) * mr_cols) + (l+3)) = dval33 + tail_vals3[l+3];
       }
       
       /* TAIL CASE FOR STORING INTO ARRAY */
       for (l = m2_cols / 4 * 4; l < m2_cols; l++) {
       // for (l = 0; l < mat2->cols; l++) {         
         double dval0 = 0;
         double arr0[4];
         __m256d vval0 = vects0[l];
         
         double dval1 = 0;
         double arr1[4];
         __m256d vval1 = vects1[l];
         
         double dval2 = 0;
         double arr2[4];
         __m256d vval2 = vects2[l];
         
         double dval3 = 0;
         double arr3[4];
         __m256d vval3 = vects3[l];
         
         _mm256_storeu_pd(arr0, vval0);
         _mm256_storeu_pd(arr1, vval1);
         _mm256_storeu_pd(arr2, vval2);
         _mm256_storeu_pd(arr3, vval3);
         
         for (int m = 0; m < 4; m++) {
           dval0 += arr0[m];
           dval1 += arr1[m];
           dval2 += arr2[m];
           dval3 += arr3[m];
         }
         *(mr + (i * mr_cols) + l) = dval0 + tail_vals0[l];
         *(mr + ((i+1) * mr_cols) + l) = dval1 + tail_vals1[l];
         *(mr + ((i+2) * mr_cols) + l) = dval2 + tail_vals2[l];
         *(mr + ((i+3) * mr_cols) + l) = dval3 + tail_vals3[l];
       }
     }
   
   /* i LOOP TAIL CASE */
     for (i = mat1->rows / 4 * 4; i < mat1->rows; i++) {
       for (j = 0; j < mat2->cols; j++) {
         double value = 0;
         for (k = 0; k < mat1->cols; k++) { //mat1->cols == mat2->rows FYI
           value += *(mat1->data_1 + (i * mat1->cols) + k) * *(mat2->data_1 + j + (k * mat2->cols));
         }
         *(result->data_1 + (i * result->cols) + j) = value;
       }
     }
   }
   return 0;
}

int mul_matrix_set_unrolled(matrix *result, matrix *mat1, matrix *mat2) { 
   /* TODO: YOUR CODE HERE */
   if (mat1->rows != result->rows || mat2->cols != result->cols || mat1->cols != mat2->rows) {
      PyErr_SetString(PyExc_TypeError, "Dimensions do not match");
      return -1;
   } else {
       __m256d vects[mat2->cols]; // 1D __m256d vector for storing
       __m256d mat1_in_k0, mat1_in_k1, mat1_in_k2;
       
       __m256d set0, set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11;
       
       __m256d sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11;
       
       int i, j, k, l;
       
       __m256d tail_set0, tail_set1, tail_set2;
       
        __m256d vval0, vval1, vval2, vval3, vval4, vval5, vval6, vval7, vval8, vval9, vval10, vval11;
       double dval0, dval1, dval2, dval3, dval4, dval5, dval6, dval7, dval8, dval9, dval10, dval11;
       
       // double arr0[4], arr1[4], arr2[4], arr3[4];
       
       double tail_vals[mat2->cols];
       
       double* m1 = mat1->data_1;
       double* m2 = mat2->data_1;
       double* mr = result->data_1;
       int m1_cols = mat1->cols;
       int m2_cols = mat2->cols;
       int mr_cols = result->cols;
     
      #pragma omp parallel for private(i, j, k, l, mat1_in_k0, mat1_in_k1, mat1_in_k2, vects, tail_vals, set0, set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, tail_set0, tail_set1, tail_set2, vval0, vval1, vval2, vval3, vval4, vval5, vval6, vval7, vval8, vval9, vval10, vval11, dval0, dval1, dval2, dval3, dval4, dval5, dval6, dval7, dval8, dval9, dval10, dval11)
      for (i = 0; i < mat1->rows; i++) {

         //initialize vectors[] to __m256(0) for each entry of vectors[] for each loop of i, same with dvals and tail_vals
        for (l = 0; l < m2_cols; l++) {
          vects[l] = _mm256_setzero_pd();
          tail_vals[l] = 0;
        }
        for (k = 0; k < m1_cols / 12 * 12; k+=12) {
          mat1_in_k0 = _mm256_loadu_pd(m1 + (i * m1_cols) + k);
          mat1_in_k1 = _mm256_loadu_pd(m1 + (i * m1_cols) + k+4);
          mat1_in_k2 = _mm256_loadu_pd(m1 + (i * m1_cols) + k+8);

          for (j = 0; j < m2_cols / 12 * 12; j+=12) {
             sum0 = vects[j];
             sum1 = vects[j+1];
             sum2 = vects[j+2];
             sum3 = vects[j+3];
             sum4 = vects[j+4];
             sum5 = vects[j+5];
             sum6 = vects[j+6];
             sum7 = vects[j+7];
             sum8 = vects[j+8];
             sum9 = vects[j+9];
             sum10 = vects[j+10];
             sum11 = vects[j+11];
              
             set0 = _mm256_set_pd(*(m2 + j + ((k+3) * m2_cols)), *(m2 + j + ((k+2) * m2_cols)), *(m2 + j + ((k+1) * m2_cols)), *(m2 + j + (k * m2_cols)));
             set1 = _mm256_set_pd(*(m2 + (j+1) + ((k+3) * m2_cols)), *(m2 + (j+1) + ((k+2) * m2_cols)), *(m2 + (j+1) + ((k+1) * m2_cols)), *(m2 + (j+1) + (k * m2_cols)));
             set2 = _mm256_set_pd(*(m2 + (j+2) + ((k+3) * m2_cols)), *(m2 + (j+2) + ((k+2) * m2_cols)), *(m2 + (j+2) + ((k+1) * m2_cols)), *(m2 + (j+2) + (k * m2_cols)));
             set3 = _mm256_set_pd(*(m2 + (j+3) + ((k+3) * m2_cols)), *(m2 + (j+3) + ((k+2) * m2_cols)), *(m2 + (j+3) + ((k+1) * m2_cols)), *(m2 + (j+3) + (k * m2_cols)));
             
             sum0 = _mm256_fmadd_pd(mat1_in_k0, set0, sum0);
             sum1 = _mm256_fmadd_pd(mat1_in_k0, set1, sum1);
             sum2 = _mm256_fmadd_pd(mat1_in_k0, set2, sum2);
             sum3 = _mm256_fmadd_pd(mat1_in_k0, set3, sum3);
             
             set0 = _mm256_set_pd(*(m2 + j + ((k+7) * m2_cols)), *(m2 + j + ((k+6) * m2_cols)), *(m2 + j + ((k+5) * m2_cols)), *(m2 + j + ((k+4) * m2_cols)));
             set1 = _mm256_set_pd(*(m2 + (j+1) + ((k+7) * m2_cols)), *(m2 + (j+1) + ((k+6) * m2_cols)), *(m2 + (j+1) + ((k+5) * m2_cols)), *(m2 + (j+1) + ((k+4) * m2_cols)));
             set2 = _mm256_set_pd(*(m2 + (j+2) + ((k+7) * m2_cols)), *(m2 + (j+2) + ((k+6) * m2_cols)), *(m2 + (j+2) + ((k+5) * m2_cols)), *(m2 + (j+2) + ((k+4) * m2_cols)));
             set3 = _mm256_set_pd(*(m2 + (j+3) + ((k+7) * m2_cols)), *(m2 + (j+3) + ((k+6) * m2_cols)), *(m2 + (j+3) + ((k+5) * m2_cols)), *(m2 + (j+3) + ((k+4) * m2_cols)));
             
             sum0 = _mm256_fmadd_pd(mat1_in_k1, set0, sum0);
             sum1 = _mm256_fmadd_pd(mat1_in_k1, set1, sum1);
             sum2 = _mm256_fmadd_pd(mat1_in_k1, set2, sum2);
             sum3 = _mm256_fmadd_pd(mat1_in_k1, set3, sum3);
             
             set0 = _mm256_set_pd(*(m2 + j + ((k+11) * m2_cols)), *(m2 + j + ((k+10) * m2_cols)), *(m2 + j + ((k+9) * m2_cols)), *(m2 + j + ((k+8) * m2_cols)));
             set1 = _mm256_set_pd(*(m2 + (j+1) + ((k+11) * m2_cols)), *(m2 + (j+1) + ((k+10) * m2_cols)), *(m2 + (j+1) + ((k+9) * m2_cols)), *(m2 + (j+1) + ((k+8) * m2_cols)));
             set2 = _mm256_set_pd(*(m2 + (j+2) + ((k+11) * m2_cols)), *(m2 + (j+2) + ((k+10) * m2_cols)), *(m2 + (j+2) + ((k+9) * m2_cols)), *(m2 + (j+2) + ((k+8) * m2_cols)));
             set3 = _mm256_set_pd(*(m2 + (j+3) + ((k+11) * m2_cols)), *(m2 + (j+3) + ((k+10) * m2_cols)), *(m2 + (j+3) + ((k+9) * m2_cols)), *(m2 + (j+3) + ((k+8) * m2_cols)));
             
             vects[j] = _mm256_fmadd_pd(mat1_in_k2, set0, sum0);
             vects[j+1] = _mm256_fmadd_pd(mat1_in_k2, set1, sum1);
             vects[j+2] = _mm256_fmadd_pd(mat1_in_k2, set2, sum2);
             vects[j+3] = _mm256_fmadd_pd(mat1_in_k2, set3, sum3);
             
             set4 = _mm256_set_pd(*(m2 + (j+4) + ((k+3) * m2_cols)), *(m2 + (j+4) + ((k+2) * m2_cols)), *(m2 + (j+4) + ((k+1) * m2_cols)), *(m2 + (j+4) + (k * m2_cols)));
             set5 = _mm256_set_pd(*(m2 + (j+5) + ((k+3) * m2_cols)), *(m2 + (j+5) + ((k+2) * m2_cols)), *(m2 + (j+5) + ((k+1) * m2_cols)), *(m2 + (j+5) + (k * m2_cols)));
             set6 = _mm256_set_pd(*(m2 + (j+6) + ((k+3) * m2_cols)), *(m2 + (j+6) + ((k+2) * m2_cols)), *(m2 + (j+6) + ((k+1) * m2_cols)), *(m2 + (j+6) + (k * m2_cols)));
             set7 = _mm256_set_pd(*(m2 + (j+7) + ((k+3) * m2_cols)), *(m2 + (j+7) + ((k+2) * m2_cols)), *(m2 + (j+7) + ((k+1) * m2_cols)), *(m2 + (j+7) + (k * m2_cols)));
             
             sum4 = _mm256_fmadd_pd(mat1_in_k0, set4, sum4);
             sum5 = _mm256_fmadd_pd(mat1_in_k0, set5, sum5);
             sum6 = _mm256_fmadd_pd(mat1_in_k0, set6, sum6);
             sum7 = _mm256_fmadd_pd(mat1_in_k0, set7, sum7);
             
             set4 = _mm256_set_pd(*(m2 + (j+4) + ((k+7) * m2_cols)), *(m2 + (j+4) + ((k+6) * m2_cols)), *(m2 + (j+4) + ((k+5) * m2_cols)), *(m2 + (j+4) + ((k+4) * m2_cols)));
             set5 = _mm256_set_pd(*(m2 + (j+5) + ((k+7) * m2_cols)), *(m2 + (j+5) + ((k+6) * m2_cols)), *(m2 + (j+5) + ((k+5) * m2_cols)), *(m2 + (j+5) + ((k+4) * m2_cols)));
             set6 = _mm256_set_pd(*(m2 + (j+6) + ((k+7) * m2_cols)), *(m2 + (j+6) + ((k+6) * m2_cols)), *(m2 + (j+6) + ((k+5) * m2_cols)), *(m2 + (j+6) + ((k+4) * m2_cols)));
             set7 = _mm256_set_pd(*(m2 + (j+7) + ((k+7) * m2_cols)), *(m2 + (j+7) + ((k+6) * m2_cols)), *(m2 + (j+7) + ((k+5) * m2_cols)), *(m2 + (j+7) + ((k+4) * m2_cols)));
             
             sum4 = _mm256_fmadd_pd(mat1_in_k1, set4, sum4);
             sum5 = _mm256_fmadd_pd(mat1_in_k1, set5, sum5);
             sum6 = _mm256_fmadd_pd(mat1_in_k1, set6, sum6);
             sum7 = _mm256_fmadd_pd(mat1_in_k1, set7, sum7);
             
             set4 = _mm256_set_pd(*(m2 + (j+4) + ((k+11) * m2_cols)), *(m2 + (j+4) + ((k+10) * m2_cols)), *(m2 + (j+4) + ((k+9) * m2_cols)), *(m2 + (j+4) + ((k+8) * m2_cols)));
             set5 = _mm256_set_pd(*(m2 + (j+5) + ((k+11) * m2_cols)), *(m2 + (j+5) + ((k+10) * m2_cols)), *(m2 + (j+5) + ((k+9) * m2_cols)), *(m2 + (j+5) + ((k+8) * m2_cols)));
             set6 = _mm256_set_pd(*(m2 + (j+6) + ((k+11) * m2_cols)), *(m2 + (j+6) + ((k+10) * m2_cols)), *(m2 + (j+6) + ((k+9) * m2_cols)), *(m2 + (j+6) + ((k+8) * m2_cols)));
             set7 = _mm256_set_pd(*(m2 + (j+7) + ((k+11) * m2_cols)), *(m2 + (j+7) + ((k+10) * m2_cols)), *(m2 + (j+7) + ((k+9) * m2_cols)), *(m2 + (j+7) + ((k+8) * m2_cols)));
             
             vects[j+4] = _mm256_fmadd_pd(mat1_in_k2, set4, sum4);
             vects[j+5] = _mm256_fmadd_pd(mat1_in_k2, set5, sum5);
             vects[j+6] = _mm256_fmadd_pd(mat1_in_k2, set6, sum6);
             vects[j+7] = _mm256_fmadd_pd(mat1_in_k2, set7, sum7);
             
             set8 = _mm256_set_pd(*(m2 + (j+8) + ((k+3) * m2_cols)), *(m2 + (j+8) + ((k+2) * m2_cols)), *(m2 + (j+8) + ((k+1) * m2_cols)), *(m2 + (j+8) + (k * m2_cols)));
             set9 = _mm256_set_pd(*(m2 + (j+9) + ((k+3) * m2_cols)), *(m2 + (j+9) + ((k+2) * m2_cols)), *(m2 + (j+9) + ((k+1) * m2_cols)), *(m2 + (j+9) + (k * m2_cols)));
             set10 = _mm256_set_pd(*(m2 + (j+10) + ((k+3) * m2_cols)), *(m2 + (j+10) + ((k+2) * m2_cols)), *(m2 + (j+10) + ((k+1) * m2_cols)), *(m2 + (j+10) + (k * m2_cols)));
             set11 = _mm256_set_pd(*(m2 + (j+11) + ((k+3) * m2_cols)), *(m2 + (j+11) + ((k+2) * m2_cols)), *(m2 + (j+11) + ((k+1) * m2_cols)), *(m2 + (j+11) + (k * m2_cols)));
             
             sum8 = _mm256_fmadd_pd(mat1_in_k0, set8, sum8);
             sum9 = _mm256_fmadd_pd(mat1_in_k0, set9, sum9);
             sum10 = _mm256_fmadd_pd(mat1_in_k0, set10, sum10);
             sum11 = _mm256_fmadd_pd(mat1_in_k0, set11, sum11);
             
             set8 = _mm256_set_pd(*(m2 + (j+8) + ((k+7) * m2_cols)), *(m2 + (j+8) + ((k+6) * m2_cols)), *(m2 + (j+8) + ((k+5) * m2_cols)), *(m2 + (j+8) + ((k+4) * m2_cols)));
             set9 = _mm256_set_pd(*(m2 + (j+9) + ((k+7) * m2_cols)), *(m2 + (j+9) + ((k+6) * m2_cols)), *(m2 + (j+9) + ((k+5) * m2_cols)), *(m2 + (j+9) + ((k+4) * m2_cols)));
             set10 = _mm256_set_pd(*(m2 + (j+10) + ((k+7) * m2_cols)), *(m2 + (j+10) + ((k+6) * m2_cols)), *(m2 + (j+10) + ((k+5) * m2_cols)), *(m2 + (j+10) + ((k+4) * m2_cols)));
             set11 = _mm256_set_pd(*(m2 + (j+11) + ((k+7) * m2_cols)), *(m2 + (j+11) + ((k+6) * m2_cols)), *(m2 + (j+11) + ((k+5) * m2_cols)), *(m2 + (j+11) + ((k+4) * m2_cols)));
             
             sum8 = _mm256_fmadd_pd(mat1_in_k1, set8, sum8);
             sum9 = _mm256_fmadd_pd(mat1_in_k1, set9, sum9);
             sum10 = _mm256_fmadd_pd(mat1_in_k1, set10, sum10);
             sum11 = _mm256_fmadd_pd(mat1_in_k1, set11, sum11);
             
             set8 = _mm256_set_pd(*(m2 + (j+8) + ((k+11) * m2_cols)), *(m2 + (j+8) + ((k+10) * m2_cols)), *(m2 + (j+8) + ((k+9) * m2_cols)), *(m2 + (j+8) + ((k+8) * m2_cols)));
             set9 = _mm256_set_pd(*(m2 + (j+9) + ((k+11) * m2_cols)), *(m2 + (j+9) + ((k+10) * m2_cols)), *(m2 + (j+9) + ((k+9) * m2_cols)), *(m2 + (j+9) + ((k+8) * m2_cols)));
             set10 = _mm256_set_pd(*(m2 + (j+10) + ((k+11) * m2_cols)), *(m2 + (j+10) + ((k+10) * m2_cols)), *(m2 + (j+10) + ((k+9) * m2_cols)), *(m2 + (j+10) + ((k+8) * m2_cols)));
             set11 = _mm256_set_pd(*(m2 + (j+11) + ((k+11) * m2_cols)), *(m2 + (j+11) + ((k+10) * m2_cols)), *(m2 + (j+11) + ((k+9) * m2_cols)), *(m2 + (j+11) + ((k+8) * m2_cols)));
             
             vects[j+8] = _mm256_fmadd_pd(mat1_in_k2, set8, sum8);
             vects[j+9] = _mm256_fmadd_pd(mat1_in_k2, set9, sum9);
             vects[j+10] = _mm256_fmadd_pd(mat1_in_k2, set10, sum10);
             vects[j+11] = _mm256_fmadd_pd(mat1_in_k2, set11, sum11);
         }
     
         /* j LOOP TAIL CASE */
          for (j = m2_cols / 12 * 12; j < m2_cols; j++) {
             tail_set0 = _mm256_set_pd(*(m2 + j + ((k+3) * m2_cols)), *(m2 + j + ((k+2) * m2_cols)), *(m2 + j + ((k+1) * m2_cols)), *(m2 + j + (k * m2_cols)));
             tail_set1 = _mm256_set_pd(*(m2 + j + ((k+7) * m2_cols)), *(m2 + j + ((k+6) * m2_cols)), *(m2 + j + ((k+5) * m2_cols)), *(m2 + j + ((k+4) * m2_cols)));
             tail_set2 = _mm256_set_pd(*(m2 + j + ((k+11) * m2_cols)), *(m2 + j + ((k+10) * m2_cols)), *(m2 + j + ((k+9) * m2_cols)), *(m2 + j + ((k+8) * m2_cols)));

             vects[j] = _mm256_fmadd_pd(mat1_in_k0, tail_set0, vects[j]);
             vects[j] = _mm256_fmadd_pd(mat1_in_k1, tail_set1, vects[j]);
             vects[j] = _mm256_fmadd_pd(mat1_in_k2, tail_set2, vects[j]);
            }
        }
     
       /* k LOOP TAIL CASE */
      for (k = m1_cols / 12 * 12; k < m1_cols; k++) { //m1_cols == mat2->rows FYI
          for (j = 0; j < m2_cols; j++) {
            double tail_val = 0;
            tail_val = *(m1 + (i * m1_cols) + k) * *(m2 + j + (k * m2_cols));
            tail_vals[j] += tail_val;
          } 
      }
     
       /* adds each __m256d vector into dval and then stores */
      for (l = 0; l < m2_cols / 12 * 12; l+=12) {         
         dval0 = 0;
         double arr0[4];
         vval0 = vects[l];
       
         dval1 = 0;
         double arr1[4];
         vval1 = vects[l+1];
       
         dval2 = 0;
         double arr2[4];
         vval2 = vects[l+2];
       
         dval3 = 0;
         double arr3[4];
         vval3 = vects[l+3];
       
         dval4 = 0;
         double arr4[4];
         vval4 = vects[l+4];
       
         dval5 = 0;
         double arr5[4];
         vval5 = vects[l+5];
       
         dval6 = 0;
         double arr6[4];
         vval6 = vects[l+6];
       
         dval7 = 0;
         double arr7[4];
         vval7 = vects[l+7];
         
         dval8 = 0;
         double arr8[4];
         vval8 = vects[l+8];
       
         dval9 = 0;
         double arr9[4];
         vval9 = vects[l+9];
       
         dval10 = 0;
         double arr10[4];
         vval10 = vects[l+10];
       
         dval11 = 0;
         double arr11[4];
         vval11 = vects[l+11];
       
         _mm256_storeu_pd(arr0, vval0);
         _mm256_storeu_pd(arr1, vval1);
         _mm256_storeu_pd(arr2, vval2);
         _mm256_storeu_pd(arr3, vval3);
         _mm256_storeu_pd(arr4, vval4);
         _mm256_storeu_pd(arr5, vval5);
         _mm256_storeu_pd(arr6, vval6);
         _mm256_storeu_pd(arr7, vval7);
         _mm256_storeu_pd(arr8, vval8);
         _mm256_storeu_pd(arr9, vval9);
         _mm256_storeu_pd(arr10, vval10);
         _mm256_storeu_pd(arr11, vval11);
         
         for (int m = 0; m < 4; m++) {
             dval0 += arr0[m];
             dval1 += arr1[m];
             dval2 += arr2[m];
             dval3 += arr3[m];
             dval4 += arr4[m];
             dval5 += arr5[m];
             dval6 += arr6[m];
             dval7 += arr7[m];
             dval8 += arr8[m];
             dval9 += arr9[m];
             dval10 += arr10[m];
             dval11 += arr11[m];
         }
         *(mr + (i * mr_cols) + l) = dval0 + tail_vals[l];
         *(mr + (i * mr_cols) + (l+1)) = dval1 + tail_vals[l+1];
         *(mr + (i * mr_cols) + (l+2)) = dval2 + tail_vals[l+2];
         *(mr + (i * mr_cols) + (l+3)) = dval3 + tail_vals[l+3];
         *(mr + (i * mr_cols) + (l+4)) = dval4 + tail_vals[l+4];
         *(mr + (i * mr_cols) + (l+5)) = dval5 + tail_vals[l+5];
         *(mr + (i * mr_cols) + (l+6)) = dval6 + tail_vals[l+6];
         *(mr + (i * mr_cols) + (l+7)) = dval7 + tail_vals[l+7];
         *(mr + (i * mr_cols) + (l+8)) = dval8 + tail_vals[l+8];
         *(mr + (i * mr_cols) + (l+9)) = dval9 + tail_vals[l+9];
         *(mr + (i * mr_cols) + (l+10)) = dval10 + tail_vals[l+10];
         *(mr + (i * mr_cols) + (l+11)) = dval11 + tail_vals[l+11];
       }
       
       /* TAIL CASE FOR STORING INTO ARRAY */
      for (l = m2_cols / 12 * 12; l < m2_cols; l++) {
       // for (l = 0; l < mat2->cols; l++) {         
         double dval = 0;
         double arr[4];
         __m256d vval = vects[l];
         
         _mm256_storeu_pd(arr, vval);
         
         for (int m = 0; m < 4; m++) {
           dval += arr[m];
         }
         *(mr + (i * mr_cols) + l) = dval + tail_vals[l];
       }
     }
   }
   return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (pow < 0 || result->rows != mat->rows || result->cols != mat->cols || mat->rows != mat->cols) {
      return -1; 
    }
    if (pow == 0) {
      for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
          if (i == j) {
            result->data_1[i * result->cols + j] = 1.0;
          } else {
            result->data_1[i * result->cols + j] = 0.0;
          }
        }
      }
    } else {
      matrix *res = NULL;
      allocate_matrix(&res, result->rows, result->cols);
      matrix *A = NULL;
      allocate_matrix(&A, result->rows, result->cols);
      matrix *B = NULL;
      allocate_matrix(&B, result->rows, result->cols);
      // matrix *C = NULL;
      // allocate_matrix(&C, result->rows, result->cols);
      
      int length = mat->rows * mat->cols;
      for (int i = 0; i < length; i++) {
        A->data_1[i] = mat->data_1[i];
      }
      
      for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
          if (i == j) {
            result->data_1[i * result->cols + j] = 1.0;
          } else {
            result->data_1[i * result->cols + j] = 0.0;
          }
        }
      }
      while (pow > 0) {
        if (pow % 2 == 1) {
          for (int i = 0; i < length; i++) {
            res->data_1[i] = result->data_1[i];
          }
          mul_matrix_set_unrolled(result, res, A);
        }
        
        for (int i = 0; i < length; i++) {
          B->data_1[i] = A->data_1[i];
          // C->data[i] = A->data[i];
        }  
        mul_matrix_set_unrolled(A, B, B); 
        pow = pow / 2; 
      }
      deallocate_matrix(res);
      deallocate_matrix(A);
      deallocate_matrix(B);
      // deallocate_matrix(C);
    }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if (mat == NULL || result->rows != mat->rows || result->cols != mat->cols) {
      return -1;
    }
    int length = mat->rows * mat->cols;
    // double* result_addr;
    // double* mat_addr;
    __m256d res0;
    __m256d res1;
    __m256d res2;
    __m256d res3;
    #pragma omp parallel for private(res0, res1, res2, res3)
    for (int i = 0; i < length / 16 * 16; i+=16) { 
        res0 = _mm256_mul_pd(_mm256_loadu_pd(mat->data_1 + i), _mm256_set1_pd(-1));
        res1 = _mm256_mul_pd(_mm256_loadu_pd(mat->data_1 + i+4), _mm256_set1_pd(-1));
        res2 = _mm256_mul_pd(_mm256_loadu_pd(mat->data_1 + i+8), _mm256_set1_pd(-1));
        res3 = _mm256_mul_pd(_mm256_loadu_pd(mat->data_1 + i+12), _mm256_set1_pd(-1));
        _mm256_storeu_pd(result->data_1 + i, res0);
        _mm256_storeu_pd(result->data_1 + i+4, res1);
        _mm256_storeu_pd(result->data_1 + i+8, res2);
        _mm256_storeu_pd(result->data_1 + i+12, res3);
    }
    for (int i = length / 16 * 16; i < length; i++) {
      result->data_1[i] = -(*(mat->data_1+i));
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if (result->cols != mat->cols || result->rows != mat->rows) {
        return 1;
    }
    __m256d temp1;
    __m256d temp2 = _mm256_set1_pd(-1);
    __m256d temp3;
    __m256d total;
    double *x;
    double *z;
    #pragma omp parallel for
    for(int i = 0; i < (mat->rows * mat->cols) / 16 * 16; i += 16) {
        x = mat->data_1 + i;
        z = result->data_1 + i;
        temp1 = _mm256_loadu_pd(x);
        temp3 = _mm256_mul_pd(temp1, temp2);
        total = _mm256_max_pd(temp1, temp3);
        _mm256_storeu_pd(z, total);

        x = mat->data_1 + i + 4;
        z = result->data_1 + i + 4;
        temp1 = _mm256_loadu_pd(x);
        temp3 = _mm256_mul_pd(temp1, temp2);
        total = _mm256_max_pd(temp1, temp3);
        _mm256_storeu_pd(z, total);

        x = mat->data_1 + i + 8;
        z = result->data_1 + i + 8;
        temp1 = _mm256_loadu_pd(x);
        temp3 = _mm256_mul_pd(temp1, temp2);
        total = _mm256_max_pd(temp1, temp3);
        _mm256_storeu_pd(z, total);

        x = mat->data_1 + i + 12;
        z = result->data_1 + i + 12;
        temp1 = _mm256_loadu_pd(x);
        temp3 = _mm256_mul_pd(temp1, temp2);
        total = _mm256_max_pd(temp1, temp3);
        _mm256_storeu_pd(z, total);
    }
    
    for(int i = (mat->rows * mat->cols) / 16 * 16; i < mat->rows * mat->cols; i++) {
        if(mat->data_1[i] < 0) {
            result->data_1[i] = mat->data_1[i] * -1;
        } else {
            result->data_1[i] = mat->data_1[i];
        }
    }
    return 0;
}

