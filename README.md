# numc

Here's what I did in project 4:
- allocate_matrix was written so that it does not set its parent to the direct parent so that a long chain of parents can be avoided. Instead, I created a tree like structure where every matrix/slice points to one single parent. Then, for deallocate matrix, I made sure that while all the 1-d and 2-d data are being freed, that other pointer pointing to the same parent matrix is not being freed.

For number methods:
- For each method, I threw a value error if the arguments’ dimensions are invalid and a runtime error if any memory allocation fails during execution.
- add: I checked for the argument type of “args”, the second operand for the type error. I checked if the dimensions match for both self and args for the value error. Then, I allocate matrix *sum, and use the method add_matrix to add the self matrix and args matrix. Then, I initialized a new numc.Matrix object to hold my result, and set the shape attribute of the new matrix.
- Sub: This was very similar to add method.
multiply: Similar, except that the dimensions check was different.
- pow: I checked if pow is the PyLong type, and also checked if the dimensions match (should be a square matrix). Then I allocated space for the power matrix, which was called to pow_matrix function. 
Then I created a PyNumberMethods struct for overloading operators with all the number methods that I defined.
- set: “args” = (row, col, val). Firstly I checked if the arguments passed in “args” has 3 arguments by using PyArg_UnpackTuple, and then if not, returned invalid arguments type error. If there are 3 arguments, those were unpacked into different pyobjects, and checked if row and col are integers. Then, they were checked for right range of index. For val, I checked if it is either a float or an integer, and then used the function “set” to set the value correctly.
- get: Check if the args size is 2, and if arg1 and arg2 are integers. Check for the right index range, and then use the function get to return the value at the row, col index. Since the type is double, I put PyFloat_FromDouble to return PyObject.

- subscript: I divided into two parts: 
1. self->mat is 1-d matrix
A) key is an integer: check for the index out of range, and return the data[key] directly using PyFloat_FromDouble.
B) key is a slice: check for the valid slice, if slicelength is 1, return the data[key] directly, if not, then allocate a matrix by setting row_offset = start, rows = slicelength, and column offset to 0 and cols to 1. Since it is a 1d matrix, it will be a list of values, so the column size = 1.
2. self->mat is 2-d matrix:
A) key is an integer: similar with 1-d matrix, except that size = number of rows. Also, we allocate matrix and set the row_offset = index, rows = 1, col_offset = 0, cols = number of cols. This will mean that we are extracting a single row from the matrix.
B) key is a slice: similar with 1-d matrix, except that now we allocate matrix: row_offset = start, rows = slicelength, and for columns it is all the columns. This ensures that we extract a row slice from the matrix.
C) key is a tuple:
1) arg1 = integer, arg2 = integer: check if both args are index within the range, and call Matrix61c get function. This handles if arg1 and arg2 are integers type.
2) arg1 = integer, arg2 = slice: Allocate matrix: row_offset = index (arg1), rows = 1, col_offset = start, cols = slicelength. This means that only one row is extracted with a col slice. 
3) arg1 = slice, arg2 = integer: again, check if arg2 is index within the range, and if the arg1 slice is a valid slice. Then, allocate matrix : row_offset = start, rows = slicelength, col_offset = index (which is PyLong_AsLong(arg2)), cols = 1. This will mean that we only care 
4) arg1 = slice, arg2 = slice: For this case, I initialized new pyobjects for start2, stop2, step2, and slicelength2 for arg2. By using the function PySlice_GetIndicesEx, I I checked for a valid slice, and allocate matrix accordingly. 
- The reason my slice was not working for 1d matrix was that I wasn’t checking whether the number of row = 1 or number of col = 1. For each case, it had to be handled differently using different arguments for allocate_matrix_ref. For the integer value, I used 1d array to index into so that I was able to get the type right.
Also, my code was not running correctly because the arguments to PySlice_GetIndicesEx were not the right types. After changing it to Py_ssize_t and addressing it so that it points to this long integer type, it worked. A lot of conversions between pyobject types and number types were needed.  

- set_subscript: Similarly, I splitted it up into all possibilities.
Most of the issue came from not taking care of a case when slice length = 1, and also not setting the indices correctly when it comes to slicing. There were many type checks required so I created four helper functions for each case of the tuples. This helped me to identify where the errors were coming from when debugging.

For speeding up:
- For multiply, I first naively coded an algorithm where each ith row of the first matrix is multiplied by each jth column of the second matrix and stored in the (i, j)th entry of the result matrix. This was then vectorized specifically by using functions _mm256_loadu_pd to load 4 entries from each ith row of the first matrix, and _mm256_set_pd to load 4 entries from each 4th column of the second matrix, and respective tail cases using the naive code were implemented. The k and j loops were re-ordered to increase cache-friendliness and they were unrolled to size of 8 to speed up. The outermost loop was unrolled to size of 4 as well. 
Instead of storing each value inside the innermost loop, I created a vector __m256d values to hold each result of row * column, which were then all converted into double values and stored into the result array at the end of each ith loop/iteration. 
- Naive matrix multiplication is so slow because accesses to the second operand are not contiguous in memory. Therefore, we can rearrange loop orders or change the way you store the second operand so that the stride in the innermost loop is 1. I rearranged loop orders and change the second operand to store 1-d array and set the stride in the innermost loop to 1, with simd and multiple unrolling. I also reduced the number of storeu operations since they are expensive.
- I implemented matrix exponentiation algorithm for power, and used my unrolled matrix multiply within the code. I checked if power n = even for a**n, and then carried out a**(n/2) * a **(n/2). For n = odd, I carried out a**(n/2) * a**(n/2) * a.
- All functions except power implemented OMP for parallel computation. These functions were wrapped in Matrix61c in numc.c and linked in setup.py. This allows us to use numc as a library in python just like the library numpy in python. Hence, I was able to compute matrix arithmetic as well as throw typeerror, index error and runtime errors upon invalid behaviors. 
I used calloc instead of malloc in allocate_matrix since it is faster. Malloc manually zeroes out the data you allocate for matrices.
- Then, for add, neg, sub, abs, I used SIMD instructions as well as loop unrolling to increase the speed. The outermost for loop was parallelized. I treated the entire matrix as a 1d array so that there will be only a single tail case. This one contiguous array helps improve the simple methods, since we are treating them element-wise. So basically I added 1d row major matrix called data_1 inside matrix struct. Then, I let the 2d data to point to that data_1 in the right place. So for all the arithmetic functions for matrices, I changed 2d array style into a 2d array: mat[i][j] => *(mat + i*m + j), where m = number of columns. 
- I tested by printing out individual values in the unittest.py for correctness, and checked performance using the speedup print out. 
