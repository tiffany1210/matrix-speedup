import numc as nc 

def testslice():
	mat = nc.Matrix(1, 3)
	print(mat[0])
	print(mat[0:2])

	mat1 = nc.Matrix(3, 1)
	print(mat[0:2])

def testslice2():
	a = nc.Matrix(3, 3) 
	print(a[0:2, 0])
	print(a[0, 0:2])
	print(a[0:2, 0:2])

	print(a[0:1, 0:1])

def testslice3():
	a = nc.Matrix(1, 3) 
	a[0:2] = [1, 2]
	print(a)

if __name__ == "__main__":
    #testslice()
    testslice3()
    print("Everything passed")