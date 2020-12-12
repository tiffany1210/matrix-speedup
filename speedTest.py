import time
import numc as nc
import numpy as np
import threading

def addTime():
    print("=" * 60)
    print("{:^54}".format("test Add speed"))
    print("{:>34}{:>10}{:>10}".format(" ", "numc", "NumPy"))
    print("=" * 60)
    size = [1000, 2000, 5000, 10000, 20000]
    for s in size:
        mat1 = nc.Matrix(s, s, rand=True)
        mat2 = nc.Matrix(s, s, rand=True)
        start = time.time()
        res = mat1 + mat2
        end = time.time()
        t1 = end - start

        mat1 = np.random.random((s, s))
        mat2 = np.random.random((s, s))
        start = time.time()
        res = mat1 + mat2
        end = time.time()
        print("Add two {:6d} by {:6d} matrices {:10.3f} {:10.3f}".format(s, s, t1, end - start))

def subTime():
    print("=" * 60)
    print("{:^59}".format("test Subtract speed"))
    print("{:>39}{:>10}{:>10}".format(" ", "numc", "NumPy"))
    print("=" * 60)
    size = [100, 1000, 2000, 5000, 10000, 20000]
    for s in size:
        mat1 = nc.Matrix(s, s, rand=True)
        mat2 = nc.Matrix(s, s, rand=True)
        start = time.time()
        res = mat1 - mat2
        end = time.time()
        t1 = end - start

        mat1 = np.random.random((s, s))
        mat2 = np.random.random((s, s))
        start = time.time()
        res = mat1 - mat2
        end = time.time()
        print("Subtract two {:6d} by {:6d} matrices {:10.3f} {:10.3f}".format(s, s, t1, end - start))

def mulTime():
    print("=" * 60)
    print("{:^59}".format("test Multiplication speed"))
    print("{:>39}{:>10}{:>10}".format(" ", "numc", "NumPy"))
    print("=" * 60)

    size = [100, 1000, 2000, 5000]
    for s in size:
        mat1 = nc.Matrix(s, s, rand=True)
        mat2 = nc.Matrix(s, s, rand=True)
        start = time.time()
        res = mat1 * mat2
        end = time.time()
        t1 = end - start

        mat1 = np.random.random((s, s))
        mat2 = np.random.random((s, s))
        start = time.time()
        res = mat1 @ mat2
        end = time.time()
        print("Multiply two {:6d} by {:6d} matrices {:10.3f} {:10.3f}".format(s, s, t1, end - start))


def negTime():
    print("=" * 60)
    print("{:^55}".format("test Negation speed"))
    print("{:>35}{:>10}{:>10}".format(" ", "numc", "NumPy"))
    print("=" * 60)
    size = [100, 1000, 2000, 5000, 10000, 20000]
    for s in size:
        mat1 = nc.Matrix(s, s, rand=True)
        start = time.time()
        res = -mat1
        end = time.time()
        t1 = end - start

        mat1 = np.random.random((s, s))
        start = time.time()
        res = np.negative(mat1)
        end = time.time()
        print("Negate a {:6d} by {:6d} matrices {:10.3f} {:10.3f}".format(s, s, t1, end - start))


def absTime():
    print("=" * 60)
    print("{:^57}".format("test Abs speed"))
    print("{:>37}{:>10}{:>10}".format(" ", "numc", "NumPy"))
    print("=" * 60)
    size = [100, 1000, 2000, 5000, 10000, 20000]
    for s in size:
        mat1 = nc.Matrix(s, s, rand=True)
        start = time.time()
        res = abs(mat1)
        end = time.time()
        t1 = end - start

        mat1 = np.random.random((s, s))
        start = time.time()
        res = np.abs(mat1)
        end = time.time()
        print("Absolute a {:6d} by {:6d} matrices {:10.3f} {:10.3f}".format(s, s, t1, end - start))


def powTime():
    print("=" * 65)
    print("{:^65}".format("test Power speed"))
    print("{:>45}{:>10}{:>10}".format(" ", "numc", "NumPy"))
    print("=" * 65)
    #size = [100, 1000, 2000, 5000, 10000]
    size = [100, 1000]
    pth = [0, 1, 2, 5, 10, 100]
    for s in size:
        for p in pth:
            mat1 = nc.Matrix(s, s, rand=True)
            start = time.time()
            res = mat1 ** p
            end = time.time()
            t1 = end - start
            
            mat1 = np.random.random((s, s))
            start = time.time()
            res = np.power(mat1, p)
            end = time.time()
            print("{:5d}th power a {:5d} by {:6d} matrices {:10.3f} {:10.3f}".format(p, s, s, t1, end - start))


def main():
    #addTime()
    # subTime()
    # mulTime()
    # negTime()
    # absTime()
    powTime()

if __name__ == "__main__":
    main()