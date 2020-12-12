import numc as nc

def test_add():
    mat1 = nc.Matrix(3, 3, 1)
    mat2 = nc.Matrix(3, 3, 2)
    add = nc.Matrix(3, 3)
    sol = nc.Matrix(3, 3, 3)
    add = mat1 + mat2
    for i in range(3):
        for j in range(3):
            assert sol[i][j] == add[i][j], "Should be equal"


if __name__ == "__main__":
    test_add()
    print("Everything passed")