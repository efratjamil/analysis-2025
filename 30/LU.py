import numpy as np
import scipy.linalg

def lu_decomposition(A):
    return scipy.linalg.lu_factor(A)

def lu_solve(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    lu, piv = lu_decomposition(A)
    x = scipy.linalg.lu_solve((lu, piv), b)
    return x

def main():
    A = np.array([[1, 2, -2],
                  [1, 1, 1],
                  [2, 2, 1]], dtype=float)
    b = np.array([7, 2, 5], dtype=float)

    solution = lu_solve(A, b)
    print("Solution:", solution)

if __name__ == "__main__":
    main()