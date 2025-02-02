import numpy as np
from numpy.linalg import norm


def gauss_seidel(A, b, x0, tol=1e-10, max_iter=500):
    n = len(A)
    x = np.zeros(n, dtype=np.double)

    print("Iteration" + "\t".join([" {:>12}".format(f"x{i + 1}") for i in range(n)]))
    print("-" * 90)

    for k in range(1, max_iter + 1):
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - x0, np.inf) < tol:
            return tuple(x)

        x0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == "__main__":
    A = np.array([[1, 1 / 2, 1 / 3],
                  [1 / 2, 1 / 3, 1 / 4],
                  [1 / 3, 1 / 4, 1 / 5]])
    b = np.array([1, 0, 0])
    x0 = np.zeros_like(b)

    solution = gauss_seidel(A, b, x0)
    print(f"\033[94m\nApproximate solution: {solution}\033[0m")