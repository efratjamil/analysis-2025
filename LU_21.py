import numpy as np

def lu_decomposition(A):
    N = len(A)
    L = np.eye(N)

    for i in range(N):
        pivot_row = max(range(i, N), key=lambda j: abs(A[j][i]))

        if A[pivot_row][i] == 0:
            raise ValueError("Cannot perform LU Decomposition")

        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]
            print(f"Swapped rows {i} and {pivot_row}\n{A}\n")

        for j in range(i + 1, N):
            m = -A[j][i] / A[i][i]
            A[j] += m * A[i]
            L[j, i] = -m

    return L, A

def backward_substitution(U):
    N = len(U)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (U[i, -1] - np.dot(U[i, i + 1:N], x[i + 1:N])) / U[i, i]
    return x

def lu_solve(A_b):
    L, U = lu_decomposition(np.array(A_b, dtype=float))
    print("L:\n", L)
    print("U:\n", U)
    solution = backward_substitution(U)
    print(f"\033[94mSolution:\n{solution}\033[0m")

if __name__ == "__main__":
    A_b = [[1, 1/2, 1/3, 1],
           [1/2, 1/3, 1/4, 0],
           [1/3, 1/4, 1/5, 0]]
    lu_solve(A_b)