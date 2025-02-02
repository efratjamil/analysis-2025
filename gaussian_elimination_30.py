
import numpy as np

def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]

def forward_elimination(matrix):
    N = len(matrix)
    for k in range(N):
        pivot_row = np.argmax(np.abs(matrix[k:, k])) + k
        if matrix[pivot_row, k] == 0:
            return k

        if pivot_row != k:
            swap_rows(matrix, k, pivot_row)

        for i in range(k + 1, N):
            factor = matrix[i, k] / matrix[k, k]
            matrix[i, k:] -= factor * matrix[k, k:]

    return -1

def backward_substitution(matrix):
    N = len(matrix)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (matrix[i, -1] - np.dot(matrix[i, i + 1:N], x[i + 1:N])) / matrix[i, i]
    return x

def gaussian_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    augmented_matrix = np.hstack((A, b))
    singular_flag = forward_elimination(augmented_matrix)
    if singular_flag != -1:
        return "Singular Matrix - No Unique Solution"
    return backward_substitution(augmented_matrix)


def main():
    A = np.array([[1, 2, -2],
                  [1, 1, 1],
                  [2, 2, 1]], dtype=float)
    b = np.array([7, 2, 5], dtype=float)

    solution = gaussian_elimination(A, b)
    print("Solution:", solution)

if __name__ == "__main__":
    main()