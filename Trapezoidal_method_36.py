
import numpy as np
import math

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)
    return integral * h

def interpolate(points, x):
    n = len(points)
    result = 0
    for i in range(n):
        term = points[i][1]
        for j in range(n):
            if i != j:
                term *= (x - points[j][0]) / (points[i][0] - points[j][0])
        result += term
    return result

if __name__ == "__main__":
    data_points = [(2, 0), (2.25, 0.112463), (2.3, 0.167996), (2.7, 0.222709)]
    x_target = 2.4
    interpolated_value = interpolate(data_points, x_target)
    print(f"Interpolated value at x={x_target}: {interpolated_value}")

   
    f = lambda x: (2*x*np.exp(-x) + np.log(2*x**2)) * (2*x**2 - 3*x - 5)
    a, b, n = 0.5, 1, 10
    integral_value_trapezoidal = trapezoidal_rule(f, a, b, n)
    print(f"Integral value over [{a},{b}] using Trapezoidal rule: {integral_value_trapezoidal}")