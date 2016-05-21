"""
File: Backward2.py
Exercise 9.11
Copyright (c) 2016 Andrew Malfavon
License: MIT
Description: implementing a new subclass for differentiating a function.
"""

import numpy as np
import pandas as pd

class Diff:
    def __init__(self, f, h=1E-9):
        self.f = f
        self.h = float(h)

class Forward1(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h) - f(x))/h

#old subclass to be compared:
class Backward1(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x) - f(x-h))/h

#new subclass:
class Backward2(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x-2*h) - 4 * f(x-h) + 3 * f(x))/(2*h)

class Central2(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h) - f(x-h))/(2*h)

class Central4(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (4./3)*(f(x+h) - f(x-h)) /(2*h) - (1./3)*(f(x+2*h) - f(x-2*h))/(4*h)

class Central6(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (3./2) *(f(x+h) - f(x-h)) /(2*h) - (3./5) *(f(x+2*h) - f(x-2*h))/(4*h) + (1./10)*(f(x+3*h) - f(x-3*h))/(6*h)

class Forward3(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (-(1./6)*f(x+2*h) + f(x+h) - 0.5*f(x) - (1./3)*f(x-h))/h

#function to be implemented
def g(t):
    return np.exp(-t)

#exact analytic derivative
def g_prime(t):
    return -1 * np.exp(-t)

def compare():
    #compare Backward1 to Backward2
    Backward1_array = []
    Backward2_array = []
    Backward1_error_array = []
    Backward2_error_array = []
    for k in range(15):
        #Backward1 and Backward2 approximations
        Backward_1 = Backward1(g, 2**(-k))
        Backward_2 = Backward2(g, 2**(-k))
        Backward1_array.append(Backward_1(0))
        Backward2_array.append(Backward_2(0))
    for i in range(15):
        #Backward1 and Backward2 errors using analytic derivative
        Backward1_error_array.append(abs(Backward1_array[i] - g_prime(0)))
        Backward2_error_array.append(abs(Backward2_array[i] - g_prime(0)))
    return Backward1_array, Backward2_array, Backward1_error_array, Backward2_error_array

def table():
    #put the approximations for both and the errors into a table for 15 different k values
    header_x = np.array(['Backward1', 'Backward1 Error', 'Backward2', 'Backward2 Error'])
    header_y = np.array(['k=0', 'k = 1', 'k = 2', 'k = 3', 'k = 4', 'k = 5', 'k = 6', 'k = 7', 'k = 8', 'k = 9', 'k = 10', 'k = 11', 'k = 12', 'k = 13', 'k = 14',])
    long_array = []
    matrix = np.zeros([15, 4])#empty matrix
    counter = 0
    for i in range(len(compare()[0])):
        #arrange the four arrays from compare() so they can be put into the empty matrix in the right order
        long_array.append(compare()[0][i])
        long_array.append(compare()[2][i])
        long_array.append(compare()[1][i])
        long_array.append(compare()[3][i])
    for j in range(15):
        for k in range(4):
            #separate the array into a matrix
            matrix[j, k] = long_array[counter]
            counter += 1
    table = pd.DataFrame(matrix, index = header_y, columns = header_x)
    return table

def test_Backward2():
    assert abs(compare()[0][-1] + 1.0) < 1E-4