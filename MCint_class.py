"""
File: MCint_class.py
Exercise 9.14
Copyright (c) 2016 Andrew Malfavon
License: MIT
Description:
"""

import numpy as np
import random

class Integrator(object):
    def __init__(self, a, b, n):
        self.a, self.b, self.n = a, b, n
        self.points, self.weights = self.construct_method()

    def construct_method(self):
        raise NotImplementedError('no rule in class %s' %
                                  self.__class__.__name__)

    def integrate(self, f):
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i]*f(self.points[i])
        return s

    def vectorized_integrate(self, f):
        return dot(self.weights, f(self.points))

class Midpoint(Integrator):
    def construct_method(self):
        a, b, n = self.a, self.b, self.n # quick forms
        h = (b-a)/float(n)
        x = linspace(a + 0.5*h, b - 0.5*h, n)
        w = zeros(len(x)) + h
        return x, w

class Trapezoidal(Integrator):
    def construct_method(self):
        x = linspace(self.a, self.b, self.n)
        h = (self.b - self.a)/float(self.n - 1)
        w = zeros(len(x)) + h
        w[0] /= 2
        w[-1] /= 2
        return x, w

class Simpson(Integrator):
    def construct_method(self):
        if self.n % 2 != 1:
            print 'n=%d must be odd, 1 is added' % self.n
            self.n += 1
        x = linspace(self.a, self.b, self.n)
        h = (self.b - self.a)/float(self.n - 1)*2
        w = zeros(len(x))
        w[0:self.n:2] = h*1.0/3
        w[1:self.n-1:2] = h*2.0/3
        w[0] /= 2
        w[-1] /= 2
        return x, w

class GaussLegendre2(Integrator):
    def construct_method(self):
        if self.n % 2 != 0:
            print 'n=%d must be even, 1 is subtracted' % self.n
            self.n -= 1
        nintervals = int(self.n/2.0)
        h = (self.b - self.a)/float(nintervals)
        x = zeros(self.n)
        sqrt3 = 1.0/sqrt(3)
        for i in range(nintervals):
            x[2*i] = self.a + (i+0.5)*h - 0.5*sqrt3*h
            x[2*i+1] = self.a + (i+0.5)*h + 0.5*sqrt3*h
        w = zeros(len(x)) + h/2.0
        return x, w

class MCint_vec(Integrator):
    def construct_method(self):
        h = float(self.b - self.a) / self.n
        x = np.random.uniform(self.a, self.b, self.n)
        w = np.zeros(len(x)) + h
        return x, w

def example_func(x):
    return (8 * x**3) + (3 * x**2) +(4 * x)

def func_integrate(x):
    return (2 * x**4) + (x**3) +(2 * x**2)

def test_Mcint_vec():
    #the exact error is dependent on values selected during np.random
    example_exact = (func_integrate(3) - func_integrate(2))
    assert abs(MCint_vec(2, 3, 1000000).integrate(example_func) - example_exact) < 0.1
    assert abs(MCint_vec(0, np.pi, 1000000).integrate(np.cos)) < 0.01
    assert abs(MCint_vec(0, np.pi, 1000000).integrate(np.sin) - 2.0) < 0.01