#!/usr/bin/env python3

from gradient_descent import (step, move_point_along_ei, partial_difference_quotient,
                              estimate_gradient, minimize)
from pytest import approx
from math import sqrt

def test_step():
    v = [0.,0.,0.]
    d = [1.,2.,3.]
    s = step(v, d, 1.)
    assert v == approx([0.,0.,0.]), "Code shouldn't change inputs!"
    assert d == approx([1.,2.,3.]), "Code shouldn't change inputs!"
    assert s == approx([1.,2.,3.]), "Check step code, step_size=1.0"
    s = step(v, d, 0.1)
    assert v == approx([0.,0.,0.]), "Code shouldn't change inputs!"
    assert d == approx([1.,2.,3.]), "Code shouldn't change inputs!"
    assert s == approx([0.1,0.2,0.3]), "Check step code, step_size=0.1"
    s = step(v, d, 0.01)
    assert v == approx([0.,0.,0.]), "Code shouldn't change inputs!"
    assert d == approx([1.,2.,3.]), "Code shouldn't change inputs!"
    assert s == approx([0.01,0.02,0.03]), "Check step code, step_size=0.01"
    v = [1.,1.,1.]
    s = step(v, d, 0.001)
    assert v == approx([1.,1.,1.]), "Code shouldn't change inputs!"
    assert d == approx([1.,2.,3.]), "Code shouldn't change inputs!"
    assert s == approx([1.001,1.002,1.003]), "Check step code, step_size=0.001"
    

def test_move_point_along_ei():
    v = [0., 0., 0.]
    s = move_point_along_ei(v, 0, 1.)
    assert v == approx([0, 0, 0]), "Code shouldn't change inputs!"
    assert s == approx([1, 0, 0]), "Check definition of i'th component (starts from 0)"
    s = move_point_along_ei(v, 1, 1.)
    assert s == approx([0, 1, 0]), "Check definition of i'th component (starts from 0)"
    s = move_point_along_ei(v, 2, 1.)
    assert s == approx([0, 0, 1]), "Check definition of i'th component (starts from 0)"
    v = [1., 2., 3.]    
    s = move_point_along_ei(v, 0, 0.1)
    assert s == approx([1.1, 2, 3]), "Check definition of i'th component (starts from 0)"
    s = move_point_along_ei(v, 1, 0.1)
    assert s == approx([1., 2.1, 3]), "Check definition of i'th component (starts from 0)"
    

def test_partial_difference_quotient():
    # f(x, y, z) = x^2 + y^2 + z^2, @ x=(0,0,0)
    f = lambda v: v[0]**2 + v[1]**2 + v[2]**2
    v = [0., 0., 0.]
    # ((0.01^2 + 0^2 + 0^2) - (0 - 0 - 0)) / .01
    assert partial_difference_quotient(f, v, 0, 0.01) == approx(0.01)
    assert partial_difference_quotient(f, v, 1, 0.01) == approx(0.01)
    assert partial_difference_quotient(f, v, 2, 0.01) == approx(0.01)
    # f(x, y, z) = x^2 + y^2 + z^2, @ x=(1,0,0)
    f = lambda v: v[0]**2 + v[1]**2 + v[2]**2 + 4
    v = [1., 0., 0.]
    # ((1.01^2 + 0^2 + 0^2) - (1^2 - 0 - 0)) / .01
    assert partial_difference_quotient(f, v, 0, 0.01) == approx((1.01**2 - 1) / 0.01)
    assert partial_difference_quotient(f, v, 1, 0.01) == approx(0.01)
    assert partial_difference_quotient(f, v, 2, 0.01) == approx(0.01)
    


def test_estimate_gradient():
    # f(x, y, z) = x^2 + y^2 + z^2, @ x=(1,0,0)
    f = lambda v: v[0]**2 + v[1]**2 + v[2]**2 + 1
    v = [1., 0., 0.]
    # ((1.01^2 + 0^2 + 0^2) - (1^2 - 0 - 0)) / .01
    assert estimate_gradient(f, v, 0.01) == [approx((1.01**2 - 1) / 0.01), approx(0.01), approx(0.01)]
    # f(x, y, z) = x^2 + y^2 + z^2, @ x=(1,0,0)
    f = lambda v: v[0]**3 + v[1]**2 + v[2]**2 + 2
    v = [1., 0., 1.]
    # ((1.01^2 + 0^2 + 0^2) - (1^2 - 0 - 0)) / .01
    assert estimate_gradient(f, v, 0.01) == [approx((1.01**3 - 1) / 0.01), approx(0.01), approx((1.01**2 - 1) / 0.01)]


# Need special approximately since we don't exactly reach the minima
def approximately(a, b):
    return all(abs(aa - bb) < 1e-4 for aa, bb in zip(a, b))


def test_minimize():
    ## Test specific aspects of the algorithm
    
    # test tolerance
    assert minimize(lambda x: x[0]**2, lambda x: [x[0]], [0], [0.1], 0.1) == [0]
    assert minimize(lambda x: x[0]**2, lambda x: [x[0]], [1], [1], 2) == [1]
    # test one step
    assert minimize(lambda x: x[0]**2, lambda x: [x[0]], [1], [1], 0.1) == [0]

    ## General minimization tests

    assert approximately(minimize(lambda x: x[0]**2 + x[1]**2, lambda x: [2*x[0], 2*x[1]], [1, 1], [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 1e-9), [0, 0])
    assert approximately(minimize(lambda x: (x[0]-1.)**2 + x[1]**2, lambda x: [2*(x[0]-1), 2*x[1]], [1, 1], [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 1e-9), [1, 0])
    # test with approximate gradient
    f = lambda x: (x[0] - 1.)**2 + (x[1] - 3)**2
    df = lambda x: estimate_gradient(f, x, 1e-5)
    assert approximately(minimize(f, df, [1, 1], [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 1e-9), [1, 3])
    # multiple minima
    f = lambda x: x[0]**4 - x[0]**2
    df = lambda x: [4*x[0]**3 - 2*x[0]]
    step_sizes = [10, 1, 0.1, 0.001, 0.0001]
    tol = 1e-9
    # grad f exactly
    assert approximately(minimize(f, df, [0.1], step_sizes, tol), [1.0/sqrt(2)])
    assert approximately(minimize(f, df, [-0.1], step_sizes, tol), [-1.0/sqrt(2)])
    # approximate gradient
    assert approximately(minimize(f, lambda x: estimate_gradient(f, x, 1e-5), [0.1], step_sizes, tol),
                         [1.0/sqrt(2)])
    assert approximately(minimize(f, lambda x: estimate_gradient(f, x, 1e-5), [-0.1], step_sizes, tol),
                         [-1.0/sqrt(2)])
