#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:25:12 2020

@author: Natalie
"""

import numpy as np
import matplotlib.pyplot as plt

#defining the numerical differentiation closure 
def fun(f,h):
    def numerical_diff(x):   #closure is the inner function
        
        output = (f(x+h) - f(x)) / h
        return output
    
    return numerical_diff

#defining the independent variable x as an array of evenly spaced points between two limits
xmin = 0.2
xmax = 0.4
numpoints = 100
x = np.linspace(xmin,xmax,numpoints)

#Two ways to define the function:
# 1st
def f(x):
    return np.log(x) 

#2nd 
ex = lambda x: np.log(x) #the input function is ln(x)

ex_dr = np.array(1./x)    #the exact derivative of ln(x)

#print(ex_dr)

#the 3 values of h 
h = [0.1, 1e-7, 1e-15]


#creating a list of 3 numerical derivatives for each value of h over all x values
ndiff = []

for i in range(len(h)):
    derivative = fun(ex,h[i])
    difflist = np.array(derivative(x))
    ndiff.append(difflist)

#print(ndiff)


'''
Plotting the numerical derivative at each value of h,
and the exact derivative.
'''
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(1,1,1)
ax.plot(x, ndiff[0], 'r', lw=1, label='h=0.1')
ax.plot(x, ndiff[1], 'g', lw=1, label='h=1e-7')
ax.plot(x, ndiff[2], 'y', lw=1, label='h=1e-15')
ax.plot(x, ex_dr, 'b', ls='--', lw=3, label='exact')
# ax.plot(x, x, ls='--', lw=3, label=r'$y_{2} = x$')

ax.set_xlim(0.18, 0.43)
ax.set_ylim(2.0, 6.0)
ax.set_xlabel(r'$x$', fontsize=24)
ax.set_ylabel(r'$y$', fontsize=24)
ax.set_title(r'Derivatives', fontsize=24)
ax.tick_params(labelsize=24)
ax.legend(fontsize=24);

print("Answer to Q-a:\n")
print("The h-value that most closely approximates the true derivative is: {:.3e}".format(h[1]))
print("\nh values that are too small cause variance because of numerical error build-up.  \n")
print("h values that are too large cause a bias, a deviation from the original data.\n")

print("Answer to Q-b:\n")
print("Automatic differentiation addresses these problems by breaking down the function into\n")
print("elementary functions and combines them using the chain rule. ")

plt.show()
