# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:46:57 2017

@author: vshubhansh
"""

from numpy import *

def compute_error_for_given_line(b,m,points):
    #intialize error
    totalerror = 0
    for i in range(0, len(points)):
        #get the x-axis values
        x = points[i,0]
        #get teh y-axis values
        y = points[i,1]
        #total summation of the error
        #squaring the error just to get the magnitude of the deviation
        totalerror = (y-(m*x+b))**2
    #returning the average error
    return totalerror/float(len(points))

def gradient_decent(points,starting_b,starting_m,
                    learning_rate,number_of_iterations):
    #intializing b and m. m is the slope and b is the intercept on y-axis
    b = starting_b
    m = starting_m
    #starting the decent
    for i in range(number_of_iterations):
        
        b,m = step_decent(b,m, array(points),learning_rate)
    return[b,m]

def step_decent(current_b,current_m,points,learning_rate):
    #starting point for gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x= points[i,0]
        y= points[i,1]
        #calcualting teh partial derivative of the below formula wrt b and m
        #error=(1/N)(Sum(i=0 to N)((y-(mx+b))**2))
        b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2/N) * x * (y - ((current_m * x) + current_b))
        
    #updating the value of b and m for this step
    new_b= current_b - (learning_rate * b_gradient)
    new_m= current_m - (learning_rate * m_gradient)
    return[new_b,new_m]


def run():
    # Collect the data
    points = genfromtxt("data.csv", delimiter=",")
    print (points[0,0])
    #Below parameter defines how fast the model learns
    #learning rate is the rate at which the program learns
    #we make it reasonable so it could learn better at relative quick pace
    learning_rate = 0.001
    #we are going to use the slope formula y=mx + b  to calculate
    #the equation of the line
    initial_b = 0
    initial_m = 0
    number_iterations=1000
    print( "starting gradient decent at b= {0}, m= {1}, error= {2}".format
          (initial_b, initial_m,compute_error_for_given_line(initial_b,
           initial_m,points)))
    print ("Running")
    [b,m]=gradient_decent(points,initial_b,initial_m,learning_rate,
                            number_iterations)
    print ("After {0} iterations, b= {1}, m= {2}, error= {3}".
            format(number_iterations,b,m,compute_error_for_given_line
            (initial_b,initial_m,points)))

if __name__ == '__main__':
    run()


