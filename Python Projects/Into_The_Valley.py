# Problem 1: Into the valley
# Import math libraries

import numpy as np # Import numpy for array manipulation
import matplotlib.pyplot as plt # Import matplotlib for plotting functions
import random # Import random for random number generation when asked

# Define the Objective Function
def f(x, y):
        return np.sin(x) ** 2 + np.sin(y) ** 2 + (2 * x ** 2 + y ** 2) # Landscape function as-defined in the problem

# Define the Gradient of the Objective Function
def gradF(x, y):
        df_dx = 2 * np.cos(x) * np.sin(x) + 4*x # Partial derivative of the objective function with respect to x
        df_dy = 2 * np.cos(y) * np.sin(y) + 2*y # Partial derivative of the objective function with respect to y
        return np.array([df_dx, df_dy]) # Return the gradient as an array

# Implement the first Algorithm called "Gradient Descent"
def gradient_descent(startingPoint, alpha, numIterations): # Mark, the words after def are the name of a "Method" in Python, the "Variables" in the parentheses are called "Parameters" that are passed into the method
       path = [startingPoint] # Initialize the "path array" with the starting point

       for _ in range(numIterations): # This is called a "for loop" in Python, the "_" is a placeholder for a variable that is not used
            currentPoint = path[-1] # The current point is the last point in the path array
            gradient = gradF(*currentPoint) # The gradient is the gradient of the objective function evaluated at the current point
            nextPoint = currentPoint - alpha * gradient
            path.append(nextPoint) # Append the next point to the "path" array
            return np.array(path) # Return the path as an array

# Define Parameters, these are the inputs to the methods
startingPoint = np.array([random.uniform(-8, 8), random.uniform(-8, 8)]) # The starting point is defined by Domain (x,y) ∈ [-8, 8] x [-8, 8]
alpha = 0.01 # The step size is defined as 0.1, it may need to be adjusted
num_iterations = 200 # The number of iterations is defined as 200, it may need to be adjusted

#Execute the Gradient Descent Algorithm
path = gradient_descent(startingPoint, alpha, num_iterations)

#Plotting
x = np.linspace(-8, 8, 400) # Create a 400 point array from -8 to 8
y = np.linspace(-8, 8, 400) # Create a 400 point array from -8 to 8
X, Y = np.meshgrid(x, y) # Create a grid of points from the x and y arrays
Z = f(X, Y) # Evaluate the objective function at the grid points

plt.contour(X, Y, Z, levels=50) # Plot the contour lines of the objective function
plt.plot(path[:,0], path[:,1], '-o', color='red') # Plot the Path in red, this is arbitrary
plt.show() # Plot the path taken by the gradient descent algorith