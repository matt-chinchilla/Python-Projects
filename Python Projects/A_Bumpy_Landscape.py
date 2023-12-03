# Import requisite packages for the problem
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d # Import 3D plotting tool

# The Demands in order
    # 1) Define Objective Function f(x,y) w/ randomly generated c_nm coefficients
    # 2) Implement Heavy Ball Algorithm
    # 3) Plot the countour of f(x,y) over the domain and the paths of the optimization starting from 5 random points
    # 4) Generate plots for each starting point w/ different colors
    # 5) Perform optimization w/ 3 choices of step-size α

# Set random seed for reproducibility
np.random.seed(0)

# Generate random coefficients c_nm between 1 and 0
c_nm = np.random.rand(4, 4)

# Define the bumpy landscape function
def f_bumpy(x, y, c_nm):
    value = 0 # Initialize the value of the function
    for n in range (1,5):
        for m in range (1,5):
            value += c_nm[n-1, m-1] * np.sin(n * x / 4) * np.sin(m * y / 4) # Evaluate the function at the given x and y
    return value

# Gradient of the bumpy landscape function
def grad_f_bumpy(x, y, c_nm):
    grad_x = 0
    grad_y = 0
    for n in range(1, 5): # Iterate through the n's and m's
        for m in range(1, 5):
            grad_x += c_nm[n-1, m-1] * (n / 4) * np.cos(n * x / 4) * np.sin(m * y / 4)
            grad_y += c_nm[n-1, m-1] * (m / 4) * np.sin(n * x / 4) * np.cos(m * y / 4)
    return np.array([grad_x, grad_y])

# Define Heavy Ball Method
def heavyBallMethod(startingPoint, alpha, beta, numIterations, c_nm):
    path = [startingPoint] # Initialize the path array with the starting point
    velocity = np.array([0, 0]) # Initialize the velocity array with 0's

    for _ in range(numIterations):
        currentPoint = path[-1] # The current point is the last point in the path array
        gradient = grad_f_bumpy(*currentPoint, c_nm) # The gradient is the gradient of the objective function evaluated at the current point
        velocity = beta * velocity - alpha * gradient # Update the velocity
        nextPoint = currentPoint + velocity # Update the position
        path.append(nextPoint) # Append ({this means to add, but in code}) the next point to the path array

    return np.array(path) # Return the path as an array

# Define Parameters for the Gradient Descent Algorithm
alphas = [0.01, 0.05, 0.1] # Different Step Sizes
betas = [0.5, 0.0] #Different Momentum Coefficients
numIterations = 200
startingPoints = [np.array([np.random.uniform(np.pi, 3 * np.pi), np.random.uniform(0, 4 * np.pi)]) for _ in range(5)] # 5 Random Starting Points defined with methods

# Plot the Contour of the Objective Function
x = np.linspace(0, 3 * np.pi, 400) # Create a 400 point array from 0 to 3pi
y = np.linspace(0, 4 * np.pi, 400) # Create a 400 point array from 0 to 4pi

X, Y = np.meshgrid(x, y) # Create a grid of points from the x and y arrays
Z = f_bumpy(X, Y, c_nm) # Evaluate the objective function at the grid points

fig, ax = plt.subplots() # Create a figure and an axes
CS = ax.contour(X, Y, Z) # Plot the contour lines of the objective function

############ Perform the Optimization and ploth the Paths ############

for alpha in alphas:
    for beta in betas:
        for sp in startingPoints:
            path = heavyBallMethod(sp, alpha, beta, numIterations, c_nm) # Execute the Heavy Ball Method
            ax.plot(path[:, 0], path[:, 1], '-o', label=f'α={alpha}, β={beta}, start=({sp[0]:.2f}, {sp[1]:.2f})') # Plot the path taken by the Heavy Ball Method

# Set the title and labels for the axes
ax.set_title('Paths of Heavy Ball Method on Bumpy Landscape')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place the legend outside of the plot

plt.subplots_adjust(right=0.8) # Adjust the plot so the legend is not cut off

plt.show() # Show the plot