import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lorenz 63 model
def lorenz63(t, state, sigma=10.0, beta=8/3, rho=28.0):
    """
    Lorenz 63 model differential equations.
    
    Parameters:
        t : float
            Time variable (not used explicitly in the equations, but required by the solver).
        state : list or array
            A list or array of the current state [x, y, z].
        sigma : float
            Parameter sigma (default 10.0).
        beta : float
            Parameter beta (default 8/3).
        rho : float
            Parameter rho (default 28.0).
    
    Returns:
        list
            Derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Set initial conditions and time span
initial_state = [1.0, 1.0, 1.0]
t_start = 0.0
t_end = 40.0
num_points = 10000
t_eval = np.linspace(t_start, t_end, num_points)

# Solve the ODEs
solution = solve_ivp(
    lorenz63,
    t_span=(t_start, t_end),
    y0=initial_state,
    t_eval=t_eval,
    args=(),  # Using default parameters sigma=10, beta=8/3, rho=28
    rtol=1e-8,
    atol=1e-10
)

# Plot the x-component of the solution
plt.figure(figsize=(10, 4))
plt.plot(solution.t, solution.y[0], label="x(t)")
plt.xlabel("Time")
plt.ylabel("x")
plt.title("Lorenz 63 Model - x Component")
plt.legend()
plt.show()

# Optionally, plot the 3D trajectory
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz 63 Attractor")
plt.show()