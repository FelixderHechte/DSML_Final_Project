import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lorenz 63 model
def lorenz63_model(t, state, sigma=10.0, beta=8/3, rho=28.0):
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz63_timeseries(initial_state=[1.0, 1.0, 1.0], t_start=0.0, t_end=40.0, num_points=10000, sigma=10.0, beta=8/3, rho=28.0):
    """
    Solves the Lorenz 63 system and returns the time series.
    
    Parameters:
        initial_state : list or array, optional
            Initial conditions [x, y, z] (default: [1.0, 1.0, 1.0])
        t_start : float, optional
            Start time of the simulation (default: 0.0)
        t_end : float, optional
            End time of the simulation (default: 40.0)
        num_points : int, optional
            Number of time points in the output (default: 10000)
        sigma : float, optional
            Parameter sigma (default: 10.0)
        beta : float, optional
            Parameter beta (default: 8/3)
        rho : float, optional
            Parameter rho (default: 28.0)
    
    Returns:
        t : ndarray
            Time values of the solution
        sol : ndarray
            Solution array of shape (num_points, 3) containing x, y, z values
    """
    def lorenz63_model(t, state, sigma, beta, rho):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]
    
    t_eval = np.linspace(t_start, t_end, num_points)
    solution = solve_ivp(
        lorenz63_model,
        t_span=(t_start, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(sigma, beta, rho),
        rtol=1e-8,
        atol=1e-10
    )
    
    return solution.t, solution.y.T

# Example usage with plotting
t, sol = lorenz63_timeseries()
plt.plot(t, sol[:, 0], label="x(t)")
plt.plot(t, sol[:, 1], label="y(t)")
plt.plot(t, sol[:, 2], label="z(t)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Lorenz 63 System")
plt.show()

# Optionally, plot the 3D trajectory
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol[:,0], sol[:,1], sol[:,2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz 63 Attractor")
plt.show()