import pickle
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def throw(
    speed: float,
    angle: int,
    drag: float,
    r: float = 0.050,
    m: float = 0.2,
    h_noise: float = 1.0,
    rho: float = 1.225,
    g: float = 9.81,
) -> dict:
    """Simulate the throw of a ball.

    Following https://scipython.com/book2/chapter-8-scipy/examples/a-projectile-with-air-resistance/

    Args:
        speed: magnitude of initial speed (m/s).
        angle: launch angle with horizontal (degrees)
        drag: drag coefficient
        r: projectile radius (m)
        m: projectile mass (kg)
        h_noise: std of measurements of altitude (m)
        rho: air density (default is at sea surface, 15C)
        g: gravitational acceleration (default is average at surface of Earth)

    Returns:
        array: simulation results containing distance travelled and height. (x-gridpts,height)
    """

    k = 0.5 * drag * rho * (np.pi * r**2)  # drag constant, proportional to area

    def deriv(t, u):
        """Return right-hand side of ODE system for the throw."""
        # see diagram at e.g. http://www.physics.smu.edu/fattarus/ballistic.html

        x, v_x, z, v_z = u
        speed = np.hypot(v_x, v_z)
        a_x, a_z = -k / m * speed * v_x, -k / m * speed * v_z - g

        return v_x, a_x, v_z, a_z

    # position and velocity components at launch
    x0, z0 = 0, 0
    rad_angle = np.radians(angle)
    v0_x, v0_z = speed * np.cos(rad_angle), speed * np.sin(rad_angle)

    # integration grid
    t = np.linspace(0, 400, 4000)

    # solve initial value problem (ivp) for distance traveled(t) and height(t)
    # df/dt = f(t,y); f(t_i) = y_i,

    solution = solve_ivp(
        deriv, t_span=(t[0], t[-1]), y0=(x0, v0_x, z0, v0_z), dense_output=True
    )  # dense => continuous solution

    # evaluate solution to obtain distance and height at each time point
    d, _, h, _ = solution.sol(t)

    # the simulator always uses the same time grid, interpolate to the same distance grid
    f = interp1d(d, h, bounds_error=False)
    d_target = np.linspace(0, 150, 151)
    h_target = f(d_target)

    # add noise to make the process stochastic, these are the final y-positions to return
    y = h_target + h_noise * np.random.randn(d_target.shape[0])

    # to obtain the params from the interactive plot, we need to return parameters here as well
    # return dict(theta=(speed, angle, drag, r, m, h_noise, rho, g), d=d_target, y=y)

    return np.concatenate([d_target, y]).reshape(2, -1)


# summary statistics
def get_landing_distance(x):
    """Compute distance travelled until projectile hits the ground.
    Args:
        x (array): (2,n) as (d,y)
            (distance travelled by projectile, height of projectile at given distance)
    Returns:
        Distance traveled in meter until projectile hits ground.
    """
    d = x[0]
    y = x[1]

    #### INSERT YOUR CODE HERE ####

    min_distance = 10  # because of noise, otherwise not reliable
    height_greater_zero = (y > 0) | (d < min_distance)
    try:
        landing_dist = d[np.argwhere(~height_greater_zero).min()]
    except ValueError:
        landing_dist = float("nan")

    ###############################

    return landing_dist


def get_distance_at_highest_point(x):
    """Compute distance travelled until projectile reaches highest point.
    Args:
        x (array): (2,n) as (d,y)
            (distance travelled by projectile, height of projectile at given distance)
    Returns:
        Distance traveled in meter until projectile reaches highest point of its trajectory.
    """
    d = x[0]
    y = x[1]

    mask = ~np.isnan(y)

    #### INSERT YOUR CODE HERE ####

    dist_at_highest_point = d[mask][y[mask].argmax()]

    ###############################
    return dist_at_highest_point


def get_highest_point(x):
    """Compute distance travelled until projectile reaches highest point.
    Args:
        x (array): (2,n) as (d,y)
            (distance travelled by projectile, height of projectile at given distance)
    Returns:
        Distance traveled in meter until projectile reaches highest point of its trajectory.
    """
    d = x[0]
    y = x[1]

    #### INSERT YOUR CODE HERE ####

    highest_point = y[~np.isnan(y)].max()

    ###############################
    return highest_point


def calculate_summary_statistics(x):
    """Calculate summary statistics for results in x"""

    return np.array(
        [
            get_landing_distance(x),
            get_distance_at_highest_point(x),
            get_highest_point(x),
        ]
    )


# distance functions
def distance(x1: np.array, x2: np.array, d_func: Callable) -> float:
    """Returns distance according to specified distance measure.

    Args:
        y1, y2: y-values (important: need to be sampled at fixed x-grid so that point in y match!)
        d_func: distance function (symmetric)

    Returns:
        distance between prediction and data.
    """

    return d_func(x1[x1 > 0], x2[x1 > 0])


# choose a distance function ...
# mse = lambda x1, x2: np.square(np.subtract(x1, x2)).mean()
# chebyshev = lambda x1, x2: np.max(np.abs(np.subtract(x1, x2)))


# plotting function to compare two throws
def plot_throw2(
    x_o,
    y_o,
    x,
    y,
    color_1="black",
    color_2="red",
    grid=True,
    legend=True,
    ax=None,
):
    """Return axis comparing simulation output to observations. Show obs. error.

    Creates axis if not passed.
    Args:
        x_o: distance travelled of sim1
        y_o: hight of sim1
        x: distance travelled of sim2
        y: hight of sim2
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x_o, y_o, label="$x_\mathrm{o}$", marker=".", color=color_1, s=10)
    ax.scatter(x, y, label="s", marker="o", color=color_2, s=15, facecolors="none")

    title = f"MSE: {np.square(np.subtract(y_o[y_o>0], y[y_o>0])).mean():.2f}"
    plt.title(title)

    if legend:
        ax.legend()
    ax.grid(True)

    return ax


if __name__ == "__main__":
    # run two versions of simulator to "ground truth" data

    velocity = 35
    angle = 47
    drag = 0.13
    sim = throw(velocity, angle, drag, h_noise=1.5)  # run the simulation with noise
    data_dict = dict(d=sim[0], y=sim[1])
    with open("throw-x_o-1.pickle", "wb") as f:
        pickle.dump(data_dict, f)

    velocity = 47
    angle = 32
    drag = 0.21
    sim = throw(velocity, angle, drag, h_noise=1.5)  # run the simulation with noise
    data_dict = dict(d=sim[0], y=sim[1])
    with open("throw-x_o-2.pickle", "wb") as f:
        pickle.dump(data_dict, f)
