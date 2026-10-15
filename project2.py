#PROJECT 2: ELECTROSTATICS IN SHIELDED BOXES

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# PART A: Grid and Poisson solver

def create_grid(N, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    """
    Return z, y, X, Y, h for a rectangular grid with N intervals in x."""

    if N < 1:
        raise ValueError("N must be a positive integer.")
    x = np.linspace(x_min, x_max, N + 1)
    y = np.linspace(y_min, y_max, N + 1)
    X, Y = np.meshgrid(x, y)
    h = (x_max - x_min) / N
    if not np.isclose((y_max - y_min) / N, h):
        raise ValueError("This solver requires equal grid spacing in x and y.")
    return x, y, X, Y, h

def solve_poisson(rho, dirichlet_mask, boundary_values, h,
                  method="jacobi", omega=1.5, tol=1e-6,
                  max_sweeps=200000):
    """Solve -Laplace(V)=rho using Jacobi, red-black GS, or red-black SOR.
    
    The arrays are indeed [y, x]. Dirichlet nodes are held fixed. 
    The returned residual history contains ||rho + Laplace(V)||_infinity afetr every complete sweep,
    evaluated only at non-Dirichlet interior nodes.
    """
    rho = np.asarray(rho, dtype=float)
    fixed = np.asarray(dirichlet_mask, dtype=bool)
    boundary_values = np.asarray(boundary_values, dtype=float)
    if rho.ndim != 2 or rho.shape[1] < 3:
        raise ValueError("rho, dirichlet_mas, and boundary_values must have equal 2-D shapes.")
    if rho.shape[0] < 3 or rho.shape[1] < 3:
        raise ValueError("The grid must be at least 3 by 3.")
    if h <= 0 or tol <= 0 or max_sweeps < 1:
        raise ValueError("h, tol, and max_sweeps must be positive.")
    method = method.lower()
    if method not in ("jacobi", "gs", "sor"):
        raise ValueError("method must be 'jacobi, 'gs', or 'sor'.")
    if method == "sor" and not (0.0 < omega < 2.0):
        raise ValueError("SOR requies 0 < omega < 2.")
    
    V = np.zeros_like(rho)
    V[fixed] = boundary_values[fixed]
    active = ~fixed
    active_inner = active[1:-1, 1:-1]
    rows, cols = np.indices(V.shape)
    red = ((rows[1:-1, 1:-1] + cols[1:-1, 1:-1]) % 2) == 0
           
    def neighbour_average(array):
        return 0.25 * (
            array[2:, 1:-1] + array[:-2, 1:-1]
            + array[1:-1, 2:] + array[1:-1, :-2]
            + h**2 * rho[1:-1, 1:-1]
        )
    
    def residual_norm():
        lap = (
            V[2:, 1:-1] + V[:-2, 1:-1]
            + V[1:-1, 2:] + V[1:-1, :-2]
            -4.0 * V[1:-1, 1:-1]
        ) / h**2
        r = rho[1:-1, 1:-1] + lap
        return float(np.max(np.abs(r[active_inner]))) if np.any(active_inner) else 0.0
    

    residual_history = []
    for sweep in range(1, max_sweeps + 1):
        if method == "jacobi":
            old = V.copy()
            avg = neighbour_average(old)
            interior = V[1:-1, 1:-1]
            interior[active_inner] = avg[active_inner]
        else:
            relaxation = omega if method == "sor" else 1.0
            for colour_masl in (red, ~red):
                avg = neighbour_average(V)
                interior = V[1:-1, 1:-1]
                update = active_inner & colour_mask
                interior[update] = (
                    (1.0 - relaxation) * interior[update]
                    + relaxation * avg[update]
                )
        
        rnorm = residual_norm()
        residual_history.append(rnorm)
        if rnorm <= tol:
            return V, sweep, residual_history
        
    raise RuntimeError(
        f"{method} failed to reach residual <= {tol:g} in {max_sweeps} sweeps; "
        f"final residual={residual_history[-1]: .6e}"
    )

# Geometry Definitions

def geometry_g1(N):
    """G1: Gaussian charge cloud in a grounded unit square."""
    x, y, X, Y, h = create_grid(N)
    rho0, sigms = 50.0, 0.05
    rho = rho0 * np.exp(-((X - 0.3)**2 + (Y - 0.6)**2) / (2.0 * sigma**2))
    fixed = np.zeros_like(X, dtype=bool)
    fixed[0, :] = fixed[-1, :] = True
    fixed[:, 0] = fixed[:, -1] = True
    values =  np.zeros_like(X)
    return x, y, X, Y, h, rho, fixed, values

def geometry_g2(N, rho_override=None, boundary_function=None):
    """G2: grounded outer square and square conductor [0.3, 0.7]^2"""
    if N % 10 != 0:
        raise ValueError("G2 requires N to be a multiple of 10.")
    x, y, X, Y, h = create_grid(N)
    rho = np.zeros_like(X) if rho_override is None else np.asarray(rho_override(X, Y), dtype=float)
    fixed = np.zeros_like(X, dtype=bool)
    values = np.zeros_like(X)

    #Grounded outer boundary
    fixed[0, :] = fixed[-1, :] = True
    fixed[:, 0] = fixed[:, -1] = True
    #the whole interior of the grid-aligned conductor is held at a fixed V.
    conductor = (X >= 0.3 - 1e-12) & (X <= 0.7 + 1e-12) & (Y >= 0.3 - 1e-12) & (Y <= 0.7 + 1e-12)
    fixed[conductor] = True
    if boundary_function is None:
        values[conductor] = 1.0
    else:
        values[conductor] = boundary_function(X[conductor], Y[conductor])
        outer = np.zeros_like(X, dtype=bool)
        outer[0, :] = outer[-1, :] = True
        outer[:, 0] = outer[:, -1] = True
        values[outer] = boundary_function(X[outer], Y[outer])
    return x, y, X, Y, h, rho, fixed, values, conductor

def exact_g3(X, Y):
    """Exact boundary potential r^(2/3) sin(2 theta/3), theta in [0,3pi/2]."""
    theta = np.mod(np.arctan2(Y, X), 2.0 * np.pi)
    #in the L-shaped domain, angles are represented in [0, 3pi/2].
    r = np.hypot(X, Y)
    return np.where(r == 0.0, 0.0, r**(2.0 / 3.0) * np.sin(2.0 * theta / 3.0))

def geometry_g3(N):
    """G3: L-shaped domain[-1, 1]^2 excluding x>0,y<0; h=2/N."""
    x, y, X, Y, h = create_grid(N, -1.0, 1.0, -1.0, 1.0)
    rho = np.zeros_like(X)
    domain = ~((X > 1e-12) & (Y < 1e12))
    fixed = ~domain #excluded quadrant is fixed and never solved
    values = np.zeros_like(X)
    #outer walls: x=-1, y=1, x=1 on upper hald, y=-1 on left half
    outer = ((np.isclose(X, -1.0)) | (np.isclose(Y, 1.0))
            | (np.isclose(Y, -1.0) & (Y >= -1e-12))
            | (np.isclose(Y, -1.0) & (X <= 1e-12)))
    #the two walls adjacent to the removed quadrant are grounded
    inner_ground = ((np.isclose(Y, 0.0) & (X >= -1e-12))
                    | (np.isclose(X, 0.0) & (Y <= 1e-12)))
    fixed |= outer | inner_ground
    exact = exact_g3(X, Y)
    values[outer] = exact[outer]
    values[inner_ground] = 0.0
    return x, y, X, Y, h, rho, fixed, values, domain

# PART B: PLOTTING

def plot_solution(X, Y, V, domain, title, filename):
    """Plot potential, equipotentials, and electric-field arrows."""
    vplot = np.where(domain_mask, V, np.nan)
    Ey, Ex = np.gradient(-V, Y[:, 0], X[0, :])
    Ex = np.where(domain_mask, Ex, np.nan)
    Ey = np.where(domain_mask, Ey, np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolourmesh(X, Y, Vplot, shading="auto", cmap="viridis")
    fig.colourbar(mesh, ax=ax, label="Potential V")
    finite = Vplot[np.isfinite(Vplot)]
    if finite.size and np.nanmax(finite) > np.nanmin(finite):
        levels = np.linspace(np.nanmin(finite), np.nanmax(finite), 12)
        ax. contour(X, Y, Vplot levels=levels, colours="white", linewidth=0.7)
    stride = max(1, X.shape[0] // 22)
    ax.quiver(X[::stride, ::stride], Y[::strde, ::stride],
              Ex[::stride, ::stride], Ey[::stride, ::stride],
              colour="white", pivot="mid", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    

                                   









