#PROJECT 2: ELECTROSTATICS IN SHIELDED BOXES

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# PART A: Grid and Poisson solver

def create_grid(N, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    """
    Create a square-spaced two-dimensional grid.
    N is the number of intervals in each direction.
    The arrays X and Y store x and y-coordinates at each grid point.
    Array indexing is [y, x].
    """

    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 2:
        raise ValueError("N must be an integer of at least 2.")
    
    # create equally spaced coordinates
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("The domain limits must be increasing.")
    
    x = np.linspace(x_min, x_max, N + 1)
    y = np.linspace(y_min, y_max, N + 1)

    X, Y = np.meshgrid(x, y)

    # the five-point finite-difference formula assumes equal spacing
    hx = (x_max - x_min) / N
    hy = (y_max - y_min) / N

    if not np.isclose(hx, hy):
        raise ValueError("This solver requires equal grid spacing in x and y.")
    h =hx

    return x, y, X, Y, h

def solve_poisson(rho, dirichlet_mask, boundary_values, h,
                  method="jacobi", omega=1.5, tol=1e-6,
                  max_sweeps=200000):
    """Solve -Laplace(V) = rho using Jacobi, red-black Gauss-Seidel, or red-black SOR
    
    Parameters
    -------------
    rho : 2-D array
        Charge density at grid points.
    dirichlet_mask : 2-D Boolean array
        True at nodes whose potential is prescribed.
    boundary_values : 2-D array
        Prescribed potential values at Dirichlet nodes.
    h : float
        Uniform grid spacing.
    method : str
        'jacobi', 'gs', or 'sor'.
    omega : float
        SOR relaxation parameter, with 0 < omega < 2.
    tol : float
        Required residual infinity-norm tolerance.
    max_sweeps : int
        Maximum number of complete iterations.

    Returns 
    --------------
    V : 2-D array
        Computed electric potential.
    sweep : int
        Number of complete sweeps performed.
    residual_history : list
        Residual infinity norm after each sweep.
    """

    # convert inputs to consistent array types
    rho = np.asarray(rho, dtype=float)
    fixed = np.asarray(dirichlet_mask, dtype=bool)
    boundary_values = np.asarray(boundary_values, dtype=float)

    # validate array dimensions and matching shapes
    if rho.ndim != 2 or min(rho.shape) < 3:
        raise ValueError("rho must be a 2-D array at least 3 by 3.")
    
    if fixed.shape != rho.shape or boundary_values.shape != rho.shape:
        raise ValueError(
            "rho, dirichlet_mask and boundary_values must have equal shapes."
        )
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho must contain only finite values.")
    if not np.all(np.isfinite(boundary_values)):
        raise ValueError("Prescribed boundary values must be finite.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number.")
    if not isinstance(max_sweeps, (int, np.integer)) or isinstance(max_sweeps, bool):
        raise ValueError("max_sweeps must be a positive integer.")
    
    # validate selected iterative method
    method = method.lower()

    if method not in ("jacobi", "gs", "sor"):
        raise ValueError("method must be 'jacobi', 'gs', or 'sor'.")
    
    if method == "sor" and (
        not np.isfinite(omega) or not 0.0 < omega < 2.0
    ):
        raise ValueError("SOR requires 0 < omega < 2.")
    
    # initialise unknown potentials to zero and impose fixed values
    V = np.zeros_like(rho)
    V[fixed] = boundary_values[fixed]
    
    # only non-dirichlet nodes inside the array are updated
    active = ~fixed
    active_inner = active[1:-1, 1:-1]

    # red-black ordering allows vectorised Gauss-Seidel and SOR updates
    rows, cols = np.indices(V.shape)
    red = ((rows[1:-1, 1:-1] + cols[1:-1, 1:-1]) % 2) == 0
    black = ~red
    
           
    def neighbour_average(array):
        """
        Apply the five-point update for -Laplace(V) = rho.
        V_ij = (V_up + V_down + V_left + V_right
                + h^2 * rho_ij) / 4
        """

        return 0.25 * (
            array[2:, 1:-1] 
            + array[:-2, 1:-1]
            + array[1:-1, 2:] 
            + array[1:-1, :-2]
            + h**2 * rho[1:-1, 1:-1]
        )
    
    def residual_norm():
        """
        Calculate ||rho + Laplace(V) ||_infinity over non_Dirichlet interior nodes only.
        """
        laplacian = (
            V[2:, 1:-1] 
            + V[:-2, 1:-1]
            + V[1:-1, 2:] 
            + V[1:-1, :-2]
            -4.0 * V[1:-1, 1:-1]
        ) / h**2

        residual = rho[1:-1, 1:-1] + laplacian
        if not np.any(active_inner):
            return 0.0
        
        return float(np.max(np.abs(residual[active_inner])))
    
    # store residual after each complete sweep
    residual_history = []

    for sweep in range(1, max_sweeps + 1):
        if method == "jacobi":
            # Jacobi uses only values from previous sweep
            old = V.copy()
            average = neighbour_average(old)

            interior = V[1:-1, 1:-1]
            interior[active_inner] = average[active_inner]

        else:
            # Gauss-Seidel uses omega = 1; SOR uses the chose relaxation factor
            relaxation = omega if method == "sor" else 1.0

            # update red nodes, then black nodes using latest values
            for color_mask in (red, black):
                average = neighbour_average(V)
                interior = V[1:-1, 1:-1]
                update = active_inner & color_mask

                interior[update] = (
                    (1.0 - relaxation) * interior[update]
                    + relaxation * average[update]
                )
        
        # test the actual discrete-equation residual, not potential change
        rnorm = residual_norm()
        residual_history.append(rnorm)

        if rnorm <= tol:
            return V, sweep, residual_history

    # report failure rather than returning an unconverged solution    
    raise RuntimeError(
        f"{method} failed to reach residual <= {tol:g} "
        f"within {max_sweeps} sweeps."
        f"Final residual = {residual_history[-1]:.6e}"
    )

# Geometry Definitions

def geometry_g1(N):
    """G1: Gaussian charge cloud in a grounded unit square."""
    x, y, X, Y, h = create_grid(N)
    rho0, sigma = 50.0, 0.05
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
    domain = ~((X > 1e-12) & (Y < 1e-12))
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
    """Plot the electric potential, equipotential contours, and electric field."""

    # hide grid points outside the physical domain, particularily for G3
    vplot = np.where(domain, V, np.nan)
    
    # calculate the electric field using E = -grad(V)
    h = X[0, 1] - X[0, 0]
    dV_dy, dV_dx = np.gradient(V, h, h)
    Ex = -dV_dx
    Ey = -dV_dy
    
    Explot = np.where(domain, Ex, np.nan)
    Eyplot = np.where(domain, Ey, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))

    # plot the potential colour map
    mesh = ax.pcolormesh(
        X, Y, vplot,
        shading="auto",
        cmap="viridis"
    )
    fig.colorbar(mesh, ax=ax, label="Electric potential V")

    # add equipotential contours when potential is not constant
    valid_values = vplot[np.isfinite(vplot)]
    if valid_values.size > 0:
        vmin = np.min(valid_values)
        vmax = np.max(valid_values)

        if vmin < vmax:
            levels = np.linspace(vmin, vmax, 15)
            contours = ax.contour(
                X, Y, vplot,
                levels=levels,
                colors="white",
                linewidths=0.7
            )
            ax.clabel(contours, inline=True, fontsize=8)

    # add electric-field arrows 
    step = max(1, X.shape[0] // 20)
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step], 
        Explot[::step, ::step],
        Eyplot[::step, ::step],
        color="red",
        pivot="mid"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    # Save and display figure
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_g2_diagonal_field(X, Y, V, filename):
    """Plot the elctric-field magnitude along the diagonal x = y for G2."""
    
    # grid spacing
    h = X[0, 1] - X[0, 0]

    # calculate the electric field where E -grad(V)
    dV_dy, dV_dx = np.gradient(V, h, h)
    Ex = -dV_dx
    Ey = -dV_dy

    # extract diagonal grid points where x = y
    diagonal_indices = np.arange(min(X.shape))
    diagonal_x = X[diagonal_indices, diagonal_indices]

    # calculate |E| = sqrt(Ex^2 + Ey^2) along the diagonal
    diagonal_magnitude = np.sqrt(
        Ex[diagonal_indices, diagonal_indices]**2
        + Ey[diagonal_indices, diagonal_indices]**2
    )

    # plot electric-field magnitude 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diagonal_x, diagonal_magnitude, linewidth=2)

    ax.set_xlabel("Position along diagonal, x = y")
    ax.set_ylabel(r"Electric-field magnitude $|\mathbf{E}|$")
    ax.set_title("G2: Electric-field magnitude along x = y")
    ax.grid(True)
    fig.tight_layout()

    # save and display figure
    fig.savefig("g2_diagonal_field.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def grid_refinement_study():
    """Compare convergence rates for G1 across required grid sizes."""

    grid_sizes = [20, 40, 80, 160]
    
    jacobi_sweeps = []
    gs_sweeps = []
    sor_sweeps = []

    for N in grid_sizes:
        print(f"\nGrid-refinement study: N = {N}")

        x, y, X, Y, h, rho, fixed, values = geometry_g1(N)

        # calculate optimal SOR relaxation parameter
        omega_opt = 2.0 / (1.0 + np.sin(np.pi / N))

        # run every solver on current grid
        for method in ("jacobi", "gs", "sor"):
            omega = omega_opt if method == "sor" else 1.0

            V, sweeps, residual_history = solve_poisson(
                rho,
                fixed,
                values,
                h,
                method=method,
                omega=omega,
                tol=1e-6
            )
            
            print(
                f"    {method.upper():6s}: "
                f"{sweeps} sweeps, "
                f"final residual = {residual_history[-1]:.3e}"
            )

            # store each method's sweep count for current grid
            if method == "jacobi":
                jacobi_sweeps.append(sweeps)
            elif method == "gs":
                gs_sweeps.append(sweeps)
            else:
                sor_sweeps.append(sweeps)

    # plot the sweep counts against grid size on logarithmic axes
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(grid_sizes, jacobi_sweeps, "o-", label="Jacobi")
    ax.loglog(grid_sizes, gs_sweeps, "s-", label="Gauss-Seidel")
    ax.loglog(grid_sizes, sor_sweeps, "^-", label="SOR (optimal omega)")

    # estimate scaling exponent from each log-log curve
    for sweeps, label in (
        (jacobi_sweeps, "Jacobi"),
        (gs_sweeps, "Gauss-Seidel"),
        (sor_sweeps, "SOR"),
    ):
        slope, intercept = np.polyfit(
            np.log(grid_sizes),
            np.log(sweeps),
            1
        )
        print(f"{label} log-log slope: {slope:.3f}")

    ax.set_xlabel("Grid size N")
    ax.set_ylabel("Number of sweeps")
    ax.set_title("G1: Solver grid-refinement study")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig("g1_grid_refinement.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "grid_sizes": grid_sizes,
        "jacobi_sweeps": jacobi_sweeps,
        "gs_sweeps": gs_sweeps,
        "sor_sweeps": sor_sweeps
    }

# PART C: Code Validation and Testing

# TEST 1: Exact Polynomial on the Multiply Connected G2 Domain
def test_polynomial_exactness():
    """Validate all three methods against an exact polynomial solution."""

    N = 20

    def exact_potential(X, Y):
        return X**3 + 2.0 * Y**2 + X * Y
    def exact_rho(X, Y):
        return -(6.0 * X + 4.0)
    
    x, y, X, Y, h, rho, fixed, values, conductor = geometry_g2(
        N,
        rho_override=exact_rho,
        boundary_function=exact_potential
    )
    V_exact = exact_potential(X, Y)
    
    #verify the prescribed boundary values match the exact solution
    assert np.allclose(values[fixed], V_exact[fixed]), (
        "The exact Dirichlet boundary values are incorrect."
    )

    print("\nTEST 1: Exact polynomial on G2")
    results = {}

    for method in ("jacobi", "gs", "sor"):
        omega = (
            2.0 / (1.0 + np.sin(np.pi / N))
            if method == "sor"
            else 1.0
        )

        V, sweeps, residual_history = solve_poisson(
            rho,
            fixed,
            values,
            h,
            method=method,
            omega=omega,
            tol=1e-9
        )

        error = np.max(np.abs(V - V_exact))
        final_residual = residual_history[-1]

        results[method] = error

        print(
            f"TEST 1 {method}: "
            f"max error = {error:.3e}, "
            f"sweeps = {sweeps}, "
            f"residual = {final_residual:.3e}"
        )
        
    return results

# TEST 2: Independent Reference Solution

def test_sparse_matrix_comparison(N=20, tol=1e-6):
    """Compare iterative solutions with an independent sparse-matrix solution"""

    results = {}

    #test both the Gaussian charge problem (G1) and the coaxial conductor (G2)
    for geometry_name in ("G1", "G2"):

        if geometry_name == "G1":
            x, y, X, Y, h, rho, fixed, values = geometry_g1(N)
        else:
            x, y, X, Y, h, rho, fixed, values, conductor = geometry_g2(N)

        free = ~fixed
        free_indices = -np.ones(fixed.shape, dtype=int)
        free_indices[free] = np.arange(np.count_nonzero(free))
        number_free = np.count_nonzero(free)

        A = lil_matrix((number_free, number_free))
        b = (rho * h**2)[free].copy()

        rows, cols = np.where(free)

        for row, col in zip(rows, cols):
            equation_row = free_indices[row, col]

            A[equation_row, equation_row] = 4.0

            for neighbour_row, neighbour_col in (
                (row -1, col), 
                (row + 1, col),
                (row, col -1),
                (row, col + 1),
            ):
                if (
                    0 <= neighbour_row < fixed.shape[0]
                    and 0 <= neighbour_col < fixed.shape[1]
                ):
                    if free[neighbour_row, neighbour_col]:
                        neighbour_index = free_indices[
                            neighbour_row, neighbour_col
                        ]
                        A[equation_row, neighbour_index] = -1.0
                    else:
                        b[equation_row] += values[
                            neighbour_row, neighbour_col
                        ]
        V_direct = values.copy()
        V_direct[free] = spsolve(A.tocsr(), b)

        results[geometry_name] = {}

        for method in ("jacobi", "gs", "sor"):
            omega = 1.5 if method == "sor" else 1.0

            V_iterative, sweeps, residual_history = solve_poisson(
                rho, fixed, values, h,
                method=method,
                omega=omega,
                tol=tol
            )
            max_difference = np.max(
                np.abs(V_iterative[free] - V_direct[free])
            )
            assert max_difference < 1e-6, (
                f"{geometry_name} {method}: difference from sparse reference is too large"
            )

            results[geometry_name][method] = {
                "sweeps": sweeps,
                "max_difference": max_difference,
                "residual": residual_history[-1],
            }

            print(
                f"\n{geometry_name}, {method.upper()}"
                f"sweeps = {sweeps}, "
                f"maximum difference = {max_difference:.6e}, "
                f"residual = {residual_history[-1]:.6e}"
            )
    return results

# TEST 3: Stopping Criterion

def solve_poisson_max_change(
        rho, dirichlet_mask, boundary_values, h,
        method="gs", omega=1.5, tol=1e-6, max_sweeps=200000
):
    """Solve Poisson's equation using maximum potential change as stopping criterion"""
    method = method.lower()
    if method not in ("jacobi", "gs", "sor"):
        raise ValueError("method must be 'jacobi', 'gs', or 'sor'.")
    
    V = np.zeros_like(rho, dtype=float)
    V[dirichlet_mask] = boundary_values[dirichlet_mask]

    # only update free interior grid points
    active = ~dirichlet_mask
    active[0, :] = False
    active[-1, :] = False
    active[:, 0] = False
    active[:, -1] = False

    if not np.any(active):
        return V, 0, 0.0

    # red-black masks for Gauss-Seidel and SOR
    rows, cols = np.indices(V.shape)
    red = (rows + cols) % 2 == 0
    black = ~red

    for sweep in range(1, max_sweeps + 1):
        V_old = V.copy()

        if method == "jacobi":
            V_new = V.copy()
            V_new[1:-1, 1:-1] = 0.25 * (
                V_old[2:, 1:-1] 
                + V_old[:-2, 1:-1]
                + V_old[1:-1, 2:]
                + V_old[1:-1, :-2]
                + h**2 * rho[1:-1, 1:-1]
            )
            V[active] = V_new[active]
        else:
            for color in (red, black):
                update_mask = active & color

                V_star = 0.25 * (
                    np.roll(V, 1, axis=0)
                    + np.roll(V, -1, axis=0)
                    + np.roll(V, 1, axis=1)
                    + np.roll(V, -1, axis=1)
                    + h**2 * rho
                )
                if method == "gs":
                    V[update_mask] = V_star[update_mask]
    
                else: 
                    V[update_mask] = (
                        (1.0 - omega) * V[update_mask]
                        + omega * V_star[update_mask]
                    )
        max_change = np.max(np.abs(V[active] - V_old[active]))

        if not np.isfinite(max_change):
            raise RuntimeError(
                f"{method} produced a non-finite potential change"
                f"at sweep {sweep}."
            )
        if max_change < tol:
            return V, sweep, max_change
    raise RuntimeError(
        f"{method} did not converge within {max_sweeps} sweeps."
    )

def test_alternative_stopping_criterion(N=40, tol=1e-6):
    """Compare maximum-change stopping with residual-based stopping for G1"""
    x, y, X, Y, h, rho, fixed, values = geometry_g1(N)

    results = {}

    for method in ("jacobi", "gs", "sor"):
        omega = (
            2.0 / (1.0 + np.sin(np.pi / N))
            if method == "sor"
            else 1.0
        )
        V_residual, residual_sweeps, residual_history = solve_poisson(
            rho, fixed, values, h,
            method=method,
            omega=omega,
            tol=tol 
        )

        V_change, change_sweeps, final_change = solve_poisson_max_change(
            rho, fixed, values, h,
            method=method, 
            omega=omega, 
            tol=tol 
        )
 
        residual_grid = np.zeros_like(V_change)

        residual_grid[1:-1, 1:-1] = (
            rho[1:-1, 1:-1] 
            + (
                V_change[2:, 1:-1] 
                + V_change[:-2, 1:-1]
                + V_change[1:-1, 2:] 
                + V_change[1:-1, :-2]
                - 4.0 * V_change[1:-1, 1:-1]
            ) / h**2
        )

        active_inner = (~fixed)[1:-1, 1:-1]
        actual_residual = np.max(np.abs(residual_grid[1:-1, 1:-1][active_inner]))

        #compare two final potential arrays
        difference = np.max(np.abs(V_change - V_residual))

        results[method] = {
            "residual_sweeps": residual_sweeps,
            "change_sweeps": change_sweeps,
            "maximum_change": final_change,
            "actual_residual": actual_residual,
            "solution_difference": difference,
        }

        print(f"\nG1 alternative stopping test: {method.upper()}")
        print(f"Residual-based sweeps: {residual_sweeps}")
        print(f"Maximum-change sweeps: {change_sweeps}")
        print(f"Final maximum change: {final_change:.6e}")
        print(f"Actual residual: {actual_residual:.6e}")
        print(f"Maximum solution difference: {difference:.6e}")
    
    return results

#TEST 4: Order of Accuracy

def test_exponential_convergence():
    """Test the spatial accuracy of the solver using an exact exponential solution"""

    grid_sizes = [8, 16, 32, 64]
    grid_spacings = []
    maximum_errors = []

    for N in grid_sizes:
        x, y, X, Y, h = create_grid(N)
        V_exact = np.exp(X + 2.0 * Y)

        rho = -5.0 * V_exact

        dirichlet_mask = np.zeros_like(X, dtype=bool)
        dirichlet_mask[0, :] = True
        dirichlet_mask[-1, :] = True
        dirichlet_mask[:, 0] = True
        dirichlet_mask[:, -1] = True

        boundary_values = np.zeros_like(X)
        boundary_values[dirichlet_mask] = V_exact[dirichlet_mask]

        omega_opt = 2.0 / (1.0 + np.sin(np.pi / N))

        V_numerical, sweeps, residual_history = solve_poisson(
            rho,
            dirichlet_mask,
            boundary_values,
            h,
            method="sor",
            omega=omega_opt,
            tol=1e-6
        )

        max_error = np.max(np.abs(V_numerical - V_exact))

        grid_spacings.append(h)
        maximum_errors.append(max_error)

        print(f"\nExponential convergence test: N = {N}")
        print(f"Grid spacing h: {h:.6e}")
        print(f"SOR sweeps: {sweeps}")
        print(f"Final residual: {residual_history[-1]:.6e}")
        print(f"Maximum absolute error: {max_error:.6e}")

    slope, intercept = np.polyfit(
        np.log(grid_spacings),
        np.log(maximum_errors),
        1
    )
    print(f"\nEstimated convergence order: {slope:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        grid_spacings,
        maximum_errors,
        "o-",
        label="Measured maximum error"
    )

    #reference line proportional to h squared
    reference_errors = (
        maximum_errors[0]
        * (np.array(grid_spacings) / grid_spacings[0])**2
    )
    ax.loglog(
        grid_spacings,
        reference_errors,
        "--",
        label=r"Refrence slope: $h^2$"
    )

    ax.set_xlabel("Grid spacing h")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title("Exponential solution: spatial convergence")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout() 
    fig.savefig(
        "exponential_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    return{
        "grid_spacings": grid_spacings,
        "maximum errors": maximum_errors,
        "convergence_order": slope,
    }
        

# =================================================
# Temporary test: compare all three Poisson solvers
# =================================================
if __name__ == "__main__":
    print("Testing geometry definitions...")

    # G1: Grounded square with a Gaussian charge cloud
    x, y, X, Y, h, rho, fixed, values = geometry_g1(80)

    print("\nG1: Gaussian charge")
    print("Grid shape:", X.shape)
    print("Grid spacing:", h)
    print("Charge density is finite:", np.all(np.isfinite(rho)))
    print("All fixed boundary values are zero:",
          np.allclose(values[fixed], 0.0))
    
    # G2: Grounded outer square with an inner conductor
    x, y, X, Y, h, rho, fixed, values, conductor = geometry_g2(80)

    print("\nG2: Square conductor")
    print("Grid shape:", X.shape)
    print("Grid spacing:", h)
    print("Charge density is zero:", np.allclose(rho, 0.0))
    print("Conductor potential is 1:",
          np.allclose(values[conductor], 1.0))
    print("Outer boundary potential is zero:",
          np.allclose(values[0, :], 0.0)
          and np.allclose(values[-1, :], 0.0)
          and np.allclose(values[:, 0], 0.0)
          and np.allclose(values[:, -1], 0.0))
    
    # G3: L-Shaped domain
    x, y, X, Y, h, rho, fixed, values, domain = geometry_g3(80)

    print("\nG3: L-shaped domain")
    print("Grid shape:", X.shape)
    print("Grid spacing:", h)
    print("Charge density is zero:", np.allclose(rho, 0.0))
    print("Removed quadrant excluded:",
          not np.any(domain[(X > 1e-12) & (Y < 1e-12)]))
    print("All fixed values are finite:",
          np.all(np.isfinite(values[fixed])))
    
    print("\nGeometry checks completed.")
    
    # ===============================================
    # PART A: Test the Poisson solver on G1 using SOR
    # ===============================================
    print("\n==============PART A: Testing the Poisson solver on G1=================")

    x, y, X, Y, h, rho, fixed, values = geometry_g1(20)
    omega_opt = 2.0 / (1.0 + np.sin(np.pi / 20))
    V, sweeps, residual_history = solve_poisson(
        rho,
        fixed,
        values,
        h,
        method="sor",
        omega=omega_opt,
        tol=1e-6
    )
    print("Solver converged:", residual_history[-1] <= 1e-6)
    print("Number of sweeps:", sweeps)   
    print("Final residual:", f"{residual_history[-1]:.6e}")
    print("Potential values are finite:", np.all(np.isfinite(V)))
    print("Dirichlet values preserved:", 
          np.allclose(V[fixed], values[fixed]))
    

    # =========================================================
    #  PART B: Plot the numerical solutions for G!, G2, and G3
    # =========================================================
    print("\n===============PART B: Generating potential and electric-field plots=================")

    #G1: GAussian charge distribution
    x, y, X, Y, h, rho, fixed, values = geometry_g1(80)
    V_g1, sweeps, residual_history = solve_poisson(
        rho, fixed, values, h, method="sor",
        omega=2.0 / (1.0 + np.sin(np.pi / 80)),
        tol=1e-6
    )
    plot_solution(
        X, Y, V_g1, np.ones_like(fixed, dtype=bool),
        "G1: Gaussian charge distribution",
        "g1_potential.png"
    )

    #G2 : Square conductor
    x, y, X, Y, h, rho, fixed, values, conductor = geometry_g2(80)
    V_g2, sweeps, residual_history = solve_poisson(
        rho, fixed, values, h, method="sor",
        omega=2.0 / (1.0 + np.sin(np.pi / 80)),
        tol=1e-6
    )
    plot_solution(
        X, Y, V_g2, np.ones_like(fixed, dtype=bool),
        "G2: Square conductor",
        "g2_potential.png"
    )


    print (
        f"G2: SOR sweeps={sweeps, }"
        f"Final residual={residual_history[-1]:.3e}"
    )
    #G2: Electric-field magnitude along x = y
    dV_dy, dV_dx = np.gradient(V_g2, h, h)
    Ex = -dV_dx
    Ey = -dV_dy
    n = min(X.shape[0], X.shape[1], Ex.shape[0], Ex.shape[1],
            Ey.shape[0], Ey.shape[1])
    indices = np.arange(n)
    diagonal_x = X[indices, indices]
    diagonal_magnitude = np.hypot(
        Ex[indices, indices],
        Ey[indices, indices]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(diagonal_x, diagonal_magnitude)
    ax.set_xlabel("Position along diagonal, x = y")
    ax.set_ylabel(r"Electric-field magnitude $|\mathbf{E}|$")
    ax.set_title("G2: Electric-field magnitude along x = y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("g2_diagonal_field.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("G2 Diagonal electric-field plot saved.")


    #G3: L-Shaped domain
    x, y, X, Y, h, rho, fixed, values, domain = geometry_g3(80)
    V_g3, sweeps, residual_history = solve_poisson(
        rho, fixed, values, h, method="sor",
        omega=2.0 / (1.0 + np.sin(np.pi / 80)),
        tol=1e-6
    )
    plot_solution(
        X, Y, V_g3, domain,
        "G3: L-shaped domain",
        "g3_potential.png"
    )
    # compare solver convergence as the grid is refined
    print("\nRunning the grid-refinement study...")
    grid_refinement_study()

    # ===================================
    # PART C: Code Validation and Testing
    # ===================================
    print("\n=======PART C: Validation Tests =============")

    # Test 1: Polynomial Exactness
    test_polynomial_exactness()

    # Test 2: Compare iterative solution with sparse reference solution
    test_sparse_matrix_comparison()

    # Test 3: Compare alternative stopping criteria
    test_alternative_stopping_criterion()

    # Test 4: Check convergence for exponential exact solution
    test_exponential_convergence()







