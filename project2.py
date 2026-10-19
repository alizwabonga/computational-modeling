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
            for colour_mask in (red, ~red):
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
    """Plot potential, equipotentials, and electric-field arrows."""
    # Hide points outside the physical domain, especially for G3
    vplot = np.where(domain, V, np.nan)
    
    #Calculate the electric field: E = -grad(V)
    h = X[0, 1] - X[0, 0]
    dV_dy, dV_dx = np.gradient(V, h, h)
    Ex = -dV_dx
    Ey = -dV_dy
    
    Explot = np.where(domain, Ex, np.nan)
    Eyplot = np.where(domain, Ey, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Potential colourmap
    mesh = ax.pcolourmesh(
        X, Y, vplot,
        shading="auto",
        cmap="viridis"
    )
    fig.colourbar(mesh, ax=ax, label="Electric potential V")

    # Equipotential contours
    valid_values = vplot[np.isfinite(vplot)]
    if valid_values.size > 0:
        vmin = np.min(valid_values)
        vmax = np.max(valid_values)

        if vmin < vmax:
            levels = np.linspace(valid_values)
            contours = ax.contour(
                X, Y, vplot,
                levels=levels,
                colours="white",
                linewidths=0.7
            )
            ax.clabel(contours, inline=True, fontsize=8)

    # Electric-field arrows
    step = max(1, X.shape[0] // 20)
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step], 
        Explot[::step, ::step],
        Eyplot[::step, ::step],
        colour="red",
        pivot="mid"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    # Save and display the figure
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_g2_diagonal_field(X, Y, V, filename):
    """Plot the elctric-field magnitude along the diagonal x = y for G2."""

    # Grid spacing
    h = X[0, 1] - X[0, 0]

    # Calculate the diagonal where x = y
    dV_dy, dV_dx = np.gradient(V, h, h)
    Ex = -dV_dx
    Ey = -dV_dy

    # Extract the diagonal where x = y
    diagonal_indices = np.arrange(min(X.shape))
    diagonal_x = X[diagonal_indices, diagonal_indices]

    diagonal_magnitude = np.sqrt(
        Ex[diagonal_indices, diagonal_indices]**2
        +Ey[diagonal_indices, diagonal_indices]**2
    )

    # Plot the field magnitude 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diagonal_x, diagonal_magnitude, linewidth=2)

    ax.set_xlabel("Position along diagonal, x = y")
    ax.set_ylabel(r"Electric-field magnitude $|\mathbf{E}|$")
    ax.set_title("G2: Electric-field magnitude along x = y")
    ax.grid(True)
    fig.tight_layout()

    # Save and display the figure
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def grid_refinement_study():
    """Compare solver convergence rates for G1 across grid sizes."""

    grid_sizes = [20, 40, 80, 160]
    
    jacobi_sweeps = []
    gs_sweeps = []
    sor_sweeps = []

    for N in grid_sizes:
        print(f"\nGrid-refinement study: N = {N}")

        x, y, X, Y, h, rho, fixed, values = geometry_g1(N)

        # Optimal SOR parameter
        omega_opt = 2.0 / (1.0 + np.sin(np.pi / N))

        # Run all three methods
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

        if method == "jacobi":
            jacobi_sweeps.append(sweeps)
        elif method == "gs":
            gs_sweeps.append(sweeps)
        else:
            sor_sweeps.append(sweeps)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(grid_sizes, jacobi_sweeps, "o-", label="Jacobi")
    ax.loglog(grid_sizes, gs_sweeps, "s-", label="Gauss-Seidel")
    ax.loglog(grid_sizes, sor_sweeps, "-", label="SOR (optimal omega)")

    # Fit and display log-log slopes
    for sweeps, label in (
        (jacobi_sweeps, "Jacobi"),
        (gs_sweeps, "Gauss-Seidel"),
        (sor_sweeps, "SOR"),
    ):
        slope, intercept = np.polyfit(
            np.log(grid_sizes), np.log(sweeps), 1
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

# PART C: Code Validation and Testing

# TEST 1: Exact Polynomial on the Multiply Connected G2 Domain
def test_polynomial_exactness():
    """Validate all three methods against an exact polynomial solution."""

    N = 20

    def exact_potential(X, Y):
        return X**3 + 2.0 * Y**2 + X * Y
    
    x, y, X, Y, rho, fixed, values, conductor = geometry_g2(N)
    V_exact = exact_potential(X, Y)
    rho = -(6.0 * X + 4.0)
    
    #G2 imposes the exact potential on the inner conductor and the outer boundary
    values[fixed] = V_exact[fixed]
    
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

        max_error = float(np.max(np.abs(V - V_exact)))
        final_residual = float(residual_history[-1])

        results[method] = {
            "sweeps": sweeps,
            "final_residual": final_residual,
            "max_error": max_error
        }

        print(
            f"{method.upper():6s} | "
            f"Sweeps: {sweeps:6d} | "
            f"Residual: {final_residual:.6e} | "
            f"Maximum error: {max_error:.6e}"
        )
        #solver meets stopping criterion
        assert final_residual <= 1e-9, (
            f"{method} did not reach residual tolerance."
        )
    print("Test 1 completed for all three methods.")
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
            x, y, X, Y, rho, fixed, values, conductor = geometry_g2(N)

        free = ~fixed
        free_indices = -np.ones(fixed.shape, dtype=int)
        free_indices[free] = np.arrange(np.count_nonzero(free))
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
    """Solve Poisson's equation, stopping when the maximum potential change is small"""
    V = np.zeros_like(rho, dtype=float)
    V[dirichlet_mask] = boundary_values[dirichlet_mask]

    active = ~dirichlet_mask
    active[0, :] = False
    active[-1, :] = False
    active[:, 0] = False
    active[:, -1] = False

    rows, cols = np.indices(V.shape)
    red = (rows + cols) % 2 == 0
    black = ~red

    for sweep in range(1, max_sweeps + 1):
        V_old = V.copy()

        if method == "jacobi":
            V_new = V.cpy()
            V_new[1:-1, 1:-1] = 0.25 * (
                V_old[2:, 1:-1] 
                + V_old[:-2, 1:-1]
                + V_old[1:-1, 2:]
                + V_old[1:-1, :-2]
                + h**2 * rho[1:-1, 1:-1]
            )
            V[active] = V_new[active]

        elif method in ("gs", "sor"):
            for colour in (red, black):
                update_mask = active & colour

                V_star = 0.25 * (
                    np.roll(V, 1, axis=0)
                    + np.roll(V, 1, axis=0)
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
        else:
            raise ValueError("method must be 'jacobi', 'gs', or 'sor'")
        
        #measure the largest change over the complete sweep
        max_change = np.max(np.abs(V - V_old))

        if max_change < tol:
            return V, sweep, max_change
    
    raise RuntimeError(
        f"{method} did npt converge within {max_sweeps} sweeps."
    )

def test_alternative_stopping_criterion(N=40, tol=1e-6):
    """Compare maximum-change stopping with residual-based stopping for G1"""
    x, y, X, Y, h, rho, fixed, values = geometry_g1(N)

    results = {}

    for method in ("jacobi", "gs", "sor"):
        omega = 1.5 if method == "sor" else 1.0

        V_residual, residual_sweeps, residual_history = solve_poisson(
            rho, fixed, values, h,
            method=method, omega=omega, tol=tol 
        )

        V_change, change_sweeps, final_change = solve_poisson_max_change(
            rho, fixed, values, h,
            method=method, omega=omega, tol=tol 
        )

        V = V_change
        residual = rho[1:-1, 1:-1] + (
            V[2:, 1:-1] + V[:-2, 1:-1]
            + V[1:-1, 2:] + V[1:-1, :-2]
            -4.0 * V[1:-1, 1:-1]
        ) / h**2

        active_inner = (~fixed)[1:-1, 1:-1]
        actual_residual = np.max(np.abs(residual[active_inner]))

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
        print(f"Final maxximum change: {final_change:.6e}")
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

        print(f"\nExponrntial convergence test: N = {N}")
        print(f"Grid spacing h: {h:.6e}")
        print(f"SOR sweeps: {sweeps}")
        print(f"Final residual: {residual_history[-1]:.6e}")
        print(f"Maximum absolute error: {max_error:.6e}")

    slope, intercepr = np.polyfit(
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
        bbpx_inches="tight"
    )
    plt.show()
    plt.close(fig)

    return{
        "grid_spacings": grid_spacings,
        "maximum errors": maximum_errors,
        "convergence_order": slope,
    }
        
    






# Temporary test: compare all three Poisson solvers
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
    
    
   









