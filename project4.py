#PROJECT 4: The Gauss Map: Period Bubbles, Hysteresis and Root Finding

import numpy as np
import matplotlib.pyplot as plt

# ==================================== PART A: Implemetation ==========================================

# 1. Parameters
a = 4.9
# control parameter range
B_min = -1.1
B_max = 1.0

# 2. Gauss Map
def gauss_map(x, B, a=4.9):
    return np.exp(-a * x**2) + B

def gauss_derivative(x, a=4.0):
    return -2 * a * x * np.exp(-a * x**2)

# ================== 3. Initial Tests ===================
x_test = 0.5
B_test = -0.5

print("Gauss map:", gauss_map(x_test, B_test, a))
print("Derivative:", gauss_derivative(x_test, a))

# 4. Vectorised Gauss Map Iteration
def iterate_gauss_map(B_values, a=4.9, x0=0.0,
                      transient=1000, iterations=200):
    # initialise one trajectory for each value of B
    x = np.full(len(B_values), x0, dtype=float)
    # discard transient iterations
    for n in range(transient):
        x = gauss_map(x, B_values, a)
    # store remaining iterations
    results = np.empty((iterations, len(B_values)))

    for n in range(iterations):
        x = gauss_map(x, B_values, a)
        results[n, :] = x

    return results

# =================== 5. Vectorised Iteration Test =======================
B_test_values = np.array([-1.0, -0.5, 0.0, 0.5])
test_results = iterate_gauss_map(
    B_test_values,
    a=4.9,
    x0=0.0,
    transient=1000,
    iterations=200
)
print("\nVectorised iteration test:")
print("Array shape:", test_results.shape)
print("Final iterates:", test_results[-1, :])

# 6. Lyapunov Exponent
def lyapunov_exponent(B_values, a=4.9, x0=0.0,
                      transient=1000, iterations=200):
    # initialise trajectories
    x = np.full(len(B_values), x0, dtype=float)
    # discard transient
    for n in range(transient):
        x = gauss_map(x, B_values, a)

    # accumulate logarithms of derivative magnitudes
    lyapunov_sum = np.zeros(len(B_values))
    for n in range(iterations):
        derivative = gauss_derivative(x, a)
        with np.errstate(divide='ignore'):
            lyapunov_sum += np.log(np.abs(derivative))
        x = gauss_map(x, B_values, a)
    return lyapunov_sum / iterations

# =============== 7. Lyapunov Exponent Test ===============
lyapunov_test = lyapunov_exponent(
    B_test_values,
    a=4.9,
    x0=0.0,
    transient=1000,
    iterations=200
)
print("\nLyapunov exponent test:")
print("Lyapunov exponents:", lyapunov_test)

# 8. Root Finding: Bisection
def bisection(g, xL, xR, tol=1e-10, max_iter=100):
    if g(xL) * g(xR) >= 0:
        raise ValueError("The interval must contain a sign change.")
    iterates = []
    for n in range(max_iter):
        xM = (xL + xR) / 2
        iterates.append(xM)

        if abs(g(xM)) < tol or (xR - xL) / 2 < tol:
            break
        if g(xL) * g(xM) < 0:
            xR = xM
        else:
            xL = xM
    return np.array(iterates)

# ================ 9. Bisection Test ================
def test_function_1(x):
    return x ** 2 - 2

bisection_results = bisection(
    test_function_1,
    1,
    2
)
print("\nBisection test:")
print("Root:", bisection_results[-1])
print("Iterations:", len(bisection_results))

# 10. Root Finding: Newton's Method
def newton(g, dg, x0, tol=1e-10, max_iter=100):
    iterates = [x0]
    for n in range(max_iter):
        gx = g(x0)
        dgx = dg(x0)

        if abs(dgx) < 1e-14:
            raise ZeroDivisionError("Derivative is too small")

        x1 = x0 - gx / dgx
        iterates.append(x1)

        if abs(x1 - x0) < tol:
            break
        x0 = x1

    return np.array(iterates)

# ============ 11. Newton's Method Test =============
def derivative_function_1(x):
    return 2*x

newton_results = newton(
    test_function_1,
    derivative_function_1,
    x0=1.5
)
print("\nNewton's method test:")
print("Root:", newton_results[-1])
print("Iterations:", len(newton_results) - 1)

# Root Finding: Secant Method
def secant(g, x0, x1, tol=1e-10, max_iter=100):
    iterates = [x0, x1]

    for n in range(max_iter):
        g0 = g(x0)
        g1 = g(x1)

        denominator = g1 - g0
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError("Denominator is too small")
        x2 = x1 - g1 * (x1 - x0) / denominator
        iterates.append(x2)

        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return np.array(iterates)

# =============== 13. Secant Test =================
secant_results = secant(test_function_1, x0=1.0, x1=2.0)

print("\nSecant method test:")
print("Root", secant_results[-1])
print("Iterations:", len(secant_results) - 2)

# =================== 14. Test ======================
# Second Root-Finding Function
def test_function_2(x):
    return np.cos(x) - x

def derivative_function_2(x):
    return - np.sin(x) - 1

# Bisection
bisection_results_2 = bisection(
    test_function_2, xL=0, xR=1, tol=1e-14
)
# Newton's Method
newton_results_2 = newton(
    test_function_2, derivative_function_2, x0=0.5, tol=1e-14
)
# Secant Method
secant_results_2 = secant(
    test_function_2, x0=0.0, x1=1.0, tol=1e-14
)

print("\nSecond root-finding test:")
print("Bisection root:", bisection_results_2[-1])
print("Bisection iterations:", len(bisection_results_2))

print("Newton root:", newton_results_2[-1])
print("Newton iterations:", len(newton_results_2) - 1)

print("Secant root:", secant_results_2[-1])
print("Secant iterations:", len(secant_results_2) - 2)

# 15. Test Analytical Derivative vs Numerical Derivative
def numerical_derivative(x, B, a, h):
    return(gauss_map(x + h, B, a) -
           gauss_map(x - h, B, a)) / (2 * h)

x_test = 0.5
B_test = -0.5

analytical = gauss_derivative(x_test, a)

print("\nDerivative validation test:")
print("Analytical derivative:", analytical)

for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    numerical = numerical_derivative(x_test, B_test, a, h)
    error = abs(analytical - numerical)

    print("h =", h,
          "| Numerical derivative =", numerical,
          "| Error =", error)

# 16. Test: Derivatve Recursion with Respect to B
def parameter_derivative(B, a=4.9, x0=0.5, iterations=10):
    x = x0
    d = 0.0

    for n in range(iterations):
        d = gauss_derivative(x, a) * d + 1
        x = gauss_map(x, B, a)

    return d

def numerical_parameter_derivative(B, a=4.9, x0=0.5,
                                   iterations=10, h=1e-5):
    x_plus = x0
    x_minus = x0

    for n in range(iterations):
        x_plus = gauss_map(x_plus, B + h, a)
        x_minus = gauss_map(x_minus, B - h, a)
    return (x_plus - x_minus) / (2 * h)

B_test = -0.5
analytical_B = parameter_derivative(B_test, a)

print("\nParameter derivative validation:")
print("Analytical derivative:", analytical_B)

for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    numerical_B = numerical_parameter_derivative(
        B_test, a, h=h
    )
    error_B = abs(analytical_B - numerical_B)

    print("h =", h,
          "| Numerical derivative =", numerical_B,
          "| Error =", error_B)

# 17. Test: Superstable Fixed Point
B_super = -1.0
x_super = 0.0
#verify fixed point
next_x = gauss_map(x_super, B_super, a)
#calculate derivative
derivative_super = gauss_derivative(x_super, a)
#calculate Lyapunov exponent
with np.errstate(divide='ignore'):
    lyapunov_super = np.log(abs(derivative_super))

print("\nSuperstable fixed-point test:")
print("Initial point:", x_super)
print("Next iterate:", next_x)
print("Derivative:", derivative_super)
print("Lyapunov exponent:", lyapunov_super)

# 18. Test: Period-2 Orbit
B_period2 = -0.25

#define the second iterate minus the initial value
def period2_function(x):
    return gauss_map(
        gauss_map(x, B_period2, a), B_period2, a
    ) - x

# find first point of period-2 orbit
period2_results = bisection(
    period2_function, xL=-0.14, xR=-0.13, tol=1e-12
)
x1 = period2_results[-1]
#calculate second point
x2 = gauss_map(x1, B_period2, a)
#calculate the exact orbit Lyapunov exponent
lambda_period2 = 0.5 * (
    np.log(abs(gauss_derivative(x1, a))) +
    np.log(abs(gauss_derivative(x2, a)))
)
#calculate the long-time Lyapunov exponent
lambda_long = lyapunov_exponent(
    np.array([B_period2]),
    a=a,
    x0=0.0,
    transient=5000,
    iterations=10000
)[0]

print("\nPeriod-2 orbit validation:")
print("First orbit point:", x1)
print("Second orbit point:", x2)
print("f(f(x1)):", gauss_map(x2, B_period2, a))
print("Exact orbit Lyapunov exponent:", lambda_period2)
print("Long-time Lyapunov exponent:", lambda_long)
print("Absolute difference:", abs(lambda_period2 - lambda_long))

# 19. Bifurcation Diagram and Lyapunov Exponent

#Paraeter values
B_values = np.linspace(-1.1, 1.0, 2000)

# calculate iterates
bifurcation_data = iterate_gauss_map(
    B_values, a=a, x0=0.0,
    transient=1000, iterations=500
)
# calculate Lyapunov exponents
lyapunov_values = lyapunov_exponent(
    B_values, a=a, x0=0.0,
    transient = 100, iterations=500
)
# create two plots sharing the B-axis
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 9), sharex=True
)
# Bifurcation diagram
for i in range(len(B_values)):
    ax1.plot(
        [B_values[i]] * bifurcation_data.shape[0],
        bifurcation_data[:, i],
        ',k', alpha=0.5
    )
ax1.set_ylabel("x")
ax2.set_title("Bifurcation Diagram of the Gauss Map (alpha = 4.9)")
ax1.grid(alpha=0.3)

# Lyapunov exponent
ax2.plot(B_values, lyapunov_values, color='blue')
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel("B")
ax2.set_ylabel("Lyapunov Exponent")
ax2.set_title("Lyapunov Exponent (alpha = 4.9)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 20.1 Hysteresis: Increasing and Decreasing B
def hysteresis_sweep(B_values, a=4.9, x0=0.0,
                     transient=1000, iterations=100):
    x = x0
    results = np.empty((iterations, len(B_values)))

    for i, B_value in enumerate(B_values):
        # allow system to settle at the current B
        for n in range(iterations):
            x = gauss_map(x, B_value, a)
        # record final iterations
        for n in range(iterations):
            x = gauss_map(x, B_value, a)
            results[n, i] = x
    return results, x

# increasing B: -1.1 to -0.6
B_increasing = np.linspace(-1.1, -0.6, 500)

increasing_results, final_state = hysteresis_sweep(
    B_increasing, a=a, x0=0.0
)

# decreasing B: -0.6 to -1.1
B_decreasing = np.linspace(-0.6, -1.1, 500)

decreasing_results, final_state = hysteresis_sweep(
    B_decreasing, a=a, x0=final_state
)

# plot both sweeps
plt.figure(figsize=(12, 6))

for i in range(len(B_increasing)):
    plt.plot(
        [B_increasing[i]] * increasing_results.shape[0],
        increasing_results[:, i],
    )

for i in range(len(B_decreasing)):
    plt.plot(
        [B_decreasing[i]] * decreasing_results.shape[0],
        decreasing_results[:, i],
    )

plt.xlabel("B")
plt.ylabel("x")
plt.title("Hysteresis of the Gauss Map")
plt.legend(["Increasing B", "Decreasing B"])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#20.2 Hysteresis: Detect Large Jumps
increasing_final = increasing_results[-1, :]
decreasing_final  = decreasing_results[-1, :]

increasing_jumps = np.where(
    np.abs(np.diff(increasing_final)) > 0.1
)[0]
decreasing_jumps = np.where(
    np.abs(np.diff(decreasing_final)) > 0.1
)[0]

print("\nHysteresis jump-point test:")
print("Increasing B jump points:")

for i in increasing_jumps:
    print(B_increasing[i], "to", B_increasing[i + 1])
print("Decreasing B jump points:")

for i in decreasing_jumps:
    print(B_decreasing[i], "to", B_decreasing[i + 1])


# ======================================= PART B: Results & Visualisation ===============================
# Bifurctaion Diagram for a = 7.5

# new parameter value
a_75 = 7.5
B_values_75 = np.linspace(-1.1, 1.0, 2000)

# calculate iterate
bifurcation_data_75 = iterate_gauss_map(
    B_values_75,
    a=a_75,
    x0=0.0,
    transient=1000,
    iterations=500
)
# calculate Lyapunov exponents
lyapunov_values_75 = lyapunov_exponent(
    B_values_75,
    a=a_75,
    x0=0.0,
    transient=1000,
    iterations=500
)

# create two plots sharing the same B-axis
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 9), sharex=True
)
# Bifurcation diagram
for i in range(len(B_values_75)):
    ax1.plot(
        [B_values_75[i]] * bifurcation_data_75.shape[0],
         bifurcation_data_75[:, i],
         ',k',
         alpha=0.5
    )
ax1.set_ylabel("x")
ax2.set_title("Bifurcation Diagram of the Gauss Map (alpha = 7.5)")

ax2.plot(B_values_75, lyapunov_values_75, color='blue')
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel("B")
ax2.set_ylabel("Lyapunov Exponent")
ax2.set_title("Lyapunov Exponent (alpha = 7.5)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===========================
# Fixed Points and Stability
# ===========================

def find_fixed_points(B, a=4.9, scan_points=10000):
    def g(x):
        return gauss_map(x, B, a) - x

    # map sends every point into [B, B + 1]
    x_values = np.linspace(B, B + 1, scan_points)
    roots = []
    # scan for intervals with sign change
    for i in range(len(x_values) - 1):

        xL = x_values[i]
        xR = x_values[i + 1]

        if g(xL) * g(xR) < 0:
            root = bisection(g, xL, xR, tol=1e-12)[-1]

            # avoid counting same root twice
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(root)
    return roots

# TEST B = 0.5 and B = -0.9
for B_test in [0.5, -0.9]:
    roots = find_fixed_points(B_test, a=4.9)
    print(f"\nFixed points for B = {B_test}:")

    for root in roots:
        derivative = gauss_derivative(root, a)
        stability = abs(derivative)

        if stability < 1:
            classification = "Stable"
        elif stability > 1:
            classification = "Unstable"
        else:
            classification = "Marginal"

        print("Fixed point:", root)
        print("|f'(x*)|:", stability)
        print("Classification:", classification)

# Convergence Comparison of Root-Finding Methods
B_convergence = 0.5

# fixed-point equation
def fixed_point_function(x):
    return gauss_map(x, B_convergence, a) - x
# derivative of fixed-point equation
def fixed_point_derivative(x):
    return gauss_derivative(x, a) - 1

# calculate accurate reference root
reference_results = bisection(
    fixed_point_function, 0.5, 0.8, tol=1e-14
)
reference_root = reference_results[-1]

# apply all three root-finding methods
bisection_results = bisection(
    fixed_point_function, 0.5, 0.8, tol=1e-14
)
newton_results = newton(
    fixed_point_function,
    fixed_point_derivative,
    x0=0.7,
    tol=1e-14
)
secant_results = secant(
    fixed_point_function,
    x0=0.5,
    x1=0.8,
    tol=1e-14
)

# calculate absolute errors
bisection_errors = np.abs(bisection_results - reference_root)
newton_errors = np.abs(newton_results - reference_root)
secant_errors = np.abs(secant_results - reference_root)

print("\nBisection errors:", bisection_errors)
print("Newton errors:", newton_errors)
print("Secant errors:", secant_errors)

# plot convergence on a logarithmic scale
plt.figure(figsize=(10, 6))
plt.semilogy(
    range(1, len(bisection_errors) + 1),
    bisection_errors,
    'o-',
    label="Bisection"
)

plt.semilogy(
    range(len(newton_errors)),
    newton_errors,
    's-',
    label="Newton"
)

plt.semilogy(
    range(len(secant_errors)),
    secant_errors,
    '^-',
    label="Secant"
)

plt.xlabel("Iteration k")
plt.ylabel("Absolute Error")
plt.title("Convergence of Root-Finding Methods")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRoot-finding convergence comparison:")
print("Reference root:", reference_root)
print("Bisection iterations:", len(bisection_results))
print("Newton iterations:", len(newton_results) - 1)
print("Secant iterations:", len(secant_results) - 2)

# ===========================
# Estimate Convergence Orders
# ===========================
def estimate_order(errors, method_name):
    errors = np.asarray(errors, dtype=float)

    # exclude errors affected by floating-point precision
    errors = errors[
        np.isfinite(errors) & (errors > 1e-12)
    ]
    print(f"\n{method_name} errors:", errors)
    if len(errors) < 3:
        print("Not enough reliable errors to estimate p")
        return

    # estimate p using three consecutive errors
    orders = []
    for i in range(1, len(errors) - 1):
        e_previous = errors[i - 1]
        e_current = errors[i]
        e_next = errors[i + 1]

        denominator = np.log(e_current / e_previous)
        if denominator != 0:
            p = np.log(e_next / e_current) / denominator
            orders.append(p)

    if orders:
        print(f"{method_name} estimated orders:",
              np.round(orders, 6))
        print(f"{method_name} average estimated p:",
              f"{np.mean(orders):.6f}")
    else:
        print("Could not estimate convergence order")

print("\nCONVERGENCE ORDER ANALYSIS")

estimate_order(bisection_errors, "Bisection")
estimate_order(newton_errors, "Newton")
estimate_order(secant_errors, "Secant")

print("\nTheoretical convergence orders:")
print("Bisection: p = 1")
print("Newton:    p = 2")
print("Secant     p = 1.618")

      
# ============================
# Superstable Periodic Orbits
# ============================

def superstable_function(B, k, a=4.9):
    """Calculate F_k(B) and its derivative with respect to B"""
    x = 0.0
    derivative_B = 0.0

    number_of_iterations = 2**k

    for n in range(number_of_iterations):
        derivative_B = (
            gauss_derivative(x, a) * derivative_B + 1
        )
        x = gauss_map(x, B, a)
    return x, derivative_B

# Newton's method for finding superstable parameters
def superstable_newton(B0, k, a=4.9,
                        tol=1e-12, max_iter=100):
    B = B0
    iterates = [B]

    for n in range(max_iter):
        F, dF = superstable_function(B, k, a)

        if abs(dF) < 1e-14:
            raise ZeroDivisionError(
                "Derivative is too small"
            )
        B_new = B - F / dF
        iterates.append(B_new)

        if abs(B_new - B) < tol:
            break

        B = B_new
    return np.array(iterates)
# test k = 0
test_superstable = superstable_newton(-1.0, k=0, a=a)

print("\nSuperstable orbit test:")
print("B for k = 0:", test_superstable[-1])
print("F_0(B):", superstable_function(
    test_superstable[-1], 0, a
)[0])

# Find Non-Trivial Superstable Parameter for k = 1
superstable_k1 = superstable_newton(
    B0=0.7, k=1, a=a
)
B1 = superstable_k1[-1]

print("\nNon-trivial superstable test:")
print("B for k = 1:", B1)
print("F_1(B):", superstable_function(B1, 1, a)[0])

# Safeguard Newton Method for Superstable Parameters
def superstable_newton_checked(B0, k, a=4.9,
                               tol=1e-12, max_iter=100):
    B = B0
    iterates = [B]

    for n in range(max_iter):
        F, dF = superstable_function(B, k, a)

        if abs(F) < tol:
            break

        if abs(dF) < 1e-14:
            raise ZeroDivisionError(
                "Derivative is too small"
            )
        
        B_new = B - F / dF
        iterates.append(B_new)

        if abs(B_new - B) < tol:
            F_new, _ = superstable_function(B_new, k, a)

            if abs(F_new) < tol:
                B = B_new
                break
        B = B_new
    else:
        raise RuntimeError(
            "Newton's method did not converge"
        )
    return np.array(iterates)

# Test non-trivial root for k = 1
superstable_k1 = superstable_newton_checked(
    B0=-0.77, k=1, a=a
)
B1 = superstable_k1[-1]

print("\nChecked superstable test:")
print("B for k = 1:", B1)
print("F_1(B)", superstable_function(B1, 1, a)[0])
print("Iterationss", len(superstable_k1) - 1)

# Find the Superstable Parameter for k = 2
superstable_k2 = superstable_newton_checked(
    B0=-0.65, k=2, a=a
)
B2 = superstable_k2[-1]

print("\nSuperstable period-4 test:")
print("B for k = 2:", B2)
print("F_2(B):", superstable_function(B2, 2, a)[-0])
print("Iterations:", len(superstable_k2) - 1)

# Find The Superstable Parameter for k = 3
superstable_k3 = superstable_newton_checked(
    B0=-0.60, k=3, a=a
)
B3 = superstable_k3[-1]

print("\nSuperstable period-8 test:")
print("B for k = 3:", B3)
print("F_3(B):",superstable_function(B3, 3, a)[0])
print("Iterations:", len(superstable_k3) - 1)

# FInd The Superstable Parameter for k = 4
superstable_k4 = superstable_newton_checked(
    B0=-0.56, k=4, a=a
)
B4 = superstable_k4[-1]

print("\nSuperstable period-16 test:")
print("B for k = 4:", B4)
print("F_4(B):", superstable_function(B4, 4, a)[0])
print("Iterations:", len(superstable_k4) - 1)

# Find The Superstable Parameter for k = 5
superstable_k5 = superstable_newton_checked(
    B0=-0.565, k=5, a=a
)
B5 = superstable_k5[-1]

print("\nSuperstable period-32 test:")
print("B for k = 5:", B5)
print("F_5(B):", superstable_function(B5, 5, a)[0])
print("Iterations:", len(superstable_k5) - 1)

# Find The Superstable Parameter for k = 6
superstable_k6 = superstable_newton_checked(
    B0=-0.568, k=6, a=a
)
B6 = superstable_k6[-1]

print("\nSuperstable period-64 test:")
print("B for k = 6:", B6)
print("F_6(B):", superstable_function(B6, 6, a)[0])
print("Iterations:", len(superstable_k6) - 1)

# Find The Superstable Parameter for k = 7
superstable_k7 = superstable_newton_checked(
    B0=-0.5682, k=7, a=a
)
B7 = superstable_k7[-1]

print("\nSuperstable period-128 test:")
print("B for k = 7:", B7)
print("F_7(B):", superstable_function(B7, 7, a)[0])
print("Iterations:", len(superstable_k7) - 1)

# =============================
# Feigenbaum Convergence Ratios
# =============================

# store superstable parameter values
B_superstable = np.array([
    -1.0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7
])
# caculate successive Feigembaum ratios
feigenbaum_ratios = []
for k in range(2, len(B_superstable)):
    numerator = B_superstable[k-1] - B_superstable[k-2]
    denominator = B_superstable[k] - B_superstable[k-1]

    ratio = numerator / denominator
    feigenbaum_ratios.append(ratio)

    print(f"Delta_{k} = {ratio:.10f}")
print("\nTheoretical Feigenbaum constant:")
print(4.669201609)

# =============================
# Backward Superstable Cascade
# =============================
def backward_superstable_newton(k, initial_guess, a=4.9,
                                tol=1e-12, max_iter=100):
    B = initial_guess
    for iteration in range(max_iter):
        x = 0.0
        derivative_B = 0.0

        # iterate map 2^(k+1) times
        for _ in range(2**(k + 1)):
            # derivative of the map with respect to x
            derivative_x = -2 * a * x * np.exp(-a * x**2)

            # derivative recurrance with respect to B
            derivative_B = derivative_x * derivative_B + 1

            # gauss map
            x = np.exp(-a * x**2) + B

        # check convergence
        if abs(x) < tol:
            return B, x, iteration + 1
        
        if abs(derivative_B) < 1e-14:
            raise ValueError("Newton derivative too small")

        # Newton update
        B = B - x / derivative_B
    raise RuntimeError(f"Newton method did not converge for k={k}")

# starting guesses for backard cascade
backward_guesses = [
    -0.0080557283,
    -0.312,
    -0.372,
    -0.388,
    -0.3914,
    -0.3922,
    -0.39236,
    -0.39240
]  
B_backward = []

print("\nBACKWARD SUPERSTABLE CASCADE")

for k, guess in enumerate(backward_guesses):
    B, residual, iterations = backward_superstable_newton(k, guess)
    B_backward.append(B)

    print(f"\nk = {k}")
    print(f"Period = {2**(k+1)}")
    print(f"B_{k} = {B:.12f}")
    print(f"F_{k+1}(B) = {residual:.6e}")
    print(f"Iterations = {iterations}")

# =============================
# Backward Feigenbaum Ratios
# =============================
print("\nBACKWARD FEIGENBAUM RATIOS")

for k in range(2, 8):
    delta = (
        (B_backward[k-1] - B_backward[k-2]) /
        (B_backward[k] - B_backward[k-1])
    )
    print(f"Delta_{k} = {delta:.10f}")

print("\nTheroetical Feigenbaum constant:")
print("4.669201609")

# ======================================== PART C ==========================================
# TEST 1: Root Finders on Known Problems ===================================================

def test_function_1(x):
    return x**2 - 2

def test_derivative_1(x):
    return 2*x

def test_function_2(x):
    return np.cos(x) - x

def test_derivative_2(x):
    return -np.sin(x) - 1

tolerance = 1e-14
max_iterations = 100

# known roots
known_roots = {
    "x^2 - 2": np.sqrt(2),
    "cos(x) - x": 0.7390851332151607
}

# test equation 1: x^2 - 2 = 0
bisection_1 = bisection(
    test_function_1, 1, 2,
    tolerance, max_iterations
)
newton_1 = newton(
    test_function_1, test_derivative_1, 1.5,
    tolerance, max_iterations
)
secant_1 = secant(
    test_function_1, 1, 2,
    tolerance, max_iterations
)

# test equation 2: cos(x) - x = 0
bisection_2 = bisection(
    test_function_2, 0, 1,
    tolerance, max_iterations
)
newton_2 = newton(
    test_function_2, test_derivative_2, 0.5,
    tolerance, max_iterations
)
secant_2 = secant(
    test_function_2, 0, 1,
    tolerance, max_iterations
)

# display results
test_results = [
    ("x^2 - 2", "Bisection", bisection_1, known_roots["x^2 - 2"]),
    ("x^2 - 2", "Newton", newton_1, known_roots["x^2 - 2"]),
    ("x^2 - 2", "Secant", secant_1, known_roots["x^2 - 2"]),
    ("cos(x) - x", "Bisection", bisection_2, known_roots["cos(x) - x"]),
    ("cos(x) - x", "Newton", newton_2, known_roots["cos(x) - x"]),
    ("cos(x) - x", "Secant", secant_2, known_roots["cos(x) - x"])
]
print("\nTEST 1: ROOT FINDER VALIDATION")

for equation, method, results, exact_root in test_results:
    calculated_root = results[-1]
    error = abs(calculated_root - exact_root)

    print(f"\nEquation: {equation}")
    print(f"Method: {method}")
    print(f"Calculated root: {calculated_root:.6f}")
    print(f"Absolute error: {error:.3e}")
    print(f"Number of iterations: {len(results) - 1}")

# Convergence Order Validation
def calculate_errors(iterates, exact_root):
    return np.abs(iterates - exact_root)

def estimate_convergence_order(errors):
    orders = []
    precision_limit = 1e-12

    for k in range(1, len(errors) - 1):
        e0 = errors[k - 1]
        e1 = errors[k]
        e2 = errors[k + 1]

        if e0 > precision_limit and e1 > precision_limit and e2 > precision_limit:
            denominator = np.log(e1 / e0)
            
            if abs(denominator) > 1e-12:
                p = np.log(e2 / e1) / denominator

                if np.isfinite(p):
                    orders.append(p)
    return orders

convergence_tests = [
    ("Newton", newton_1, np.sqrt(2)),
    ("Secant", secant_1, np.sqrt(2)),
    ("Newton", newton_2, 0.7390851332151607),
    ("Secant", secant_2, 0.7390851332151607)
]
print("\nCONVERGENCE ORDER VALIDATION")

for method, iterates, exact_root in convergence_tests:
    errors = calculate_errors(iterates, exact_root)
    orders = estimate_convergence_order(errors)

    print(f"\nMethod: {method}")
    print(f"Estimated orders: {orders}")

    if len(orders) > 0:
        print(f"Mean estimated order: {np.mean(orders):.6f}")

# diagnose convergence errors
print("\nCONVERGENCE ERROR DIAGNOSTICS")
for method, iterates, exact_root in [
    ("Newton, x^2 - 2", newton_1, np.sqrt(2)),
    ("Secant, x^2 - 2", secant_1, np.sqrt(2)),
    ("Newton, cos(x) - x", newton_2, 0.7390851332151607),
    ("Secant, cos(x) - x", secant_2, 0.7390851332151607)
]:
    errors = np.abs(iterates - exact_root)

    print(f"\n{method}")
    for k, error in enumerate(errors):
        print(f"Iteration {k}: {error:.16e}")

# Bisection: verify the halving of the interval
print("\nBISECTION INTERVAL HALVING")

for equation, iterates in [
    ("x^2 - 2", bisection_1),
    ("cos(x) - x", bisection_2)
]:
    step_sizes = np.abs(np.diff(iterates))
    ratios = step_sizes[1:] / step_sizes[:-1]

    print(f"\nEquation: {equation}")
    print(f"Last interval-width ratios: {ratios[-5:]}")

# TEST 2: DERIVATIVES ====================================================================
print("\nTEST 2: DERIVATIVE VALIDATION")

# Test 1: compare the analytical derivative with central differences
alpha_test = 4.9
beta_test = -0.6
x_test = 0.37

analytical_derivative = gauss_derivative(x_test, alpha_test)
h_values = np.logspace(-1, -8, 8)

print("\nDerivative f'(x):")
print("Analytical derivative:", analytical_derivative)

for h in h_values:
    finite_difference = (
        gauss_map(x_test + h, beta_test, alpha_test)
        - gauss_map(x_test - h, beta_test, alpha_test)
    ) / (2 * h)
    error = abs(finite_difference - analytical_derivative)
    print(f"h = {h:.1e}, error = {error:.6e}")

# Test 2: Compare the recursive derivative dF_k/d beta with central finite differences
def calculate_F_and_derivative(beta, k, alpha=4.9):
    x = 0.0
    d = 0.0

    for n in range(2**k):
        d = gauss_derivative(x, alpha) * d + 1
        x = gauss_map(x, beta, alpha)
    return x, d
k_test = 3

analytical_F, analytical_dF = calculate_F_and_derivative(
    beta_test, k_test, alpha_test
)
print("\nDerivative dF_k/dbeta:")
print("Analytical derivative:", analytical_dF)

for h in h_values:
    F_plus, _ = calculate_F_and_derivative(
        beta_test + h, k_test, alpha_test
    )
    F_minus, _ = calculate_F_and_derivative(
        beta_test - h, k_test, alpha_test
    )
    finite_difference = (F_plus - F_minus) / (2 * h)
    error = abs(finite_difference - analytical_dF)
    print(f"h = {h:.1e}, error = {error:.6e}")

# TEST 3: AN EXACT SPECIAL CASE ====================================================================
print("\nTEST 3: EXACT SPECIAL CASE")

alpha_test = 4.9
beta_test = -1.0
x_test = 0.0

# analytical verification
print("\nAnalytical verification:")
print("f(0) =", gauss_map(x_test, beta_test, alpha_test))
print("f'(0) =", gauss_derivative(x_test, alpha_test))

# numerical verification
fixed_point = gauss_map(x_test, beta_test, alpha_test)

print("\nNumerical verification:")
print("Initial point:", x_test)
print("Next iterate:", fixed_point)
print("Is it a fixed point?", np.isclose(fixed_point, x_test))
print("Is it superstable?",
      abs(gauss_derivative(x_test, alpha_test)) == 0)

# Lyapunov exponent at the superstable fixed point
lyapunov_special = lyapunov_exponent(
    np.array([-1.0]),
    a=alpha_test,
    x0=0.0,
    transient=0,
    iterations=10
)

print("\nLyapunov exponent verification:")
print("Calculated Lyapunov exponent:", lyapunov_special[0])
print("Expected value: -inf")

# TEST 4: LYAPUNOV EXPONENT OF PERIODIC ORBITS =============================================
print("\nTEST 4: LYAPUNOV EXPONENT OF PERIODIC ORBITS")

alpha_test = 4.9
beta_test = -0.75

# equation for a period-2 orbit
def period_two_function(x):
    next_x = gauss_map(x, beta_test, alpha_test)
    return gauss_map(next_x, beta_test, alpha_test) - x

# find one point of period-2 orbit
period_two_results = secant(
    period_two_function,
    x0=-0.03,
    x1=0.0
)
orbit_x1 = period_two_results[-1]
orbit_x2 = gauss_map(orbit_x1, beta_test, alpha_test)

print("\nPERIOD-2 CONSISTENCY CHECK")
print("f(x1) =", gauss_map(orbit_x1, beta_test, alpha_test))
print("f(x2) =", gauss_map(orbit_x2, beta_test, alpha_test))
print("f(f(x1)) - x1 =", period_two_function(orbit_x1))
print("f(x2) - x1 =", gauss_map(orbit_x2, beta_test, alpha_test) - orbit_x1)

print("\nPeriod-2 orbit:")
print("x1 =", orbit_x1)
print("x2 =", orbit_x2)
print("f(x2) =", gauss_map(orbit_x2, beta_test, alpha_test))
print("Period-2 residual =", period_two_function(orbit_x1))

print("Is this a genuine period-2 orbit?",
      not np.isclose(orbit_x1, orbit_x2))

# calculate Lyapunov exponent
derivative_1 = gauss_derivative(orbit_x1, alpha_test)
derivative_2 = gauss_derivative(orbit_x2, alpha_test)

periodic_lyapunov = 0.5 * (
    np.log(abs(derivative_1))
    + np.log(abs(derivative_2))
)
print("\nPeriodic-orbit Lyapunov exponent:")
print(periodic_lyapunov)

# compare with long-time Lyapunov average
long_time_lyapunov = lyapunov_exponent(
    np.array([beta_test]),
    a=alpha_test,
    x0=x1,
    transient=1000,
    iterations=20000
)[0]

print("\nLong-time Lyapunov exponent:")
print(long_time_lyapunov)

print("\nAbsolute difference:")
print(abs(periodic_lyapunov - long_time_lyapunov))