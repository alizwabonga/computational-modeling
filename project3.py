#PROJECT 3: One-Lane Traffic: The Totally Asymmetric Simple Exclusion Process

import numpy as np
import matplotlib.pyplot as plt

# ==========================================PART A ================================================
# Basic TASEP model : Represent each site as 1 (occupied) or 0 (empty)

# 1. Initialise functions
def initialise_ring(L, N, rng=None):
    """Create a ring of L sites containing exactly N particiles"""
    if L < 2:
        raise ValueError("A ring must contain at least 2 sites.")
    if not 0 <= N <= L:
        raise ValueError("N must be between 0 and L.")

    if rng is None:
        rng = np.random.default_rng()

    # choose N distinct sites uniformly at random
    lattice = np.zeros(L, dtype=int)
    occupied_sites = rng.choice(L, size=N, replace=False)
    lattice[occupied_sites] = 1

    return lattice

def initialise_open_chain(L, density, rng=None):
    """Create an open chain with random initial occupation"""
    if L < 1:
        raise ValueError("L must be at least 1.")
    if not 0 <= density <= 1:
        raise ValueError("Density must be between 0 and 1.")

    if rng is None:
        rng = np.random.default_rng()

    # occupy each site independently with the given probability
    return (rng.random(L) < density).astype(int)

# 2. Random-sequential update functions
def random_sequential_ring_step(lattice, rng):
    """Attempt one random bond update on a ring"""
    L = len(lattice)

    # select a site and the site immediately to its right
    i = rng.integers(L)
    j = (i + 1) % L

    # move if current site is occupied; and next site is empty
    if lattice[i] == 1 and lattice[j] == 0:
        lattice[i] = 0
        lattice[j] = 1
        return 1
    
    return 0

def random_sequential_open_step(lattice, alpha, beta, rng):
    """Attempt one random bond update on an open chain"""
    L = len(lattice)

    # select one of the L+1 bonds, including entry and exit
    bond = rng.integers(L + 1)

    if bond == 0:
        # entry at left boundary
        if lattice[0] == 0 and rng.random() < alpha:
            lattice[0] = 1
            return 1
        return 0
    elif bond == L:
        # exit at right boundary
        if lattice[L - 1] == 1 and rng.random() < beta:
            lattice[L - 1] = 0
            return 1
        return 0
    else:
        # internal bond: move one particle to the right
        i = bond -1
        if lattice[i] == 1 and lattice[i + 1] == 0:
            lattice[i] = 0
            lattice[i + 1] = 1
            return 1
        return 0

# 3. Sweep functions
def random_sequential_ring_sweep(lattice, rng):
    """Perform one sweep: L random bond attempts"""
    L = len(lattice)
    successful_hops = 0

    for _ in range(L):
        successful_hops += random_sequential_ring_step(lattice, rng)

    return successful_hops

def random_sequential_open_sweep(lattice, alpha, beta, rng):
    """Perform one sweep: L+1 random bond attempts"""
    L = len(lattice)

    successful_events = 0

    for _ in range(L + 1):
        successful_events += random_sequential_open_step(
            lattice, alpha, beta, rng
        )
    return successful_events

# 4. Parallel update
def parallel_ring_step(lattice, p, rng):
    """Perform one simultaneous-update time step on a ring"""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")

    # identify all particles that can move in the original configuration
    eligible = (lattice == 1) & (np.roll(lattice, -1) == 0)

    # each eligible particle attempts to move with probability p
    moving = eligible & (rng.random(len(lattice)) < p)

    # apply all successful moves simultaneously
    successful_hops = int(np.sum(moving))
    lattice[moving] = 0
    lattice[np.roll(moving, 1)] = 1

    return successful_hops

# 5. Measurement & Error functions
def measure_density(lattice):
    """Return the mean occupation (density) of each site"""
    return lattice.astype(float)

def blocking_error(data, block_size=None):
    """Estimate the standard error of a mean using block averages"""
    data = np.asarray(data, dtype=float)

    if len(data) < 2:
        return np.nan
    
    if block_size is None:
        block_size = max(1, len(data) // 20)

    # keep only complete blocks
    number_of_blocks = len(data) // block_size
    if number_of_blocks < 2:
        return np.nan

    trimmed = data[:number_of_blocks * block_size]
    block_means = trimmed.reshape(number_of_blocks, block_size).mean(axis=1)

    return block_means.std(ddof=1) / np.sqrt(number_of_blocks)

# 6. Simulation functions
def simulate_ring_random(L, N, sweeps, burn_in, rng):
    """Simulate a random-sequential ring and measure its cuurrent"""
    lattice = initialise_ring(L, N, rng)
    currents = []
    density_samples = []

    # allow the system to reach its stationary state
    for _ in range(burn_in):
        random_sequential_ring_sweep(lattice, rng)

    #record measurements after burn-in
    for _ in range(sweeps):
        hops = random_sequential_ring_sweep(lattice, rng)
        currents.append(hops / L)
        density_samples.append(lattice.copy())

    mean_current = np.mean(currents)
    current_error = blocking_error(currents)
    density_profile = np.mean(density_samples, axis=0)

    return mean_current, current_error, density_profile

def simulate_open_random(L, alpha, beta, sweeps, burn_in, rng):
    """Simulate an open chain and measure its current and density"""
    lattice = np.zeros(L, dtype=int)
    entry_currents = []
    exit_currents = []
    density_samples = []

    for _ in range(burn_in):
        random_sequential_open_sweep(lattice, alpha, beta, rng)
    # measurement period
    for _ in range(sweeps):
        entry_hops = 0
        exit_hops = 0

        for _ in range(L + 1):
            bond = rng.integers(L + 1)
            if bond == 0:
                if lattice[0] == 0 and rng.random() < alpha:
                    lattice[0] = 1
                    entry_hops += 1
            elif bond == L:
                if lattice[L - 1] == 1 and rng.random() < beta:
                    lattice[L - 1] = 0
                    exit_hops += 1
            else:
                i = bond -1
                if lattice[i] == 1 and lattice[i + 1] == 0:
                    lattice[i] = 0
                    lattice[i + 1] = 1

        entry_currents.append(entry_hops)
        exit_currents.append(exit_hops)
        density_samples.append(lattice.copy())
    # calculate stationary current from entry and exit measurements
    mean_entry = np.mean(entry_currents)
    mean_exit = np.mean(exit_currents)
    mean_current = (mean_entry + mean_exit) / 2

    current_error = blocking_error(
        (np.array(entry_currents) + np.array(exit_currents)) / 2
    )
    # average occupation at every site
    density_profile = np.mean(density_samples, axis=0)
    return mean_current, current_error, density_profile

def simulate_ring_parallel(L, N, p, steps, burn_in, rng):
    """Simulate a parallel-update ring and measure its current"""
    lattice = initialise_ring(L, N, rng)

    # let system settle b4 taking measurements
    for _ in range(burn_in):
        parallel_ring_step(lattice, p, rng)

    currents = []
    density_samples = []

    for _ in range(steps):
        hops = parallel_ring_step(lattice, p, rng)

        currents.append(hops / L)
        density_samples.append(lattice.copy())

    mean_current = np.mean(currents)
    current_error = blocking_error(currents)
    density_profile = np.mean(density_samples, axis=0)

    return mean_current, current_error, density_profile

# 7. Space-Time recording and plotting
def record_ring_spacetime(L, N, p, steps, burn_in, rng):
    """Record the ring occupation after each parallel time step"""
    lattice = initialise_ring(L, N, rng)

    for _ in range(burn_in):
        parallel_ring_step(lattice, p, rng)

    history = np.zeros((steps + 1, L), dtype=int)
    history[0] = lattice.copy()

    for t in range(1, steps + 1):
        parallel_ring_step(lattice, p, rng)
        history[t] = lattice.copy()
    return history

def plot_ring_spacetime(history, density, p):
    """Plot the occupation of every ring site over time"""
    plt.figure(figsize=(10, 6))

    plt.imshow(
        history,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="binary",
        vmin=0, 
        vmax=1
    )
    plt.xlabel("Site")
    plt.ylabel("Time step")
    plt.title(f"Ring space-time plot: density = {density:.1f}, p = {p}")
    plt.colorbar(label="Occupation (0 = empty, 1 = occupied)")
    plt.tight_layout()
    plt.show()

# ================================================= PART B ================================================
# SIMULATIONS AND ANALYSIS

def open_chain_phase_diagrams():
    L = 200
    sweeps = 5000
    burn_in = 1000

    rates = np.arange(0.1, 1.0, 0.1)
    # arrays to store measured quantities
    current_grid = np.zeros((len(rates), len(rates)))
    density_grid = np.zeros((len(rates), len(rates)))

    rng = np.random.default_rng(42)

    # simulate every combination if alpha and beta
    for i, alpha in enumerate(rates):
        for j, beta in enumerate(rates):
            J, error, profile = simulate_open_random(
                L=L, 
                alpha=alpha,
                beta=beta,
                sweeps=sweeps,
                burn_in=burn_in,
                rng=rng
            )
            current_grid[i, j] = J
            # bulk density measured over central half
            density_grid[i, j] = np.mean(
                profile[L//4:3*L//4]
            )
            print(
                f"alpha={alpha:.1f}, beta={beta:.1f}, "
                f"J={J:.4f}, density={density_grid[i, j]:.4f}"
            )
    # plot current phase diagrams
    plt.figure(figsize=(8, 6))
    plt.imshow(
        current_grid,
        origin="lower",
        extent=[0.5, 0.95, 0.05, 0.95],
        aspect="auto",
        cmap="viridis"
    )
    plt.colorbar(label="Current J")
    plt.xlabel("Exit rate beta")
    plt.ylabel("Entry rate alpha")
    plt.title("Open TASEP: Current phase diagram")
    plt.tight_layout()
    plt.show()

    # plot bulk density phase diagram
    plt.figure(figsize=(8, 6))

    plt.imshow(
        density_grid,
        origin="lower",
        extent=[0.05, 0.95, 0.05, 0.95],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1
    )
    plt.colorbar(label="Bulk density")
    plt.xlabel("Exit rate beta")
    plt.ylabel("Entry rate alpha")
    plt.title("Open TASEP: Bulk density phase diagram")
    plt.tight_layout()
    plt.show()

    return current_grid, density_grid               
               
# Current versus Density
def plot_current_vs_density():
    # ring sizes and densities required for Part B
    L = 1000
    densities = np.arange(0.1, 1.0, 0.1)
    p_values = [0.5, 0.75, 1.0]

    # use one random number generator for reproducibility
    rng = np.random.default_rng(42)
    # siulation settings
    burn_in = 1000
    sweeps = 5000

    # store one simulated current for each density
    rs_simulated = []

    for rho in densities:
        N = round(rho * L)
        J, error, profile = simulate_ring_random(
            L=L,
            N=N,
            sweeps=sweeps,
            burn_in=burn_in,
            rng=rng
        )
        rs_simulated.append(J)
    rs_simulated = np.array(rs_simulated)
    print("All simulated currents:", rs_simulated)
    print("Number of simulated currents:", len(rs_simulated))

        # exact finite-size theoretical current
    rs_exact = np.array([
            round(rho * L) * (L - round(rho * L)) / (L * (L -1))
            for rho in densities
    ])
        # check both arrays contain nine values
    print("Random-sequential:")
    print("Densities:", len(densities))
    print("Simulated currents:", len(rs_simulated))

    # plot random sequential results
    
    plt.figure(figsize=(8, 5))

    plt.plot(
            densities, rs_simulated, "o",
            label="Simulation"
    )

    plt.plot(densities, rs_exact, "-",
                label="Exact theory"
    )

    plt.xlabel("Density")
    plt.ylabel("Current J")
    plt.title("Random sequential update: current vs density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        # parallel update for each hopping probability
    plt.figure(figsize=(9, 6))

    for p in p_values:
        # store simulated currents for htis value of p
        simulated = []

        for rho in densities:
            N = round(rho * L)
            J, error, profile = simulate_ring_parallel(
                L=L,
                N=N,
                p=p,
                steps=sweeps,
                burn_in=burn_in,
                rng=rng
            )
            simulated.append(J)
        simulated = np.array(simulated)
        # exact theoretical parallel-update current
        exact = (
            1 - np.sqrt(1 - 4 * p * densities * (1 - densities))
        ) / 2
        # mean field approxiamtion
        mean_field = p * densities * (1 - densities)
        # plot simulation, exact theory & mean-field results
        plt.plot(
            densities, simulated, "o",
            label=f"Simulation, p={p}"
        )
        plt.plot(
            densities, exact, "-",
            label=f"Exact theory, p={p}"
        )
        plt.plot(
            densities, mean_field, "--",
            label=f"Mean field, p={p}"
        )
        print(f"Parallel update, p={p}:")
        print("Densities:", len(densities))
        print("Simulated currents:", len(simulated))

        plt.xlabel("Density")
        plt.ylabel("Current J")
        plt.title("Parallel update: current vs density")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_open_density_profiles():
    L = 200
    cases = [
        (0.2, 0.7, "Low-density phase"),
        (0.7, 0.2, "High-density phase"),
        (0.7, 0.7, "Maximal-current phase"),
        (0.3, 0.3, "Coexistence phase")
    ]
    burn_in = 5000
    sweeps = 20000

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (alpha, beta, phase_name) in zip(axes.flat, cases):
        # simuulate open chain & obtain average density profile
        current, error, profile = simulate_open_random(
            L=L, 
            alpha=alpha,
            beta=beta,
            sweeps=sweeps,
            burn_in=burn_in,
            rng=rng
        )
        # site positions along the chain
        sites = np.arange(1, L + 1)
        # plot average occupation at each site
        ax.plot(sites, profile, label="Simulated density")

        ax.set_xlabel("Site")
        ax.set_ylabel("Average density")
        ax.set_title(
            f"{phase_name}\n"
            f"alpha={alpha}, beta={beta}, J={current:.3f}"
        )
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

        print(
            f"{phase_name}: alpha={alpha}, beta={beta}, "
            f"current={current:.4f}, error={error:.5f}"
        )

    plt.suptitle("Open TASEP: Density Profiles")
    plt.tight_layout()
    plt.show()

# =========================================== PART C =======================================================
# Validation Tests

def run_part_c_tests():
    """Run theoretical validation tests for Part C"""
    rng = np.random.default_rng(42)
    print("\n=============== PART C: VALIDATION TESTS =================\n")

    # =============================
    # TEST 1: Open chain with L = 1
    # =============================

    L = 1
    alpha = 0.4
    beta = 0.6

    sweeps = 100000
    burn_in = 10000

    current, error, profile = simulate_open_random(
        L=L,
        alpha=alpha,
        beta=beta,
        sweeps=sweeps,
        burn_in=burn_in,
        rng=rng
    )
    exact_density = alpha / (alpha + beta)
    exact_current = alpha * beta / (alpha + beta)

    print("TEST 1: Open chain, L = 1")
    print(f"Simulated density: {profile[0]:.5f}")
    print(f"Exact density:     {exact_density:.5f}")
    print(f"Simulated current: {current:.5f}")
    print(f"Exact current:     {exact_current:.5f}")
    print(f"Estimated error:   {error:.5f}\n")

    # ===================================================
    # TEST 2: Exact Finite-Size Current, alpha = beta = 1
    # ===================================================
   
    print("TEST 2: Open chain, alpha = beta = 1")
    lengths = [1, 2, 8, 64]

    for L in lengths:
        current, error, profile = simulate_open_random(
            L=L,
            alpha=1.0,
            beta=1.0,
            sweeps=100000,
            burn_in=20000,
            rng=rng
        )
        exact_current = (L + 2) / (2 * (2 * L + 1))

        print(
            f"L = {L}, "
            f"Simulates J = {current:.5f}, "
            f"Exact J = {exact_current:.5f},"
            f"Error = {error:.5f}"
        )
    print("The theoretical current approaches 0.25 as L increases.\n")

    # ======================================
    # TEST 3.1: Random sequential small rings
    # ======================================

    print("TEST 3.1: Random-sequential small rings")
    L = 10
    for N in range (L + 1):
        current, error, profile = simulate_ring_random(
            L=L,
            N=N,
            sweeps=30000,
            burn_in=5000,
            rng=rng
        )
        exact_current = N * (L - N) / (L * (L - 1))
        print(
            f"N = {N:2d}, density = {N/L:.2f}: "
            f"Simulated J = {current:.5f}, "
            f"exact J = {exact_current:.5f}"
        )
    # ====================================
    # TEST 3.2: Parallel update with p = 1
    # ====================================

    print("\nTest 3.2: Parallel update, p = 1")
    L = 10
    p = 1.0
    for N in range(L + 1):
        current, error, profile = simulate_ring_parallel(
            L=L, N=N,
            p=p,
            steps=1000,
            burn_in=500,
            rng=rng
        )
        exact_current = min(N, L - N) / L
        print(
            f"N = {N:2d}, density = {N/L:.2f}: "
            f"simulated J = {current:.5f},"
            f"exact J = {exact_current:.5f}"
        )

    print("\nTransient duration for parallel update (p = 1):")
    L = 10
    rng_transient = np.random.default_rng(42)

    for N in range(L + 1):
        lattice = initialise_ring(L, N, rng_transient)
        # exact number of hops in steady state
        target_hops = min(N, L - N)

        if target_hops == 0:
            transient_duration = 0
        else:
            transient_duration = None
            for t in range(1, 100 * L + 1):
                hops = parallel_ring_step(
                    lattice,
                    p=1.0,
                    rng=rng_transient
                )
                if hops == target_hops:
                    transient_duration = t
                    break
        print(
            f"N = {N:2d}: transient duration = "
            f"{transient_duration} time steps"
        )

    # ==============================
    # TEST 4: Particle-hole Symmetry
    # ==============================

    print("\nTEST 4: Particle-hole symmetry")
    L = 100
    sweeps = 50000
    burn_in = 10000

    parameter_pairs = [
        (0.2, 0.7),
        (0.3, 0.6)
    ]
    for alpha, beta in parameter_pairs:
        # original simulation: alpha, beta
        J_ab, error_ab, profile_ab = simulate_open_random(
            L=L,
            alpha=alpha,
            beta=beta,
            sweeps=sweeps,
            burn_in=burn_in,
            rng=rng
        )
        # swapped simulation: beta, alpha
        J_ba, error_ba, profile_ba = simulate_open_random(
            L=L,
            alpha=beta,
            beta=alpha,
            sweeps=sweeps, 
            burn_in=burn_in,
            rng=rng
        )
        # particle hole symmetry predicts: 
        # J(alpha,beta) = J(beta,alpha) 
        # rho_i(alpha,beta) = 1 - rho_(L+1-i)(beta,alpha)
        expected_profile = 1 - profile_ba[::-1]

        current_difference = abs(J_ab - J_ba)
        profile_difference = np.mean(
            np.abs(profile_ab - expected_profile)
        )
        print(f"\nEntry rate alpha = {alpha}, exit rate beta = {beta}")
        print(f"J(alpha, beta) = {J_ab:.5f}")
        print(f"J(beta, alpha) = {J_ba:.5f}")
        print(f"Current difference = {current_difference:.5f}")
        print(
            "Mean density-profile difference = "
            f"{profile_difference:.5f}"
        )
        print("\nParticle-hole symmetry test completed\n")


    print("\n=============== PART C TEST FINISHED ===============")

# ========================================== PART D ==============================================
# Rarefaction Fans and Shocks

# D1: Rarefaction fan
def simulate_rarefaction_fan(L=4000, realisations=100):
    rng = np.random.default_rng(42)

    observation_times = [200, 400, 800]

    # coordinates measured from initial front
    x = np.arange(-800, 801)
    # store density profile from every realisation
    profiles = {
        t: [] for t in observation_times
    }
    for run in range(realisations):
        # initially fill sites 1,...,2000
        lattice = np.zeros(L, dtype=int)
        lattice[:L // 2] = 1

        for t in range(1, max(observation_times) + 1):

            # one random sequential Monte Carlo sweep
            random_sequential_ring_sweep(lattice, rng)

            if t in observation_times:
                # initial front is between sites 2000 and 2001
                # python index 1999 corresponds to site 2000
                indices = (1999 + x) % L

                local_density = lattice[indices].copy()
                profiles[t].append(local_density)

        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{realisations} realisations")

    # average profiles over realisations
    average_profiles = {
        t: np.mean(profiles[t], axis=0)
        for t in observation_times
    }
    return x, average_profiles

def plot_rarefaction_fan():
    L = 4000
    realisations = 100

    x, average_profiles = simulate_rarefaction_fan(
        L=L,
        realisations=realisations
    )
    plt.figure(figsize=(10, 6))

    for t in [200, 400, 800]:
        # rescale position by time
        scaled_x = x / t
        plt.plot(
            scaled_x,
            average_profiles[t],
            label=f"Simulation, t = {t}"
        )
    # theoretical rarefaction profile
    theoretical_x = np.linspace(-1, 1, 500)
    theoretical_density = 0.5 * (1 - theoretical_x)

    plt.plot(
        theoretical_x,
        theoretical_density,
        "k--",
        linewidth=2,
        label="Theory"
    )
    plt.xlabel("x/t")
    plt.ylabel("Average density")
    plt.title("Rarefaction fan on a ring (L = 4000)")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# D2: Shock formation and Propagation
def simulate_shock(L=4000, rho_L=0.2, rho_R=0.6,
                   observation_times=None, realisations=20):
    if observation_times is None:
        observation_times = [0, 200, 400, 600, 800]
    rng = np.random.default_rng(42)
    half = L // 2

    # store density profiles for each observation time
    profiles = {
        t: [] for t in observation_times
    }
    for run in range(realisations):
        # initialise two regions with random occupation
        lattice = np.zeros(L, dtype=int)
        lattice[:half] = (
            rng.random(L - half) < rho_R
        ).astype(int)

        # record initial profile
        if 0 in observation_times:
            profiles[0].append(lattice.copy())

        for t in range(1, max(observation_times) + 1):
            random_sequential_ring_sweep(lattice, rng)

            if t in observation_times:
                profiles[t].append(lattice.copy())
        if (run + 1) % 5 == 0:
            print(
                f"Completed {run + 1}/{realisations}"
                "shock realisations"
            )
    # average density profiles over realisations
    average_profiles = {
        t: np.mean(profiles[t], axis=0)
        for t in observation_times
    }
    return average_profiles

def plot_shock():
    L = 4000
    rho_L = 0.2
    rho_R = 0.6

    observation_times = [0, 200, 400, 600, 800]
    profiles = simulate_shock(
        L=L, 
        rho_L=rho_L,
        rho_R=rho_R,
        observation_times=observation_times,
        realisations=20
    )
    # focus on initial density jump at site 2000
    x = np.arange(-500, 501)

    measured_positions = []
    plt.figure(figsize=(10, 6))

    for t in observation_times:
        profile = profiles[t][(1999 + x) % L]
        plt.plot(
            x,
            profile,
            label=f"t = {t}"
        )
        if t > 0:
            # find position where density crosses midpoint between the two bulk densities
            midpoint = (rho_L + rho_R) / 2

            # smooth density profile to reduce fluctuations
            window_size = 51
            kernel = np.ones(window_size) / window_size

            smoothed_profile = np.convolve(
                profile,
                kernel,
                mode="same"
            )
            # search for shock near expected position
            expected_position = 0.2 * t
        
            search_indices = np.where(
                (x >= expected_position - 100) &
                (x <= expected_position + 100)
            )[0]
            crossing = search_indices[
                np.argmin(
                    np.abs(
                        smoothed_profile[search_indices] - midpoint
                    )
                )
            ]
            measured_positions.append(
                (t, x[crossing])
            )
    plt.xlabel("Position from initial interface")
    plt.ylabel("Average density")
    plt.title("Shock propagation on a ring (L = 4000)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # estimate shock velocity from position vs time
    times = np.array([item[0] for item in measured_positions])
    positions = np.array([item[1] for item in measured_positions])

    measured_velocity, intercept = np.polyfit(
        times, positions, 1
    )
    theoretical_velocity = 1 - rho_L - rho_R

    print("\nSHOCK VELOCITY RESULTS")
    print("===========================")

    for t, position in measured_positions:
        print(f"Time = {t:4d}, shock position = {position:6.1f}")

    print(f"\nMeasured shock velocity = {measured_velocity:.4f}")
    print(f"Theoretical shock velocity = {theoretical_velocity:.4f}")


# ===================== TEST FUNCTIONS =========================
if __name__ == "__main__":
   
   # PART B: Open-Chain Density Profiles
   plot_current_vs_density()
   open_chain_phase_diagrams()
   plot_open_density_profiles()

   # PART B.2: Space-Time plots
   rng = np.random.default_rng(42)

   history_03 = record_ring_spacetime(
       L=1000,
       N=300,
       p=0.5,
       steps=200,
       burn_in=100,
       rng=rng
   )
   plot_ring_spacetime(history_03, density=0.3, p=0.5)

   history_07 = record_ring_spacetime(
       L=1000,
       N=700,
       p=0.5,
       steps=200,
       burn_in=100,
       rng=rng
   )
   plot_ring_spacetime(history_07, density=0.7, p=0.5)

   # PART C: Validation Tests
   run_part_c_tests()

   # PART D1: Rarefaction Fan
   plot_rarefaction_fan()
   # PART D2: Shock
   plot_shock()