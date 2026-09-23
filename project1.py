import math
import matplotlib.pyplot as plt

#Physical constants
m = 100.0
A = 1.0
h0 = 80_000.0
v0 = 0.0

g0 = 9.81
RE = 6.371e6
G = 6.67430e-11
M = 5.972e24

CD = 0.5
rho0 = 1.225
H = 8500.0

#Physical functions
def gravity(h):
    return g0 * (RE / (RE + h))**2
def constant_gravity(h):
    return g0

def density(h):
    return rho0 * math.exp(-h / H)
def constant_density(h):
    return rho0
def vacuum_density(h):
    return 0.0

def acceleration(h, v, gravity_function=gravity, density_function=density):
    return gravity_function(h) - (CD * density_function(h) * A * v**2) / (2 * m)

def terminal_velocity(h):
    return math.sqrt(
        (2 * m * gravity(h)) /
        (CD * density(h) * A)
    )

#Numerical Methods
def euler(dt, gravity_function=gravity, density_function=density, initial_velocity=v0):
    t = 0.0
    h = h0
    v = initial_velocity

    times = [t]
    altitudes = [h]
    velocities = [v]

    while h > 0:
        a = acceleration(h, v, gravity_function, density_function)

        h_new = h - v * dt
        v_new  = v + a * dt
        t_new = t + dt

        if h_new <= 0:
            fraction = h / (h - h_new)

            t = t + fraction * dt
            h = 0.0
            v = v + a * fraction * dt
        else:
            h = h_new
            v = v_new
            t = t_new    

        times.append(t)
        altitudes.append(h)
        velocities.append(v)

    return times, altitudes, velocities


def euler_richardson(dt, gravity_function=gravity, density_function=density, initial_velocity=v0):
    t = 0.0
    h = h0
    v = initial_velocity

    times = [t]
    altitudes = [h]
    velocities = [v]

    while h > 0:
        a = acceleration(h, v, gravity_function, density_function)

        h_mid = h - v * dt / 2
        v_mid = v + a * dt / 2

        a_mid = acceleration(
            h_mid,
            v_mid,
            gravity_function,
            density_function
        )

        h_new = h - v_mid * dt 
        v_new = v + a_mid * dt
        t_new = t + dt

        if h_new <= 0:
            fraction = h / (h - h_new)

            t = t + fraction * dt
            h = 0.0
            v = v + a_mid * fraction * dt
        else:
            h = h_new
            v = v_new
            t = t_new    

        times.append(t)
        altitudes.append(h)
        velocities.append(v)

    return times, altitudes, velocities

#PART B
times_euler, altitudes_euler, velocities_euler = euler(0.1)
times_er, altitudes_er, velocities_er = euler_richardson(0.1)

#PART B: Altitude vs Time
plt.figure()

plt.plot(
    times_euler,
    altitudes_euler,
    label="Euler"
)
plt.plot(
    times_er,
    altitudes_er,
    label="Euler-Richardson"
)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Altitude vs Time")
plt.legend()
plt.grid()
plt.show()
#calculate local terminal velocity
terminal_velocities = [
    terminal_velocity(h)
    for h in altitudes_euler
]

#PART B: Velocity vs Time
plt.figure()

plt.plot(
    times_euler,
    velocities_euler,
    label="Euler"
)
plt.plot(
    times_er,
    velocities_er,
    label="Euler-Richardson"
)
plt.plot(
    times_euler,
    terminal_velocities,
    label="Local terminal velocity"
)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Time")
plt.legend()
plt.grid()
plt.show()

#TEST 1: Vacuum Free Fall
def test_vacuum_free_fall():
    times, altitudes, velocities = euler(
        0.1,
        constant_gravity,
        vacuum_density
    )
    print("Vacuum Free Fall Test")
    print("Final time:", times[-1])
    print("Final altitude:", altitudes[-1])
    print("Final velocities:", velocities[-1])

    exact_velocity = g0 * times[-1]
    print("Exact velocity:", exact_velocity)
    exact_altitude = h0 - 0.5 * g0 * times[-1]**2
    print("Exact altitude:", exact_altitude)

    exact_altitudes = [
        h0 - 0.5 * g0 * t**2
        for t in times
    ]
    exact_velocities = [
        g0 * t
        for t in times
    ]

    plt.figure()
    plt.plot(times, altitudes, label="Euler numerical")
    plt.plot(times, exact_altitudes, label="Exact analytical")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Vacuum Free Fall: Altitude")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(times, velocities, label="Euler numerical")
    plt.plot(times, exact_velocities, label="Exact analytical")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Vacuum Free Fall: Velocity")
    plt.legend()
    plt.grid()
    plt.show()

test_vacuum_free_fall()    

#TEST 2: Constant Density Terminal Velocity
def test_terminal_velocity(): 
    vterm = math.sqrt(
        (2 * m * g0) / (CD * rho0 * A)
    )
    print("Constant Density Terminal Velocity Test")
    print("Theoretical terminal velocity:", vterm)

    times, altitudes, velocities = euler(
        0.1,
        constant_gravity,
        constant_density,
        vterm
    )
    print("Final velocity:", velocities[-1])   

    plt.figure()
    plt.plot(times, velocities, label="Numerical velocity")
    plt.axhline(vterm, label="Theoretical terminal velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Constant Density: Terminal Velocity")
    plt.legend()
    plt.grid()
    plt.show()  

test_terminal_velocity()

#TEST 3: Approach to Terminal Velocity
def test_approach_to_terminal_velocity():
    vterm = math.sqrt(
        (2 * m * g0) / (CD * rho0 * A)
    )
    print("Approach to Terminal Velocity Test")
    print("Theoretical termianl velocity:", vterm)

    times, altitudes, velocities = euler(
        0.1,
        constant_gravity,
        constant_density,
        0.0
    )
    exact_velocities = [
        vterm * math.tanh(g0 * t / vterm)
        for t in times
    ]
    print("Final numerical velocity:", velocities[-1])
    print("Final analytical velocity:", exact_velocities[-1])

    plt.figure()
    plt.plot(
        times, 
        velocities,
        label="Euler numerical"
    )
    plt.plot(
        times,
        exact_velocities,
        label="Analytical solution"
    )    
    plt.axhline(
        vterm,
        label="Terminal velocity"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Approach to Terminal Velocity")
    plt.legend()
    plt.grid()
    plt.show()

test_approach_to_terminal_velocity()

#TEST 6: Energy/Consistency Check
def test_energy_consistency():
    #vacuum Case
    times_vacuum, altitudes_vacuum, velocities_vacuum = euler(
        0.1,
        gravity,
        vacuum_density
    )
    vacuum_energy = []
    
    for h, v in zip(altitudes_vacuum, velocities_vacuum):
        kinetic_energy = 0.5 * m * v**2
        potential_energy = -G * M * m / (RE + h)
        total_energy = kinetic_energy + potential_energy
        vacuum_energy.append(total_energy)

    print("Vacuum initial energy:", vacuum_energy[0])
    print("Vacuum final energy:", vacuum_energy[-1])

    #drag case
    times_drag, altitudes_drag, velocities_drag = euler(0.1)
    drag_energy = []

    for h, v in zip(altitudes_drag, velocities_drag):
        kinetic_energy = 0.5 * m * v**2
        potential_energy = -G * M * m / (RE + h)
        total_energy = kinetic_energy + potential_energy
        drag_energy.append(total_energy)

    print("Drag initial energy:", drag_energy[0])
    print("Drag final energy:", drag_energy[-1])

    #energy plot
    plt.figure()
    plt.plot(
        times_vacuum,
        vacuum_energy,
        label="Vacuum"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.title("Energy Conservation in Vacuum")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(
        times_drag,
        drag_energy,
        label="With atmospheric drag"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.title("Total Mechanical Energy with Atmospheric Drag")
    plt.legend()
    plt.grid()
    plt.show()
    
test_energy_consistency()

#PART D: Physical Analysis

#use existing Euler-Richardson results
times = times_er
altitudes = altitudes_er
velocities = velocities_er
#total time of descent
total_time = times[-1]
#maximum velocity achieved and the altitude at which it occurs
max_velocity = max(velocities)
max_index = velocities.index(max_velocity)
altitude_at_max_velocity = altitudes[max_index]
#final velocity at sea level
impact_velocity = velocities[-1]

print("\nPART D: Physical Analysis")
print("--------------------------")
print("Total time of descent:", total_time, "s")
print("Maximum velocity:", max_velocity, "m/s")
print("Altitude at maximum velocity:", altitude_at_max_velocity, "m")
print("Final velocity at sea level:", impact_velocity, "m/s")