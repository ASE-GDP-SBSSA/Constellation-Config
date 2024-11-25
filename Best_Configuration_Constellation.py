import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, cos, radians

'''Based on the paper 
Li, G., Liu, J., Jiang, H., & Liu, C. (2023). 
Research on the Efficient Space Debris Observation Method Based on Optical Satellite Constellations. 
Applied Sciences (Switzerland), 13(7). 
https://doi.org/10.3390/app13074127'''


# Constants
EARTH_RADIUS = 6371  # km
EARTH_GRAVITY_PARAM = 398600.4  # km^3/s^2
FOV = radians(15)  # Field of view in radians NEED TO CHECK !!!!
SIMULATION_TIME = 24 * 60 * 60  # 24 hours in seconds
TIME_STEP = 60  # Simulation step in seconds (1 minute)


def generate_walker_delta(num_satellites, num_planes, altitude, inclination): #phasing NEED TO CHECK !!!
    """
    Generates satellite positions for a Walker-Delta constellation.
    Walker-Delta distributes satellites evenly around the Earth across multiple orbital planes.
    "All the satellite orbits in the constellation were circular orbits, with the same orbital altitude and inclination"

    INPUTS
    num_satellites : Total number of satellites in the constellation
    num_planes : Total number of orbital planes in the constellation
    altitude : Altitude of the orbits in km (measured from the Earth's surface)
    inclination : Orbital inclination in degrees

    OUTPUTS
    satellites : List containing the characteristics of each satellite
                semi_major_axis in km
                inclination in radians
                RAAN in radians
                true_anomaly in radians
                satellite_period in seconds
                constellation_period in seconds
    """
    satellites = []
    semi_major_axis = EARTH_RADIUS + altitude
    for plane in range(num_planes):
        for sat in range(num_satellites // num_planes):
            RAAN = 360 / num_planes * plane  #Ω Right Ascension of Ascending Node, ensures that the orbital planes are distributed evenly around the Earth
            true_anomaly = (360 / (num_satellites // num_planes)) * sat  #f True anomaly, ensures that satellites in a plane are equally spaced in their orbit
            satellite_period = 360 * sqrt(semi_major_axis ** 3 / EARTH_GRAVITY_PARAM)
            constellation_period = (satellite_period * num_planes / num_satellites)
            satellites.append({  #Each satellite is added with its characteristics
                'semi_major_axis': semi_major_axis,
                'inclination': inclination,
                'RAAN': radians(RAAN),
                'true_anomaly': radians(true_anomaly),
                'satellite_period': radians(satellite_period),
                'constellation_period': radians(constellation_period)
            })
    return satellites


def calculate_visibility(grid, satellite): #NEED TO CHECK !!!!
    """
    Determine if a grid point (representing space debris) is visible to a satellite.

    INPUTS
    grid : Cartesian coordinates [x,y,z] of a point representing a space debris or a target (e.g. grid = [7000, 0, 0])
    satellite : characteristics of a satellite ("one line" in satellites produced by generate_walker_delta)

    OUTPUTS
    True if the grid point is visible (in the satellite's field of view)
    """
    #Coordinates in the XY plane of the satellite
    x_orb = satellite['semi_major_axis'] * cos(satellite['true_anomaly'])
    y_orb = satellite['semi_major_axis'] * sin(satellite['true_anomaly'])
    z_orb = 0

    #Inclination rotation around X axis
    x_inc = x_orb
    y_inc = y_orb * cos(satellite['inclination']) - z_orb * sin(satellite['inclination'])
    z_inc = y_orb * sin(satellite['inclination']) + z_orb * sin(satellite['inclination'])

    #RAAN rotation around Z axis
    x = x_inc * cos(satellite['RAAN']) - y_inc * sin(satellite['RAAN'])
    y = x_inc * sin(satellite['RAAN']) + y_inc * cos(satellite['RAAN'])
    z = z_inc

    sat_vector = np.array([x,y,z]) #Transform into vector
    grid_vector = np.array(grid) #Transform into vector

    angle = np.arccos(np.dot(sat_vector, grid_vector) /
                      (np.linalg.norm(sat_vector) * np.linalg.norm(grid_vector))) #Angle between the debris vector and the sat vector
    return angle < FOV


def calculate_comprehensive_coverage(satellites, grids, time_step, simulation_time):
    """
    Calculate the comprehensive coverage performance of the constellation.
    Quantify how effectively a satellite constellation covers areas of interest (the grids/space debris).
    This performance is recorded according to time during which the grills are covered and properties of the grids, such as their volume and density.

    INPUTS
    grids : List of target point parameters
          position : Cartesian coordinates [x,y,z] of the point
          volume : Volume represented by this grid
          density : Relative importance or density of debris in this grid
          coverage : Total time the grid is covered based on their importance (initialised to 0)
    time_step : Simulation step in seconds
    simulation_time : Total simulation duration in seconds

    OUTPUTS
    comprehensive_coverage : Quantify the constellation's ability to effectively cover target points based on their importance (volume and density)
    (See the paper on which this code is based)
    """
    constellation_period = satellites[0]['constellation_period']
    total_coverage = 0 #Initialisation
    max_weighted_coverage = 0 #Initialisation

    for t in range(0, simulation_time, time_step):
        for grid in grids:
            max_weighted_coverage += grid['volume'] * grid['density'] * constellation_period #Theoretical coverage if all grids were permanently visible for a full constellation period
            is_visible = any(calculate_visibility(grid['position'], sat) for sat in satellites) #If at least one satellite sees the grid, then this grid is considered visible
            if is_visible:
                grid['coverage'] += time_step * grid['volume'] * grid['density']

    for grid in grids:
        total_coverage += grid['coverage']

    comprehensive_coverage = -total_coverage / max_weighted_coverage
    return comprehensive_coverage


def optimise_constellation(grids, num_satellites_range, num_planes_range, altitudes, inclinations):
    """
    Find the best configuration for the constellation.

    INPUTS
    num_satellites_range : List of total numbers of satellites to test (e.g. [12, 15, 18, 21, 24])
    num_planes_range : List of numbers of orbital planes to test (e.g. [3, 4, 5, 6])
    altitudes : List of altitudes to be tested in km (e.g. [600, 700, 800])
    inclinations : List of inclinations to be tested in degrees (e.g. [90, 95, 98.5])

    OUTPUTS
    best_config : The best constellation configuration (nombre_de_satellites, nombre_de_plans, altitude, inclination)
    best_coverage : Coverage associated with the best configuration
    results : Results for all tested configurations as (nombre_de_satellites, nombre_de_plans, altitude, inclination, coverage)
    """
    best_config = None
    best_coverage = float('+inf')
    results = []

    for num_satellites in num_satellites_range:
        for num_planes in num_planes_range:
            if num_satellites % num_planes != 0: #Skip invalid configurations
                continue

            for altitude in altitudes:
                for inclination in inclinations:
                    satellites = generate_walker_delta(num_satellites, num_planes, altitude, inclination)
                    coverage = calculate_comprehensive_coverage(satellites, grids, TIME_STEP, SIMULATION_TIME)
                    results.append((num_satellites, num_planes, altitude, inclination, coverage))

                    if coverage < best_coverage:
                        best_coverage = coverage
                        best_config = (num_satellites, num_planes, altitude, inclination)

    return best_config, best_coverage, results


def generate_debris(num_debris, min_altitude, max_altitude):
    """
    Generate random debris positions in space between specified altitudes.

    INPUTS
    num_debris: Number of debris that will be simulated
    min_altitude: Minimum altitude at which we want them to be
    max_altitude: Maximum altitude at which we want them to be

    OUTPUTS
    debris : List containing the characteristics of each debris
            position
            volume
            density
            coverage (initialise at 0)
    """
    debris = []
    for _ in range(num_debris):
        #Random altitude
        altitude = np.random.uniform(min_altitude, max_altitude)

        #Random position on the sphere (longitude, latitude)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)

        #Convert spherical coordinates to cartesian coordinates
        x = (EARTH_RADIUS + altitude) * np.sin(phi) * np.cos(theta)
        y = (EARTH_RADIUS + altitude) * np.sin(phi) * np.sin(theta)
        z = (EARTH_RADIUS + altitude) * np.cos(phi)

        volume = np.random.uniform(0.5, 1.5)  #Volume between 0.5 and 1.5 units NEED TO CHECK !!!
        density = np.random.uniform(1e-5, 5e-5)  #Density between 1e-5 and 5e-5 NEED TO CHECK !!!

        debris.append({
            'position': [x, y, z],
            'volume': volume,
            'density': density,
            'coverage': 0
        })

    return debris

def simulate_optimisation():
    """
    Define example grids and parameter ranges for the satellite constellation.
    Call the optimise_constellation function.
    """
    #Example grids
    grids = generate_debris(3, 600, 800)

    #Parameter ranges
    num_satellites_range = [12, 15, 18, 21, 24]  #Total number of satellites
    num_planes_range = [3, 4, 5, 6]  #Number of orbital planes
    altitudes = [600, 700, 800]  #Altitude in km
    inclinations = [90, 95, 98.5]  #Inclination in degrees

    #Optimise
    best_config, best_coverage, results = optimise_constellation(
        grids, num_satellites_range, num_planes_range, altitudes, inclinations
    )

    print("Best configuration:")
    print(f"  Satellites: {best_config[0]}")
    print(f"  Orbital planes: {best_config[1]}")
    print(f"  Altitude: {best_config[2]} km")
    print(f"  Inclination: {best_config[3]}°")
    print(f"  Coverage performance: {best_coverage:.4f}")

    return results, best_config, grids

def plot_results(results):
    """
    Visualise the performance of different configurations.

    Displays two graphs:
        Altitude vs Coverage: Compares coverage performance to altitude by coloring the points by the number of satellites
        Inclination vs Coverage: Compares coverage performance to inclination by coloring the points by the number of satellites
    """
    #Extract data
    num_satellites = [r[0] for r in results]
    altitudes = [r[2] for r in results]
    inclinations = [r[3] for r in results]
    coverage = [r[4] for r in results]

    #Scatter plot of altitude vs coverage performance
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(altitudes, coverage, c=num_satellites, cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(scatter, label='Number of satellites')
    plt.title('Constellation coverage performance by altitude and number of satellites')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Coverage performance')
    plt.grid()
    plt.show()

    #Scatter plot of inclination vs coverage performance
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(inclinations, coverage, c=num_satellites, cmap='plasma', s=100, alpha=0.8)
    plt.colorbar(scatter, label='Number of satellites')
    plt.title('Constellation coverage performance by inclination and number of satellites')
    plt.xlabel('Inclination (degrees)')
    plt.ylabel('Coverage performance')
    plt.grid()
    plt.show()

def plot_earth(ax):
    """
    Add a 3D representation of the Earth as a sphere on the given axis.

    INPUT
    ax : Matplotlib 3D axis object on which to plot the Earth
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.5)


def plot_orbits_and_satellites(ax, satellites):
    """
    Add the orbits and positions of satellites on the provided axis.
    """
    for sat in satellites:
        #Coordinates in the XY plane of the orbit
        theta = np.linspace(0, 2 * np.pi, 100) #100 to define the orbit
        x_orb = sat['semi_major_axis'] * np.cos(theta)
        y_orb = sat['semi_major_axis'] * np.sin(theta)
        z_orb = np.zeros_like(theta) #Create an array of zero that has the same dimensions as theta

        #Inclination rotation around X for the orbit
        x_inc = x_orb
        y_inc = y_orb * np.cos(sat['inclination']) - z_orb * np.sin(sat['inclination'])
        z_inc = y_orb * np.sin(sat['inclination']) + z_orb * np.cos(sat['inclination'])

        #RAAN rotation around Z for the orbit
        x = x_inc * np.cos(sat['RAAN']) - y_inc * np.sin(sat['RAAN'])
        y = x_inc * np.sin(sat['RAAN']) + y_inc * np.cos(sat['RAAN'])
        z = z_inc

        #Plot the orbit
        ax.plot(x, y, z, color='green', linewidth=0.7, alpha=0.8)

        #Coordinates in the XY plane of the satellite
        x_sat = sat['semi_major_axis'] * np.cos(sat['true_anomaly'])
        y_sat = sat['semi_major_axis'] * np.sin(sat['true_anomaly'])
        z_sat = 0

        #Inclination rotation around X for the satellite
        x_sat_inc = x_sat
        y_sat_inc = y_sat * np.cos(sat['inclination']) - z_sat * np.sin(sat['inclination'])
        z_sat_inc = y_sat * np.sin(sat['inclination']) + z_sat * np.cos(sat['inclination'])

        #RAAN rotation around Z for the satellite
        x_sat_final = x_sat_inc * np.cos(sat['RAAN']) - y_sat_inc * np.sin(sat['RAAN'])
        y_sat_final = x_sat_inc * np.sin(sat['RAAN']) + y_sat_inc * np.cos(sat['RAAN'])
        z_sat_final = z_sat_inc

        #Plot the satellite
        ax.scatter(x_sat_final, y_sat_final, z_sat_final, color='red', s=20)

    #Add a legend for satellites
    ax.scatter([], [], color='red', s=30, label='Satellites')

def plot_debris(ax, grids):
    """
    Adds debris positions to the provided axis.
    """
    for grid in grids:
        ax.scatter(grid['position'][0], grid['position'][1], grid['position'][2], color='orange', s=30)
    ax.scatter([], [], color='orange', s=30, label='Débris')

def visualise_constellation(satellites, grids):
    """
    Visualise the Earth, satellite orbits, and debris in 3D.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_earth(ax)
    plot_orbits_and_satellites(ax, satellites)
    plot_debris(ax, grids)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.legend()
    plt.show()

if __name__ == "__main__": #Only executed if the file is executed directly as the main program, not if it is imported as a module in another script
    #Run the simulation and retrieve the results
    results, best_config, grids = simulate_optimisation()

    #Retrieves the settings of the best configuration
    num_satellites = best_config[0]
    num_planes = best_config[1]
    altitude = best_config[2]
    inclination = best_config[3]

    #Generates the optimal constellation
    optimaL_satellites = generate_walker_delta(num_satellites, num_planes, altitude, inclination)

    #Displays the plots
    visualise_constellation(optimaL_satellites, grids)
    plot_results(results)