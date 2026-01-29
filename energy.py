# MINIMIZE = travel time, energy consumption
# 
# ENVIRONMENT CALCULATIONS:
# 4 types of nodes:
# - 1 depot (D)
# - intersections (1,...n)
# - BSS (battery swapping station) (BSS1,...BSSn)
# - customers (C1,...Cn)
# edge variables:
# - distance
# - traffic factor
# - base_speed
# other(?) variables:
# - starting node
# - module capacity
# - 
# -
# -
# M = total vehicle mass including payload (kg)
# g = gravity (9.81 m/s²)
# f = rolling resistance coefficient (unitless)
# rho = air density (kg/m³)
# Cx = aerodynamic drag coefficient (unitless)
# A = vehicle cross-sectional area (m²)
# v = vehicle speed (m/s)
# m = mass_factor
# alpha = angle
# d = distance
#
#
#
# keep track of:
# - how many batteries full
# - how far nearest BSS
# - calculate if next movement will make nearest BSS unreachable
# - calculate shortest immediate path to closest customer

#import math


#g = 9.81
#rho = 1.205
#alpha = 0.86

#def energy_consumption(base_mass,f,rho,Cx,A,v,m,alpha,d,current_load_kg):

#    M = base_mass + current_load_kg
#    if 50 <= v_kmh <= 80:
#        dv_dt = 0.3
#    elif 81 <= v_kmh <= 120:
#        dv_dt = 2
#    else:
#         dv_dt = 0
#    return (1/3600)*(M*g*(f*math.cos(alpha) + math.sin(alpha)) + 0.0386*(rho*Cx*A*v**2) + (M+m)*(dv_dt))*d


import math

def calculate_energy_consumption(
    M: float,           # total vehicle mass including payload (kg)
    f: float,           # rolling resistance coefficient
    rho: float,         # air density (kg/m³)
    Cx: float,          # aerodynamic drag coefficient
    A: float,           # vehicle cross-sectional area (m²)
    v: float,           # vehicle speed (m/s)
    m: float,           # mass_factor
    alpha: float,       # angle (in degrees, convert to radians)
    d: float            # distance (meters)
) -> float:
    """
    Calculate energy consumption in kWh using your exact equation:
    E = (1/3600) * [M*g*(f*cos(alpha)+sin(alpha)) + 0.0386*(rho*Cx*A*v²) + (M+m)*(dv/dt)] * d
    """
    g = 9.81  # gravity m/s^2

     # Convert alpha from degrees to radians
    alpha_rad = math.radians(alpha)
    
    # Calculate dv/dt based on speed range (convert v from m/s to km/h)
    v_kmh = v * 3.6
    if 50 <= v_kmh <= 80:
        dv_dt = 0.3
    elif 81 <= v_kmh <= 120:
        dv_dt = 2
    else:
        dv_dt = 0
    
    # energy equation (gives in Watt Hours Wh)
    energy_wh = (1/3600) * (M * g * (f * math.cos(alpha_rad) + math.sin(alpha_rad)) + 0.0386 * (rho * Cx * A * v**2) + (M + m) * dv_dt) * d
    
    # Convert from Wh to kWh (1 kWh = 1000 Wh)
    energy_kwh = energy_wh / 1000
    
    return energy_kwh