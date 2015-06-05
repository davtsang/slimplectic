# orbit_util.py
#
# Orbital parameter utilities for converting between 
# cartesian position and velocity vectors and orbital 
# parameters
#
# David Tsang
# dtsang@physics.mcgill.ca

import numpy



def Calc_e_vec(r, v, mu):
    """
    Calculates the eccentricity vector e[3] for a given orbit, from position vector r, 
    velocity vector v, and mu = G*M
    """
    h = numpy.cross(r, v)
    e = numpy.cross(v, h)/mu
    e -= r/(numpy.dot(r,r)**0.5)
    return e

def Calc_e(r, v, mu):
    """
    Calculates the eccentricity scalar, e, for a given orbit, from position vector r, 
    velocity vector v, and mu = G*M
    """
    e = Calc_e_vec(r, v, mu)
    return numpy.dot(e, e)**0.5

def Calc_i(r, v, mu):
    """
    Calculates the inclination angle, i, in radians from the x-y plane for a given orbit, 
    from position vector r, velocity vector v, and mu = G*M
    """
    h = numpy.cross(r, v)
    cos_i = h[2]/(numpy.dot(h, h)**0.5)
    i = numpy.arccos(cos_i)
    return i

def Calc_a(r, v, mu):
    """
    Calculates the semi-major axis, a, for a given orbit, from position vector r, 
    velocity vector v, and mu = G*M
    """
    h = numpy.cross(r, v)
    e = Calc_e_vec(r, v, mu)
    h2 = numpy.dot(h, h)
    e2 = numpy.dot(e, e)
    return h2/(mu*(1.0-e2))

def Calc_Long_Asc_Node(r, v, mu):
    """
    Calculates the Argument of the Ascending Node, Omega, in radians, 
    for a given orbit from the position vector r, velocity vector v, and mu = G*M
    """
    h = numpy.cross(r, v)
    n = numpy.cross(numpy.array([0, 0, 1]), h)
    Omega = numpy.arccos(n[0]/(numpy.dot(n,n))**0.5)
    if n[1] < 0:
        Omega = 2*numpy.pi - Omega
    return Omega

def Calc_Arg_Periapse(r, v, mu):
    """
    Calculates the Argument of Periapse, omega, in radians, 
    for a given orbit from the position vector r, velocity vector v, and mu = G*M
    """
    h = numpy.cross(r, v)
    n = numpy.cross(numpy.array([0, 0, 1]), h)
    e = Calc_e_vec(r, v, mu)
    cos_omega = numpy.dot(n, e)/(numpy.dot(n, n)*numpy.dot(e,e))**0.5
    omega = numpy.arccos(cos_omega)
    if e[2] < 0:
        omega = 2.0*numpy.pi - omega
    return omega

def Calc_True_Anomaly(r, v, mu):
    """
    Calculates the True Anomaly, Theta, in radians, 
    for a given orbit from the position vector r, velocity vector v, and mu = G*M
    """
    e = Calc_e_vec(r, v, mu)
    cos_theta = numpy.dot(e, r)/(numpy.dot(e,e)*numpy.dot(r,r))**0.5
    theta = numpy.arccos(cos_theta)
    if numpy.dot(r, v) < 0:
        theta = 2*numpy.pi - theta
    return theta

def Calc_Eccentric_Anomaly(r, v, mu):
    """
    Calculates the Eccentric Anomaly, E, in radians, 
    for a given orbit from the position vector r, velocity vector v, and mu = G*M
    """
    e = Calc_e(r, v, mu)
    theta = Calc_True_Anomaly(r, v, mu)
    cos_E = e + numpy.cos(theta)
    cos_E /= 1.0 + e*numpy.cos(theta)
    E = numpy.arccos(cos_E)
    if theta > numpy.pi and theta < 2*numpy.pi:
        E = 2*numpy.pi - E
    return E

def Calc_Mean_Anomaly(r, v, mu):
    """
    Calculates the Mean Anomaly, E, in radians, 
    for a given orbit from the position vector r, velocity vector v, and mu = G*M
    """
    e = Calc_e(r, v, mu)
    E = Calc_Eccentric_Anomaly(r, v, mu)
    M = E - e*numpy.sin(E)
    return M
    
    
def Calc_Cartesian(a, e, i, Omega, omega, M, mu, tol = 1e-15):
    """ 
    Take the orbital parameters and calculates the position vector, r[3],
    and the velocity vector, v[3] in cartesian coordinates. 
    Inputs:
    a - Semi-Major Axis
    e - eccentricity scalar
    i - inclination angle in radians from the x-y plane
    Omega - argument of the Ascending Node
    omega - argument of the periapse
    M - mean anomaly
    mu - gravitional parameter G*M
    tol - numerical tolerance for the solution to Kepler's equation
          defaults to 1e-15
    Outputs:
    (r, v)
    r[3] - position vector
    v[3] - velocity vector
    """
    #First we determine the Eccentric Anomaly by solving
    #Kepler's equation
    #tolerance
    E = M
    while True:
        Enext = -(E - e*numpy.sin(E)- M)
        Enext /= 1.0 - e*numpy.cos(E)
        Enext += E
        if numpy.fabs(Enext - E) < tol:
            E = Enext
            break
        E = Enext
    
    #Compute the P and Q vectors
    P = numpy.zeros(3)
    Q = numpy.zeros(3)
    
    P[0] = numpy.cos(omega)*numpy.cos(Omega)
    P[0] -= numpy.sin(omega)*numpy.sin(Omega)*numpy.cos(i)
    
    P[1] = numpy.cos(omega)*numpy.sin(Omega)
    P[1] += numpy.sin(omega)*numpy.cos(Omega)*numpy.cos(i)
    
    P[2] = numpy.sin(omega)*numpy.sin(i)
    
    Q[0] = -numpy.sin(omega)*numpy.cos(Omega)
    Q[0] -= numpy.cos(omega)*numpy.sin(Omega)*numpy.cos(i)
    
    Q[1] = -numpy.sin(omega)*numpy.sin(Omega)
    Q[1] += numpy.cos(omega)*numpy.cos(Omega)*numpy.cos(i)
    
    Q[2] = numpy.cos(omega)*numpy.sin(i)

    #Calculate rate of change of the eccentric anomaly
    Edot = (mu/a**3)**0.5
    Edot /= (1.0-e*numpy.cos(E))

    #Calculate r and v vectors:
    
    r = a*(numpy.cos(E) - e)*P
    r += a*((1.0-e*e)**0.5)*numpy.sin(E)*Q
    
    v = -a*numpy.sin(E)*Edot*P
    v += a*((1.0-e*e)**0.5)*numpy.cos(E)*Edot*Q
    
    return r, v
    
        
        
    
    