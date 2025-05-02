import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sy

def model(y, t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km):
    """function to establish the system in the model
    
       Parameters:
       - y (ndarray): array of dependent variables (P, Pm, Pa, Pu, E, Em, Ea, Eu, M)
       - t (float): time
       - others (float): parameters for the equations (will be given values as we find them)

       Returns:
       - array: vector of derivatives at time t
    """
    # unpack the variables (for readability)
    P, Pm, Pa, Pu, E, Em, Ea, Eu, M = y
    
    # set up the equations
    dy = [
        p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E,                      # P
        h*Me*P + (h/w)*M*P - j*Pm - a*Pm,                                 # Pm
        a*Pm - u*Pa,                                                      # Pa
        u*Pa - n*(Ap/Ae)*Pu - b*Pu,                                       # Pu
        -f*(Ae/Ap)*E + b*Eu - (h/w)*E*M + j*Em + n*(Ap/Ae)*Pu,                    # E
        (h/w)*E*M - a*Em - j*Em,                                          # Em
        a*Em - u*Ea,                                                      # Ea
        - b*Eu + u*Ea - d*Eu,                                # Eu
        -(h/w)*M*P - (h/w)*E*M + (j + u)*(Em + Pm) - vmax*M/(V*(Km + M))  # M
    ]

    return dy


def sy_solve_LU():
    """Solve symbolically using sy.matrices.MatrixBase.LUsolve. Returns a sy.Matrix object listing the steady states in order.
    Further simplification is computed in steady_states_simplification.ipynb"""
    # set up sympy variables
    t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
    P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

    # functions to solve  
    dP = p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E                       # P
    dPm = h*Me*P + (h/w)*M*P - j*Pm - a*Pm                                 # Pm
    dPa = a*Pm - u*Pa                                                      # Pa
    dPu = u*Pa - n*(Ap/Ae)*Pu - b*Pu                                       # Pu
    dE = -f*(Ae/Ap)*E + b*Eu - (h/w)*E*M + j*Em + n*(Ap/Ae)*Pu             # E
    dEm = (h/w)*E*M - a*Em - j*Em                                          # Em
    dEa = a*Em - u*Ea                                                      # Ea
    dEu = - b*Eu + u*Ea - d*Eu                                             # Eu
    # dM = -(h/w)*M*P - (h/w)*E*M + (j + u)*(Em + Pm) - vmax*M/(V*(Km + M))  # M

    # set up matrices for use with solve_linear_LU
    M = sy.Matrix([[-h*(Me + M / w), j, 0, 0, f*Ae / Ap, 0, 0, 0],
                   [h*(Me + M / w), -(j + a), 0, 0, 0, 0, 0, 0],
                   [0, a, -u, 0, 0, 0, 0, 0],
                   [0, 0, u, -(b + n*Ap / Ae), 0, 0, 0, 0],
                   [0, 0, 0, n*Ap / Ae, -(f*Ae / Ap + M*h / w), j, 0, b],
                   [0, 0, 0, 0, M*h / w, -(a + j), 0, 0],
                   [0, 0, 0, 0, 0, a, -u, 0],
                   [0, 0, 0, 0, 0, 0, u, -(b + d)]])
    b = sy.Matrix([-p, 0, 0, 0, 0, 0, 0, 0])

    solution = sy.matrices.MatrixBase.LUsolve(M, b)
    return solution

def solve_except_Pa_Ua_M():
    """Solve all except Pa, Ua, M using sympy"""
    # set up sympy variables
    t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
    P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

    # functions to solve  
    dP = p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E                       # P
    dPm = h*Me*P + (h/w)*M*P - j*Pm - a*Pm                                 # Pm
    dPa = a*Pm - u*Pa                                                      # Pa
    dPu = u*Pa - n*(Ap/Ae)*Pu - b*Pu                                       # Pu
    dE = -f*(Ae/Ap)*E + b*Eu - (h/w)*E*M + j*Em + n*(Ap/Ae)*Pu             # E
    dEm = (h/w)*E*M - a*Em - j*Em                                          # Em
    dEa = a*Em - u*Ea                                                      # Ea
    dEu = - b*Eu + u*Ea - d*Eu                                             # Eu
    # dM = -(h/w)*M*P - (h/w)*E*M + (j + u)*(Em + Pm) - vmax*M/(V*(Km + M))  # M 

    eqs = [dP, dPm, dPu, dE, dEm, dEu]
    vars = [P, Pm, Pu, E, Em, Eu]

    return sy.solve(eqs, vars)   


def substitution():
    """Substitute results from previous function. Full steady states (except M) computed in steady_states_simplification.ipynb"""
    # set up sympy variables
    t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
    P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

    # steady states and remaining equations
    steady_states = {E: (Ae*Ap*Ea*a*b**2*u*w + Ae*Ap*Ea*b**2*j*u*w + Ap**2*Ea*a*b*n*u*w + Ap**2*Ea*b*j*n*u*w + Ap**2*Pa*a*b*n*u*w + Ap**2*Pa*a*d*n*u*w + Ap**2*Pa*b*j*n*u*w + Ap**2*Pa*d*j*n*u*w)/(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b**2*h + Ae*Ap*M*a*b*d*h + Ae*Ap*a*b*f*n*w + Ae*Ap*a*d*f*n*w + Ae*Ap*b*f*j*n*w + Ae*Ap*d*f*j*n*w + Ap**2*M*a*b*h*n + Ap**2*M*a*d*h*n), 
                     Em: (Ae*Ap*Ea*M*b**2*h*u + Ap**2*Ea*M*b*h*n*u + Ap**2*M*Pa*b*h*n*u + Ap**2*M*Pa*d*h*n*u)/(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b**2*h + Ae*Ap*M*a*b*d*h + Ae*Ap*a*b*f*n*w + Ae*Ap*a*d*f*n*w + Ae*Ap*b*f*j*n*w + Ae*Ap*d*f*j*n*w + Ap**2*M*a*b*h*n + Ap**2*M*a*d*h*n), 
                     Eu: Ea*u/(b + d), 
                     P: (Ae**2*Ea*a**2*b**2*f*u*w**2 + 2*Ae**2*Ea*a*b**2*f*j*u*w**2 + Ae**2*Ea*b**2*f*j**2*u*w**2 + Ae**2*a**2*b**2*f*p*w**2 + Ae**2*a**2*b*d*f*p*w**2 + 2*Ae**2*a*b**2*f*j*p*w**2 + 2*Ae**2*a*b*d*f*j*p*w**2 + Ae**2*b**2*f*j**2*p*w**2 + Ae**2*b*d*f*j**2*p*w**2 + Ae*Ap*Ea*a**2*b*f*n*u*w**2 + 2*Ae*Ap*Ea*a*b*f*j*n*u*w**2 + Ae*Ap*Ea*b*f*j**2*n*u*w**2 + Ae*Ap*M*a**2*b**2*h*p*w + Ae*Ap*M*a**2*b*d*h*p*w + Ae*Ap*M*a*b**2*h*j*p*w + Ae*Ap*M*a*b*d*h*j*p*w + Ae*Ap*Pa*a**2*b*f*n*u*w**2 + Ae*Ap*Pa*a**2*d*f*n*u*w**2 + 2*Ae*Ap*Pa*a*b*f*j*n*u*w**2 + 2*Ae*Ap*Pa*a*d*f*j*n*u*w**2 + Ae*Ap*Pa*b*f*j**2*n*u*w**2 + Ae*Ap*Pa*d*f*j**2*n*u*w**2 + Ae*Ap*a**2*b*f*n*p*w**2 + Ae*Ap*a**2*d*f*n*p*w**2 + 2*Ae*Ap*a*b*f*j*n*p*w**2 + 2*Ae*Ap*a*d*f*j*n*p*w**2 + Ae*Ap*b*f*j**2*n*p*w**2 + Ae*Ap*d*f*j**2*n*p*w**2 + Ap**2*M*a**2*b*h*n*p*w + Ap**2*M*a**2*d*h*n*p*w + Ap**2*M*a*b*h*j*n*p*w + Ap**2*M*a*d*h*j*n*p*w)/(Ae**2*M*a**2*b**2*f*h*w + Ae**2*M*a**2*b*d*f*h*w + Ae**2*M*a*b**2*f*h*j*w + Ae**2*M*a*b*d*f*h*j*w + Ae**2*Me*a**2*b**2*f*h*w**2 + Ae**2*Me*a**2*b*d*f*h*w**2 + Ae**2*Me*a*b**2*f*h*j*w**2 + Ae**2*Me*a*b*d*f*h*j*w**2 + Ae*Ap*M**2*a**2*b**2*h**2 + Ae*Ap*M**2*a**2*b*d*h**2 + Ae*Ap*M*Me*a**2*b**2*h**2*w + Ae*Ap*M*Me*a**2*b*d*h**2*w + Ae*Ap*M*a**2*b*f*h*n*w + Ae*Ap*M*a**2*d*f*h*n*w + Ae*Ap*M*a*b*f*h*j*n*w + Ae*Ap*M*a*d*f*h*j*n*w + Ae*Ap*Me*a**2*b*f*h*n*w**2 + Ae*Ap*Me*a**2*d*f*h*n*w**2 + Ae*Ap*Me*a*b*f*h*j*n*w**2 + Ae*Ap*Me*a*d*f*h*j*n*w**2 + Ap**2*M**2*a**2*b*h**2*n + Ap**2*M**2*a**2*d*h**2*n + Ap**2*M*Me*a**2*b*h**2*n*w + Ap**2*M*Me*a**2*d*h**2*n*w), 
                     Pm: (Ae**2*Ea*a*b**2*f*u*w + Ae**2*Ea*b**2*f*j*u*w + Ae**2*a*b**2*f*p*w + Ae**2*a*b*d*f*p*w + Ae**2*b**2*f*j*p*w + Ae**2*b*d*f*j*p*w + Ae*Ap*Ea*a*b*f*n*u*w + Ae*Ap*Ea*b*f*j*n*u*w + Ae*Ap*M*a*b**2*h*p + Ae*Ap*M*a*b*d*h*p + Ae*Ap*Pa*a*b*f*n*u*w + Ae*Ap*Pa*a*d*f*n*u*w + Ae*Ap*Pa*b*f*j*n*u*w + Ae*Ap*Pa*d*f*j*n*u*w + Ae*Ap*a*b*f*n*p*w + Ae*Ap*a*d*f*n*p*w + Ae*Ap*b*f*j*n*p*w + Ae*Ap*d*f*j*n*p*w + Ap**2*M*a*b*h*n*p + Ap**2*M*a*d*h*n*p)/(Ae**2*a**2*b**2*f*w + Ae**2*a**2*b*d*f*w + Ae**2*a*b**2*f*j*w + Ae**2*a*b*d*f*j*w + Ae*Ap*M*a**2*b**2*h + Ae*Ap*M*a**2*b*d*h + Ae*Ap*a**2*b*f*n*w + Ae*Ap*a**2*d*f*n*w + Ae*Ap*a*b*f*j*n*w + Ae*Ap*a*d*f*j*n*w + Ap**2*M*a**2*b*h*n + Ap**2*M*a**2*d*h*n), 
                     Pu: Ae*Pa*u/(Ae*b + Ap*n)}
    dPa = a*Pm - u*Pa
    dEa = a*Em - u*Ea 
    
    # substitute
    dPa = sy.simplify(dPa.subs(Pm, steady_states[Pm]))
    dEa = sy.simplify(dEa.subs(Em, steady_states[Em]))

    return sy.solve([dPa, dEa], [Pa, Ea])

def M_substitution():
    """Compute the substitution for dM so everything is in terms of M"""
    # get symbols
    t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
    P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

    # steady states computed previously
    steady_states = {E: Ap**2*n*p*w*(a*b + a*d + b*j + d*j)/(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n),
                     Em: Ap**2*M*h*n*p*(b + d)/(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n),
                     Ea: (Ap**2*M*a*b*h*n*p + Ap**2*M*a*d*h*n*p)/(Ae**2*a*b**2*f*u*w + Ae**2*a*b*d*f*u*w + Ae**2*b**2*f*j*u*w + Ae**2*b*d*f*j*u*w + Ae*Ap*M*a*b*d*h*u + Ap**2*M*a*d*h*n*u), 
                     Eu: Ap**2*M*a*h*n*p/(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n),
                     P: p*w*(a + j)*(Ae*b + Ap*n)*(Ae*a*b*f*w + Ae*a*d*f*w + Ae*b*f*j*w + Ae*d*f*j*w + Ap*M*a*d*h)/(a*h*(M + Me*w)*(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n)),
                     Pm: p*(Ae*b + Ap*n)*(Ae*a*b*f*w + Ae*a*d*f*w + Ae*b*f*j*w + Ae*d*f*j*w + Ap*M*a*d*h)/(a*(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n)),
                     Pa: (Ae**2*a*b**2*f*p*w + Ae**2*a*b*d*f*p*w + Ae**2*b**2*f*j*p*w + Ae**2*b*d*f*j*p*w + Ae*Ap*M*a*b*d*h*p + Ae*Ap*a*b*f*n*p*w + Ae*Ap*a*d*f*n*p*w + Ae*Ap*b*f*j*n*p*w + Ae*Ap*d*f*j*n*p*w + Ap**2*M*a*d*h*n*p)/(Ae**2*a*b**2*f*u*w + Ae**2*a*b*d*f*u*w + Ae**2*b**2*f*j*u*w + Ae**2*b*d*f*j*u*w + Ae*Ap*M*a*b*d*h*u + Ap**2*M*a*d*h*n*u),
                     Pu: Ae*p*(Ae*a*b*f*w + Ae*a*d*f*w + Ae*b*f*j*w + Ae*d*f*j*w + Ap*M*a*d*h)/(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n)}

    # substitutions
    dM = -(h/w)*M*P - (h/w)*E*M + (j + u)*(Em + Pm) - vmax*M/(V*(Km + M))
    new_M1 = sy.simplify(dM.subs(P, steady_states[P]))
    new_M2 = sy.simplify(new_M1.subs(Pm, steady_states[Pm]))
    new_m3 = sy.simplify(new_M2.subs(Pa, steady_states[Pa]))
    new_m4 = sy.simplify(new_m3.subs(Pu, steady_states[Pu]))
    new_m5 = sy.simplify(new_m4.subs(E, steady_states[E]))
    new_m6 = sy.simplify(new_m5.subs(Em, steady_states[Em]))
    new_m7 = sy.simplify(new_m6.subs(Ea, steady_states[Ea]))
    new_m8 = sy.simplify(new_m7.subs(Eu, steady_states[Eu]))

    return new_m8



# parameters (filled in the ones I think would be the same or similar to Fur4)
p = 8.3e-5 #'' # units mup1 per millisecond (mup1 production rate)
h = 135 #'' # per micromolar per millisecond (methionine binding rate)
w = 32 # unitless (scale factor for pH difference)
j = 100 #'' # per millisecond (methionine unbinding rate)
f = .25 #'' # per millisecond (recycling rate)
Ae = 47 # micrometers^3 (endosomal membrane surface area)
Ap = 314 # micrometers^3 (plasma membrane surface area)
u = 1 # per millisecond (ubiquitination rate)
a = 1e-5 #'' # per micromolar per millisecond (art 1 binding rate)
b = 1 # per millisecond (deubiquitination rate)
d = .002 #'' # per millisecond (degradation rate)
n = 0.1 #'' # per millisecond (endocytosis rate)
V = 523 # micrometers^3 (volume of cytoplasm)
vmax = 8.8e3 #'' # micromolars*micrometers^3 per millisecond (maximal rate of methionine metabolism)
Km = 2.5 #'' # micromolars (methionin michaelis-menten constant)

# methionine (changes)
#Me = .1

M, Me = sy.symbols('M, Me')

# Now solve for M
def bisection_M(Me, dM=sy.lambdify(Me, (-Ap**2*M*V*a*h*n*p*(Km + M)*(M + Me*w)*(a*b + a*d + b*j + d*j) - M*V*p*(Km + M)*(a + j)*(Ae*b + Ap*n)*(Ae*a*b*f*w + Ae*a*d*f*w + Ae*b*f*j*w + Ae*d*f*j*w + Ap*M*a*d*h) 
                        - M*a*vmax*(M + Me*w)*(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n) 
                        + V*p*(Km + M)*(M + Me*w)*(j + u)*(Ap**2*M*a*h*n*(b + d) + (Ae*b + Ap*n)*(Ae*a*b*f*w + Ae*a*d*f*w + Ae*b*f*j*w + Ae*d*f*j*w + Ap*M*a*d*h)))/(V*a*(Km + M)*(M + Me*w)*(Ae**2*a*b**2*f*w + Ae**2*a*b*d*f*w + Ae**2*b**2*f*j*w + Ae**2*b*d*f*j*w + Ae*Ap*M*a*b*d*h + Ap**2*M*a*d*h*n))), 
                p=8.3e-5, h=135, w=32, j=100, f=.25, Ae=47, Ap=314, u=1,
                a=1e-5, b=1, d=.002, n=0.1, V=523, vmax=8.8e3, Km=2.5, bounds=[0, 400], maxiter=10000):
    """Solving for the steady state of M at a given methionine value Me
        
        Parameters:
        - Me (float): amount of extracellular methionine
        - dM (sympy lambda function): differential equation in terms of extracellular methionine input as independent variable
        - parameters (floats, optional): set to the values we currently have, can be changed as needed
        - bounds (list, optional): initial upper and lower bound for the bisection method"""
    # save dM with substitutions for all other variables (computed in steady_states_simplification.ipynb)
    dM = sy.simplify(dM(Me))
    dM_eq = sy.lambdify(M, dM)

    # assign variables for beginning, end, and midpoint
    x0, x1 = bounds[0], bounds[1]

    # use a loop to continue splitting the interval until you find 0
    for i in range(maxiter):
        # assign midpoint
        xmid = (x0 + x1) / 2    # so that it updates every time

        # check function at midpoint
        if np.isclose(dM_eq(xmid), 0).all():
            return xmid
        elif (dM_eq(xmid) > 0):   # if greater than zero, it should replace the left endpoint
            x0 = xmid
        else:                  # if less than zero, it should replace right endpoint
            x1 = xmid
    
    # raise runtime error if it doesn't converge
    #raise RuntimeError(f"Failed to converge in {maxiter} steps")
    return xmid

t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")


if __name__ == '__main__':
    # # establish initial conditions
    # initial = [.2, .1, .1, .1, .1, .1, .1, .1, .1]
    # times = np.linspace(0, 200, 200)
    # labels = ['P', 'Pm', 'Pa', 'Pu', 'E', 'Em', 'Ea', 'Eu', 'M']

    # # solve using solve_ivp
    # system = lambda t, y: model(y, t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km)
    # solution = solve_ivp(system, [times[0], times[-1]], initial, t_eval=times)

    # for i in range(8):
    #     plt.plot(times, solution.y[i], label=f'{labels[i]}(t) (solve_ivp)', linestyle='--')
    # #plt.plot(time_points, solution_solve_ivp.y[1], label='y2(t) (solve_ivp)', linestyle='--')
    # plt.xlabel('Time')
    # plt.ylabel('y(t)')
    # plt.title(f'Solution of the System of ODEs')
    # plt.legend(labels=['P', 'P_m', 'P_a', 'P_u', 'E', 'E_m', 'E_a', 'E_u', 'M'])
    # plt.show()

    # print(solve_except_Pa_Ua_M())

    ## use bisection method to plot intracellular methionine vs extracellular
    M_func = lambda m: bisection_M(Me=m)
    m_range = np.linspace(.001, .01, 100)
    
    # plot it
    plt.plot(m_range, [M_func(m) for m in m_range])
    plt.title("Intracellular Methionine vs Extracellular")
    plt.xlabel("extracellular")
    plt.ylabel("intracellular")
    plt.show()