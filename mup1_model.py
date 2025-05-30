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
        p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E,                       # P
        h*Me*P + (h/w)*M*P - j*Pm - a*Pm,                                 # Pm
        a*Pm - u*Pa,                                                      # Pa
        u*Pa - n*Pu,                                                      # Pu
        n*(Ap/Ae)*Pu - f*E + b*Eu - (h/w)*E*M + j*Em,                      # E
        (h/w)*E*M - a*Em - j*Em,                                          # Em
        a*Em - u*Ea,                                                      # Ea
        -b*Eu + u*Ea - d*Eu,                                             # Eu
        -(h/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + u)*((Ae / V)*Em + (Ap / V)*Pm) - vmax*M/(V*(Km + M))  # M 
        ]

    return dy


def solve():
    """Solve for steady states using sympy"""
    # set up sympy variables
    t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
    P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

    # functions to solve  
    dP = p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E                       # P
    dPm = h*Me*P + (h/w)*M*P - j*Pm - a*Pm                                 # Pm
    dPa = a*Pm - u*Pa                                                      # Pa
    dPu = u*Pa - n*Pu                                                      # Pu
    dE = n*(Ap/Ae)*Pu - f*E + b*Eu - (h/w)*E*M + j*Em                      # E
    dEm = (h/w)*E*M - a*Em - j*Em                                          # Em
    dEa = a*Em - u*Ea                                                      # Ea
    dEu = - b*Eu + u*Ea - d*Eu                                             # Eu
    dM = -(h/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + u)*((Ae / V)*Em + (Ap / V)*Pm) - vmax*M/(V*(Km + M))  # M 

    eqs = [dP, dPm, dPa, dPu, dE, dEm, dEa, dEu]
    vars = [P, Pm, Pa, Pu, E, Em, Ea, Eu]

    return sy.solve(eqs, vars)   


def M_substitution():
    """Compute the substitution for dM so everything is in terms of M"""
    # get symbols
    t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
    P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

    # steady states computed previously
    steady_states = {P : p*w*(M*a**2*d*h + M*a*d*h*j + a**2*b*f*w + a**2*d*f*w + 2*a*b*f*j*w + 2*a*d*f*j*w + b*f*j**2*w + d*f*j**2*w)/(M*a**2*d*h**2*(M + Me*w)),
                     Pm : p*(M*a*d*h + a*b*f*w + a*d*f*w + b*f*j*w + d*f*j*w)/(M*a**2*d*h),
                     Pa : p*(M*a*d*h + a*b*f*w + a*d*f*w + b*f*j*w + d*f*j*w)/(M*a*d*h*u),
                     Pu : p*(M*a*d*h + a*b*f*w + a*d*f*w + b*f*j*w + d*f*j*w)/(M*a*d*h*n),
                     E : Ap*p*w*(a*b + a*d + b*j + d*j)/(Ae*M*a*d*h),
                     Em : Ap*p*(b + d)/(Ae*a*d),
                     Ea : Ap*p*(b + d)/(Ae*d*u),
                     Eu : Ap*p/(Ae*d)}

    # substitutions
    dM = -(h/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + u)*((Ae / V)*Em + (Ap / V)*Pm) - vmax*M/(V*(Km + M))  # M 
    new_dM = dM.subs(steady_states)

    return new_dM



# parameters (filled in the ones I think would be the same or similar to Fur4)
p = 8.3e-5 #'' # units mup1 per millisecond (mup1 production rate)
h = 100 / 2188 #'' # per micromolar per millisecond (methionine binding rate)
w = 32 # unitless (scale factor for pH difference)
j = 100 #'' # per millisecond (methionine unbinding rate)
f = .25 #'' # per millisecond (recycling rate)
Ae = 47 # micrometers^3 (endosomal membrane surface area)
Ap = 314 # micrometers^3 (plasma membrane surface area)
u = 1 # per millisecond (ubiquitination rate)
a = 10 #'' # per micromolar per millisecond (art 1 binding rate)
b = 1 # per millisecond (deubiquitination rate)
d = .002 #'' # per millisecond (degradation rate)
n = 0.1 #'' # per millisecond (endocytosis rate)
V = 523 # micrometers^3 (volume of cytoplasm)
vmax = 174333.33 #'' # micromolars*micrometers^3 per millisecond (maximal rate of methionine metabolism)
Km = 350 #'' # micromolars (methionin michaelis-menten constant)
# vmax = 8.8e3
# Km = 2.5

M, Me = sy.symbols('M, Me')

# Now solve for M
def bisection_M(m, dM=sy.lambdify(Me, (-Ap*M*p*(Km + M)*(M*a**2*d*h + M*a*d*h*j + a**2*b*f*w + a**2*d*f*w + 2*a*b*f*j*w + 2*a*d*f*j*w + a*h*(M + Me*w)*(a*b + a*d + b*j + d*j) + b*f*j**2*w + d*f*j**2*w) 
                                       + Ap*p*(Km + M)*(M + Me*w)*(j + u)*(M*a*d*h + M*a*h*(b + d) + a*b*f*w + a*d*f*w + b*f*j*w + d*f*j*w) 
                                       - M**2*a**2*d*h*vmax*(M + Me*w))/(M*V*a**2*d*h*(Km + M)*(M + Me*w))), 
                p=8.3e-5, h=135, w=32, j=100, f=.25, Ae=47, Ap=314, u=1,
                a=1e-5, b=1, d=.002, n=0.1, V=523, vmax=8.8e3, Km=2.5, bounds=[0, 4000], maxiter=1000000):
    """Solving for the steady state of M at a given methionine value Me
        
        Parameters:
        - Me (float): amount of extracellular methionine
        - dM (sympy lambda function): differential equation in terms of extracellular methionine input as independent variable
        - parameters (floats, optional): set to the values we currently have, can be changed as needed
        - bounds (list, optional): initial upper and lower bound for the bisection method"""
    # save dM with substitutions for all other variables (computed in steady_states_simplification.ipynb)
    dM = sy.simplify(dM(m))
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
    raise RuntimeError(f"Failed to converge in {maxiter} steps")
    #return xmid

# t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km = sy.symbols("t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km")
# P, Pm, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pm, Pa, Pu, M, E, Em, Ea, Eu")

# methionine (changes)
Me = .1

if __name__ == '__main__':
    # establish initial conditions
    initial = [10, 10, 10, 10, 10, 10, 10, 10, 500]
    times = np.linspace(0, 100, 200)
    labels = ['P', 'Pm', 'Pa', 'Pu', 'E', 'Em', 'Ea', 'Eu', 'M']

    # solve using solve_ivp
    system = lambda t, y: model(y, t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km)
    solution = solve_ivp(system, [times[0], times[-1]], initial, t_eval=times)

    for i in range(8):
        plt.plot(times, solution.y[i], label=f'{labels[i]}(t) (solve_ivp)', linestyle='--')
    #plt.plot(time_points, solution_solve_ivp.y[1], label='y2(t) (solve_ivp)', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.title(f'Solution of the System of ODEs')
    plt.legend(labels=['P', 'P_m', 'P_a', 'P_u', 'E', 'E_m', 'E_a', 'E_u', 'M'])
    plt.show()

    # ## use bisection method to plot intracellular methionine vs extracellular
    # M_func = lambda m: bisection_M(m)
    # m_range = np.linspace(.01, 20, 100)
    
    # # plot it
    # plt.plot(m_range, [M_func(m) for m in m_range])
    # plt.title("Intracellular Methionine vs Extracellular")
    # plt.xlabel("extracellular")
    # plt.ylabel("intracellular")
    # plt.show()

    # # code to print solve() formatted nicely
    # steady_states = solve()
    # for key, value in steady_states.items():
    #     print(f"{key} = {value}")