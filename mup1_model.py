import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sy

def model(state, t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km):
    """function to establish the system in the model
    
       Parameters:
       - state (ndarray): array of dependent variables (P, Pb, Pa, Pu, E, Em, Ea, Eu, M)
       - t (float): time
       - others (float): parameters for the equations (will be given values as we find them)

       Returns:
       - array: vector of derivatives at time t
    """
    # unpack the variables (for readability)
    P, Pb, Pa, Pu, E, Em, Ea, Eu, M = state
    
    # set up the equations
    dy = [
        y - k*Me*P - (k/w)*M*P + j*Pb + f*(Ae/Ap)*E,                       # P
        k*Me*P + (k/w)*M*P - j*Pb - h*Pb,                                 # Pb
        h*Pb - a*Pa,                                                      # Pa
        a*Pa - g*Pu,                                                      # Pu
        g*(Ap/Ae)*Pu - f*E + b*Eu - (k/w)*E*M + j*Em,                      # E
        (k/w)*E*M - h*Em - j*Em,                                          # Em
        h*Em - a*Ea,                                                      # Ea
        -b*Eu + a*Ea - z*Eu,                                             # Eu
        -(k/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + a)*((Ae / V)*Em + (Ap / V)*Pb) - vmax*M/(V*(Km + M))  # M 
        ]

    return dy


def solve():
    """Solve for steady states using sympy"""
    # set up sympy variables
    t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km = sy.symbols("t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km")
    P, Pb, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pb, Pa, Pu, M, E, Em, Ea, Eu")

    # functions to solve  
    dP = y - k*Me*P - (k/w)*M*P + j*Pb + f*(Ae/Ap)*E                       # P
    dPb = k*Me*P + (k/w)*M*P - j*Pb - h*Pb                                 # Pb
    dPa = h*Pb - a*Pa                                                      # Pa
    dPu = a*Pa - g*Pu                                                      # Pu
    dE = g*(Ap/Ae)*Pu - f*E + b*Eu - (k/w)*E*M + j*Em                      # E
    dEm = (k/w)*E*M - h*Em - j*Em                                          # Em
    dEa = h*Em - a*Ea                                                      # Ea
    dEu = - b*Eu + a*Ea - z*Eu                                             # Eu
    dM = -(k/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + a)*((Ae / V)*Em + (Ap / V)*Pb) - vmax*M/(V*(Km + M))  # M 

    eqs = [dP, dPb, dPa, dPu, dE, dEm, dEa, dEu]
    vars = [P, Pb, Pa, Pu, E, Em, Ea, Eu]

    return sy.solve(eqs, vars)   


def M_substitution():
    """Compute the substitution for dM so everything is in terms of M"""
    # get symbols
    t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km = sy.symbols("t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km")
    P, Pb, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pb, Pa, Pu, M, E, Em, Ea, Eu")

    # steady states computed previously
    steady_states = {P : y*w*(M*h**2*z*k + M*h*z*k*j + h**2*b*f*w + h**2*z*f*w + 2*h*b*f*j*w + 2*h*z*f*j*w + b*f*j**2*w + z*f*j**2*w)/(M*h**2*z*k**2*(M + Me*w)),
                     Pb : y*(M*h*z*k + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w)/(M*h**2*z*k),
                     Pa : y*(M*h*z*k + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w)/(M*h*z*k*a),
                     Pu : y*(M*h*z*k + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w)/(M*h*z*k*g),
                     E : Ap*y*w*(h*b + h*z + b*j + z*j)/(Ae*M*h*z*k),
                     Em : Ap*y*(b + z)/(Ae*h*z),
                     Ea : Ap*y*(b + z)/(Ae*z*a),
                     Eu : Ap*y/(Ae*z)}

    # substitutions
    dM = -(k/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + a)*((Ae / V)*Em + (Ap / V)*Pb) - vmax*M/(V*(Km + M))  # M 
    new_dM = dM.subs(steady_states)

    return new_dM



# parameters (filled in the ones I think would be the same or similar to Fur4)
y = 8.3e-5 #'' # units mup1 per millisecond (mup1 production rate)
k = 100 / 2188 #'' # per micromolar per millisecond (methionine binding rate)
w = 32 # unitless (scale factor for pH difference)
j = 100 #'' # per millisecond (methionine unbinding rate)
f = .25 #'' # per millisecond (recycling rate)
Ae = 47 # micrometers^3 (endosomal membrane surface area)
Ap = 314 # micrometers^3 (plasma membrane surface area)
a = 1 # per millisecond (ubiquitination rate)
h = 10 #'' # per micromolar per millisecond (art 1 binding rate)
b = 1 # per millisecond (deubiquitination rate)
z = .002 #'' # per millisecond (degradation rate)
g = 0.1 #'' # per millisecond (endocytosis rate)
V = 523 # micrometers^3 (volume of cytoplasm)
vmax = 174333.33 #'' # micromolars*micrometers^3 per millisecond (maximal rate of methionine metabolism)
Km = 350 #'' # micromolars (methionin michaelis-menten constant)
# vmax = 8.8e3
# Km = 2.5

M, Me = sy.symbols('M, Me')

# Now solve for M
def bisection_M(m, dM=sy.lambdify(Me, (-Ap*M*y*(Km + M)*(M*h**2*z*k + M*h*z*k*j + h**2*b*f*w + h**2*z*f*w + 2*h*b*f*j*w + 2*h*z*f*j*w + h*k*(M + Me*w)*(h*b + h*z + b*j + z*j) + b*f*j**2*w + z*f*j**2*w) 
                                       + Ap*y*(Km + M)*(M + Me*w)*(j + a)*(M*h*z*k + M*h*k*(b + z) + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w) 
                                       - M**2*h**2*z*k*vmax*(M + Me*w))/(M*V*h**2*z*k*(Km + M)*(M + Me*w))), 
                y=8.3e-5, k=135, w=32, j=100, f=.25, Ae=47, Ap=314, a=1,
                h=1e-5, b=1, z=.002, g=0.1, V=523, vmax=8.8e3, Km=2.5, bounds=[0, 4000], maxiter=1000000):
    """Solving for the steady state of M at a given methionine value Me
        
        Parameters:
        - Me (float): amount of extracellular methionine
        - dM (sympy lambda function): differential equation in terms of extracellular methionine input as independent variable
        - parameters (floats, optional): set to the values we currently have, can be changed as needed
        - bounds (list, optional): initial upper and lower bound for the bisection method"""
    # save dM with substitutions for all other variables (computed in steady_states_simplification.ipynb)
    dM = sy.simplify(dM(m))
    dM_eq = sy.lambdify(M, dM)
    print(dM_eq(500))

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

def plot_mup1_and_me():
    """Plot total Mup1 against extracellular methionine to track the impact of changes of extracellular methionine
    on total Mup1. Also plot plasma membrane Mup1 and endosomal Mup1 separately against extracellular methionine."""

    # set sympy variables for steady state keys and methionine
    P, Pb, Pa, Pu, M, E, Em, Ea, Eu, Me = sy.symbols("P, Pb, Pa, Pu, M, E, Em, Ea, Eu, Me")

    # store steady states
    steady_states = {P : y*w*(M*h**2*z*k + M*h*z*k*j + h**2*b*f*w + h**2*z*f*w + 2*h*b*f*j*w + 2*h*z*f*j*w + b*f*j**2*w + z*f*j**2*w)/(M*h**2*z*k**2*(M + Me*w)),
                     Pb : y*(M*h*z*k + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w)/(M*h**2*z*k),
                     Pa : y*(M*h*z*k + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w)/(M*h*z*k*a),
                     Pu : y*(M*h*z*k + h*b*f*w + h*z*f*w + b*f*j*w + z*f*j*w)/(M*h*z*k*g),
                     E : Ap*y*w*(h*b + h*z + b*j + z*j)/(Ae*M*h*z*k),
                     Em : Ap*y*(b + z)/(Ae*h*z),
                     Ea : Ap*y*(b + z)/(Ae*z*a),
                     Eu : Ap*y/(Ae*z)}    
    
    # create lambda functions for the steady states
    M_func = lambda m: bisection_M(m)

    P_func = sy.lambdify((Me, M), steady_states[P])  # plasma membrane
    Pb_func = sy.lambdify((Me, M), steady_states[Pb])
    Pa_func = sy.lambdify((Me, M), steady_states[Pa])
    Pu_func = sy.lambdify((Me, M), steady_states[Pu])

    E_func = sy.lambdify((Me, M), steady_states[E])  # endosome
    Em_func = sy.lambdify((Me, M), steady_states[Em])
    Ea_func = sy.lambdify((Me, M), steady_states[Ea])
    Eu_func = sy.lambdify((Me, M), steady_states[Eu])

    # get the plasma membrane, endosome, and overall totals
    plasma_membrane = lambda m: P_func(m, M_func(m)) + Pb_func(m, M_func(m)) + Pa_func(m, M_func(m)) + Pu_func(m, M_func(m))
    endosome = lambda m: E_func(m, M_func(m)) + Em_func(m, M_func(m)) + Ea_func(m, M_func(m)) + Eu_func(m, M_func(m))
    total = lambda m: plasma_membrane(m) + endosome(m)

    # now plot it
    m_vals = np.linspace(5, 20, 50)
    plt.plot(m_vals, [total(m) for m in m_vals], label="Total")
    plt.plot(m_vals, [endosome(m) for m in m_vals], label="Endosome")
    plt.plot(m_vals, [plasma_membrane(m) for m in m_vals], label="Plasma Membrane")
    plt.title("Mup1 and Methionine")
    plt.xlabel("Extracellular Methionine")
    plt.ylabel("Mup1")
    plt.legend()
    plt.show()


# t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km = sy.symbols("t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km")
# P, Pb, Pa, Pu, M, E, Em, Ea, Eu = sy.symbols("P, Pb, Pa, Pu, M, E, Em, Ea, Eu")

# methionine (changes)
Me = .1

if __name__ == '__main__':
    plot_mup1_and_me()

    # # establish initial conditions
    # initial = [10, 10, 10, 10, 10, 10, 10, 10, 500]
    # times = np.linspace(0, 100, 200)
    # labels = ['P', 'Pb', 'Pa', 'Pu', 'E', 'Em', 'Ea', 'Eu', 'M']

    # # solve using solve_ivp
    # system = lambda t, y: model(y, t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km)
    # solution = solve_ivp(system, [times[0], times[-1]], initial, t_eval=times)

    # for i in range(8):
    #     plt.plot(times, solution.y[i], label=f'{labels[i]}(t) (solve_ivp)', linestyle='--')
    # #plt.plot(time_points, solution_solve_ivp.y[1], label='y2(t) (solve_ivp)', linestyle='--')
    # plt.xlabel('Time')
    # plt.ylabel('y(t)')
    # plt.title(f'Solution of the System of ODEs')
    # plt.legend(labels=['P', 'P_m', 'P_a', 'P_u', 'E', 'E_m', 'E_a', 'E_u', 'M'])
    # plt.show()

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