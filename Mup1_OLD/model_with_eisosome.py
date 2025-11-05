import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sy

def eisosome_model(states, t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2):
    """function to establish the system in the model with eisosome
    
    Parameters:
    - states (ndarray): array of dependent variables (P, Pe, Pb, Pa, Pu, E, Em, Ea, Eu, M)
    - t (float): time
    - params (float): parameters for the equations (will be given values as we find them)

    Returns:
    - dy (ndarray): vector of derivatives at time t
    """
    # unpack the variables (for readability)
    P, Pe, Pb, Pa, Pu, E, Em, Ea, Eu, M = states
    
    # set up the equations
    dy = [
        y - k*Me*P - (k/w)*M*P + j*Pb + f*(Ae/Ap)*E + k1*Pe - k2*P,        # P
        k2*P - k1*Pe,                                                      # Pe
        k*Me*P + (k/w)*M*P - j*Pb - h*Pb,                                  # Pb
        h*Pb - a*Pa,                                                       # Pa
        a*Pa - g*Pu,                                                       # Pu
        g*(Ap/Ae)*Pu - f*E + b*Eu - (k/w)*E*M + j*Em,                      # E
        (k/w)*E*M - h*Em - j*Em,                                           # Em
        h*Em - a*Ea,                                                       # Ea
        -b*Eu + a*Ea - z*Eu,                                               # Eu
        -(k/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + a)*((Ae / V)*Em + (Ap / V)*Pb) - vmax*M/(V*(Km + M))  # M
    ]
    
    return dy

def plot_model(Me, params = [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1]):
    """Use solve_ivp to plot the model.
    
    Parameters:
    - Me (float) : extracellular methionine concentration
    - params (list) : list of parameter values [y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2]"""
    
    y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2 = params
    
    # establish initial conditions
    initial = [10, 0, 10, 10, 10, 10, 10, 10, 10, 500]
    times = np.linspace(0, 60, 200)
    labels = ['P', 'Pe', 'Pb', 'Pa', 'Pu', 'E', 'Em', 'Ea', 'Eu', 'M']

    # solve using solve_ivp
    system = lambda t, states: eisosome_model(states, t, Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2)
    solution = solve_ivp(system, [times[0], times[-1]], initial, t_eval=times)

    for i in range(9):
        plt.plot(times, solution.y[i], label=f'{labels[i]}', linestyle='--')
    #plt.plot(time_points, solution_solve_ivp.y[1], label='y2(t) (solve_ivp)', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    # plt.legend(labels=['P', 'P_e', 'P_m', 'P_a', 'P_u', 'E', 'E_m', 'E_a', 'E_u', 'M'])
    # plt.show()


def compute_steady_states():
    """Code to compute the symbolic steady states of the eisosome
    model. Computes the steady states for all except dM.
    
    Returns
    -------
    steady_states : dict
        Dictionary of steady states
    new_dM : sympy equation
        dM with steady states for other equations substituted in"""
    
    # Define sympy variables
    P, Pe, Pb, Pa, Pu, E, Em, Ea, Eu, M = sy.symbols("P, Pe, Pb, Pa, Pu, E, Em, Ea, Eu, M")
    Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2 = sy.symbols("Me, y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2")

    # Define differential equations
    dP  = y - k*Me*P - (k/w)*M*P + j*Pb + f*(Ae/Ap)*E + k1*Pe - k2*P
    dPe = k2*P - k1*Pe
    dPb = k*Me*P + (k/w)*M*P - j*Pb - h*Pb
    dPa = h*Pb - a*Pa
    dPu = a*Pa - g*Pu
    dE  = g*(Ap/Ae)*Pu - f*E + b*Eu - (k/w)*E*M + j*Em
    dEm = (k/w)*E*M - h*Em - j*Em
    dEa = h*Em - a*Ea
    dEu = -b*Eu + a*Ea - z*Eu
    dM  = -(k/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + a)*((Ae / V)*Em + (Ap / V)*Pb) - vmax*M/(V*(Km + M))

    # Make lists with equations and variables
    eqs = [dP, dPe, dPb, dPa, dPu, dE, dEm, dEa, dEu]
    var = [P,  Pe,  Pb,  Pa,  Pu,  E,  Em,  Ea,  Eu]

    # Use sympy to solve symbolically
    steady_states = sy.solve(eqs, var)

    # substitute into dM
    new_dM = dM.subs(steady_states)

    return steady_states, new_dM

### GLOBAL VARIABLES ###
y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2 = [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1]
M, Me = sy.symbols("M, Me")
STEADY_STATES = {'E': (Ap*h*b*y*w + Ap*h*z*y*w + Ap*b*j*y*w + Ap*z*j*y*w)/(Ae*M*h*z*k), 
                 'Ea': (Ap*b*y + Ap*z*y)/(Ae*z*a), 
                 'Em': (Ap*b*y + Ap*z*y)/(Ae*h*z), 
                 'Eu': Ap*y/(Ae*z), 
                 'P': (M*h**2*z*k*y*w + M*h*z*k*j*y*w + h**2*b*f*y*w**2 + h**2*z*f*y*w**2 + 2*h*b*f*j*y*w**2 + 2*h*z*f*j*y*w**2 + b*f*j**2*y*w**2 + z*f*j**2*y*w**2)/(M**2*h**2*z*k**2 + M*Me*h**2*z*k**2*w), 
                 'Pa': (M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*h*z*k*a), 
                 'Pe': (M*h**2*z*k*k2*y*w + M*h*z*k*j*k2*y*w + h**2*b*f*k2*y*w**2 + h**2*z*f*k2*y*w**2 + 2*h*b*f*j*k2*y*w**2 + 2*h*z*f*j*k2*y*w**2 + b*f*j**2*k2*y*w**2 + z*f*j**2*k2*y*w**2)/(M**2*h**2*z*k**2*k1 + M*Me*h**2*z*k**2*k1*w), 
                 'Pb': (M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*h**2*z*k), 
                 'Pu': (M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*h*z*k*g)}
dM_EQ = -M*k*(Ap*(M*h**2*z*k*y*w + M*h*z*k*j*y*w + h**2*b*f*y*w**2 + h**2*z*f*y*w**2 + 2*h*b*f*j*y*w**2 + 2*h*z*f*j*y*w**2 + b*f*j**2*y*w**2 + z*f*j**2*y*w**2)/(V*(M**2*h**2*z*k**2 + M*Me*h**2*z*k**2*w)) + (Ap*h*b*y*w + Ap*h*z*y*w + Ap*b*j*y*w + Ap*z*j*y*w)/(M*V*h*z*k))/w - M*vmax/(V*(Km + M)) + (j + a)*(Ap*(M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*V*h**2*z*k) + (Ap*b*y + Ap*z*y)/(V*h*z))

def bisection_method(Me, dM, params = [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1],
                     M=sy.symbols("M"), bounds=[0, 4000], maxiter=1000):
    """Solve for M using the bisection method.
    
    Parameters
    ----------
    Me (float) : Extracellular methionine
    dM (function) : new_dM computed using compute_steady_states (still with sympy variables)
    params (list) : Parameter values in order [y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2]
    M (sympy var) : Sympy symbol for M
    bounds (list) : List with start and end bounds for bisection method
    maxiter (int) : Stopping criteria

    Returns
    -------
    Solution to bisection method (zero of dM within the bounds)
    """

    # get params for substitution
    y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2 = sy.symbols("y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2")
    symbols_list = [y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2]
    param_dict = {symbols_list[i]: param for i, param in enumerate(params)}

    # save dM with substitutions for all other variables (computed in steady_states_simplification.ipynb)
    dM = sy.simplify(dM(Me))
    new_dM = dM.subs(param_dict)
    dM_eq = sy.lambdify(M, new_dM)

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


def plot_methionine():
    """Code to plot extracellular vs intracellular methionine."""
    # get dM equation
    steady_states, dM_symbolic = compute_steady_states()

    # lambdify dM equation
    Me = sy.symbols("Me")
    dM = sy.lambdify(Me, dM_symbolic, 'numpy')

    # use bisection method to graph Me
    met_vals = np.linspace(.1, 25, 50)
    M_vals = np.array([bisection_method(m, dM) for m in met_vals])

    # plot Me and M
    plt.plot(met_vals, M_vals)
    plt.title("Eisosome Methionine\nextracellular vs intracellular")
    plt.xlabel("Me")
    plt.ylabel("M")
    plt.show()


def plot_varying_transition_rates():
    param_changes = [[8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 10], [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1], [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 10, 1]]
    plt.figure(figsize=(12, 6))
    for i, param_list in enumerate(param_changes):
        plt.subplot(1, 3, i + 1)
        plot_model(10, params=param_list)
        plt.title(f"k1 = {param_list[-2]}, k2 = {param_list[-1]}")
        if i == 0:
            plt.legend()
    plt.suptitle("Varying eisosome transition rates")
    plt.tight_layout()
    plt.show()


def mup1_methionine_plot(parameters=[8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 10, 1]):
    """Code to plot total amount of Mup1 against amount of extracellular methionine."""
    # get Me values for plotting
    Me_vals = np.linspace(1, 50, 100)
    y, k, w, j, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, k1, k2 = parameters

    # Define steady state equations and methionine equation
    M, Me = sy.symbols("M, Me")
    dM = -M*k*(Ap*(M*h**2*z*k*y*w + M*h*z*k*j*y*w + h**2*b*f*y*w**2 + h**2*z*f*y*w**2 + 2*h*b*f*j*y*w**2 + 2*h*z*f*j*y*w**2 + b*f*j**2*y*w**2 + z*f*j**2*y*w**2)/(V*(M**2*h**2*z*k**2 + M*Me*h**2*z*k**2*w)) + (Ap*h*b*y*w + Ap*h*z*y*w + Ap*b*j*y*w + Ap*z*j*y*w)/(M*V*h*z*k))/w - M*vmax/(V*(Km + M)) + (j + a)*(Ap*(M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*V*h**2*z*k) + (Ap*b*y + Ap*z*y)/(V*h*z))
    dM_eq = sy.lambdify(Me, dM)
    steady_states = {"E": (Ap*h*b*y*w + Ap*h*z*y*w + Ap*b*j*y*w + Ap*z*j*y*w)/(Ae*M*h*z*k), 
                     "Ea": (Ap*b*y + Ap*z*y)/(Ae*z*a), 
                     "Em": (Ap*b*y + Ap*z*y)/(Ae*h*z), 
                     "Eu": Ap*y/(Ae*z), 
                     "P": (M*h**2*z*k*y*w + M*h*z*k*j*y*w + h**2*b*f*y*w**2 + h**2*z*f*y*w**2 + 2*h*b*f*j*y*w**2 + 2*h*z*f*j*y*w**2 + b*f*j**2*y*w**2 + z*f*j**2*y*w**2)/(M**2*h**2*z*k**2 + M*Me*h**2*z*k**2*w), 
                     "Pa": (M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*h*z*k*a), 
                     "Pe": (M*h**2*z*k*k2*y*w + M*h*z*k*j*k2*y*w + h**2*b*f*k2*y*w**2 + h**2*z*f*k2*y*w**2 + 2*h*b*f*j*k2*y*w**2 + 2*a*z*f*j*k2*y*w**2 + b*f*j**2*k2*y*w**2 + z*f*j**2*k2*y*w**2)/(M**2*h**2*z*k**2*k1 + M*Me*h**2*z*k**2*k1*w), 
                     "Pb": (M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*h**2*z*k), 
                     "Pu": (M*h*z*k*y + h*b*f*y*w + h*z*f*y*w + b*f*j*y*w + z*f*j*y*w)/(M*h*z*k*g)}
    
    # Set up equations for plotting
    Pb_eq = sy.lambdify([M, Me], steady_states['P'] + steady_states['Pa'] + steady_states['Pb'] + steady_states['Pu'])
    Endosome_eq = sy.lambdify([M, Me], steady_states['E'] + steady_states['Ea'] + steady_states["Em"] + steady_states["Eu"])
    eisosome_only = sy.lambdify([M, Me], steady_states['Pe'])
    total = lambda M, Me: Pb_eq(M, Me) + Endosome_eq(M, Me) + eisosome_only(M, Me)
    M_eq = lambda Me: bisection_method(Me, dM_eq, params=parameters)

    # plot equations
    plt.plot(Me_vals, [Pb_eq(M_eq(m), m) for m in Me_vals], label="Plasma Membrane")
    plt.plot(Me_vals, [Endosome_eq(M_eq(m), m) for m in Me_vals], label="Endosome")
    plt.plot(Me_vals, [eisosome_only(M_eq(m), m) for m in Me_vals], label="Eisosome")
    plt.plot(Me_vals, [total(M_eq(m), m) for m in Me_vals], label="Total")
    plt.title("Mup1 Levels in a Yeast Cell")
    plt.xlabel("Extracellular Methionine")
    plt.ylabel("Mup1")
    plt.legend()
    plt.show()

mup1_methionine_plot()
