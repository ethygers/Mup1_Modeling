import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sy

def eisosome_model(y, t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2):
    """function to establish the system in the model
    
       Parameters:
       - y (ndarray): array of dependent variables (P, Pe, Pm, Pa, Pu, E, Em, Ea, Eu, M)
       - t (float): time
       - others (float): parameters for the equations (will be given values as we find them)

       Returns:
       - array: vector of derivatives at time t
    """
    # unpack the variables (for readability)
    P, Pe, Pm, Pa, Pu, E, Em, Ea, Eu, M = y
    
    # set up the equations
    dy = [
        p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E + k1*Pe - k2*P,        # P
        k2*P - k1*Pe,                                                      # Pe
        h*Me*P + (h/w)*M*P - j*Pm - a*Pm,                                  # Pm
        a*Pm - u*Pa,                                                       # Pa
        u*Pa - n*Pu,                                                       # Pu
        n*(Ap/Ae)*Pu - f*E + b*Eu - (h/w)*E*M + j*Em,                      # E
        (h/w)*E*M - a*Em - j*Em,                                           # Em
        a*Em - u*Ea,                                                       # Ea
        -b*Eu + u*Ea - d*Eu,                                               # Eu
        -(h/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + u)*((Ae / V)*Em + (Ap / V)*Pm) - vmax*M/(V*(Km + M))  # M 
        ]

    return dy

def plot_model(Me, params = [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1]):
    """Use solve_ivp to plot the model.
    
    Parameters:
     - Me (float) : amount of extracellular methionine
     - params (list) : list of parameter values [p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2]"""
    
    # unpack parameter list
    p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2 = params

    # establish initial conditions
    initial = [10, 0, 10, 10, 10, 10, 10, 10, 10, 500]
    times = np.linspace(0, 60, 200)
    labels = ['P', 'Pe', 'Pm', 'Pa', 'Pu', 'E', 'Em', 'Ea', 'Eu', 'M']

    # solve using solve_ivp
    system = lambda t, y: eisosome_model(y, t, Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2)
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
    P, Pe, Pm, Pa, Pu, E, Em, Ea, Eu, M = sy.symbols("P, Pe, Pm, Pa, Pu, E, Em, Ea, Eu, M")
    Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2 = sy.symbols("Me, p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2")

    # Define differential equations
    dP  = p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E + k1*Pe - k2*P
    dPe = k2*P - k1*Pe
    dPm = h*Me*P + (h/w)*M*P - j*Pm - a*Pm
    dPa = a*Pm - u*Pa
    dPu = u*Pa - n*Pu
    dE  = n*(Ap/Ae)*Pu - f*E + b*Eu - (h/w)*E*M + j*Em
    dEm = (h/w)*E*M - a*Em - j*Em
    dEa = a*Em - u*Ea
    dEu = -b*Eu + u*Ea - d*Eu
    dM  = -(h/w)*M*((Ap / V)*P + (Ae / V)*E) + (j + u)*((Ae / V)*Em + (Ap / V)*Pm) - vmax*M/(V*(Km + M))

    # Make lists with equations and variables
    eqs = [dP, dPe, dPm, dPa, dPu, dE, dEm, dEa, dEu]
    var = [P,  Pe,  Pm,  Pa,  Pu,  E,  Em,  Ea,  Eu]

    # Use sympy to solve symbolically
    steady_states = sy.solve(eqs, var)

    # substitute into dM
    new_dM = dM.subs(steady_states)

    return steady_states, new_dM

### GLOBAL VARIABLES ###
p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2 = [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1]
M, Me = sy.symbols("M, Me")
STEADY_STATES = {'E': (Ap*a*b*p*w + Ap*a*d*p*w + Ap*b*j*p*w + Ap*d*j*p*w)/(Ae*M*a*d*h), 
                 'Ea': (Ap*b*p + Ap*d*p)/(Ae*d*u), 
                 'Em': (Ap*b*p + Ap*d*p)/(Ae*a*d), 
                 'Eu': Ap*p/(Ae*d), 
                 'P': (M*a**2*d*h*p*w + M*a*d*h*j*p*w + a**2*b*f*p*w**2 + a**2*d*f*p*w**2 + 2*a*b*f*j*p*w**2 + 2*a*d*f*j*p*w**2 + b*f*j**2*p*w**2 + d*f*j**2*p*w**2)/(M**2*a**2*d*h**2 + M*Me*a**2*d*h**2*w), 
                 'Pa': (M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*a*d*h*u), 
                 'Pe': (M*a**2*d*h*k2*p*w + M*a*d*h*j*k2*p*w + a**2*b*f*k2*p*w**2 + a**2*d*f*k2*p*w**2 + 2*a*b*f*j*k2*p*w**2 + 2*a*d*f*j*k2*p*w**2 + b*f*j**2*k2*p*w**2 + d*f*j**2*k2*p*w**2)/(M**2*a**2*d*h**2*k1 + M*Me*a**2*d*h**2*k1*w), 
                 'Pm': (M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*a**2*d*h), 
                 'Pu': (M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*a*d*h*n)}, 
dM_EQ = -M*h*(Ap*(M*a**2*d*h*p*w + M*a*d*h*j*p*w + a**2*b*f*p*w**2 + a**2*d*f*p*w**2 + 2*a*b*f*j*p*w**2 + 2*a*d*f*j*p*w**2 + b*f*j**2*p*w**2 + d*f*j**2*p*w**2)/(V*(M**2*a**2*d*h**2 + M*Me*a**2*d*h**2*w)) + (Ap*a*b*p*w + Ap*a*d*p*w + Ap*b*j*p*w + Ap*d*j*p*w)/(M*V*a*d*h))/w - M*vmax/(V*(Km + M)) + (j + u)*(Ap*(M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*V*a**2*d*h) + (Ap*b*p + Ap*d*p)/(V*a*d))

def bisection_method(Me, dM, params = [8.3e-5, 100 / 2188, 32, 100, .25, 47, 314, 1, 10, 1, .002, 0.1, 523, 174333.33, 350, 1, 1],
                     M=sy.symbols("M"), bounds=[0, 4000], maxiter=1000):
    """Solve for M using the bisection method.
    
    Parameters
    ----------
    Me (float) : Extracellular methionine
    dM (function) : new_dM computed using compute_steady_states (still with sympy variables)
    params (list) : Parameter values in order [p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2]
    M (sympy var) : Sympy symbol for M
    bounds (list) : List with start and end bounds for bisection method
    maxiter (int) : Stopping criteria

    Returns
    -------
    Solution to bisection method (zero of dM within the bounds)
    """

    # get params for substitution
    p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2 = sy.symbols("p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2")
    symbols_list = [p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2]
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
    p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2 = parameters

    # Define steady state equations and methionine equation
    M, Me = sy.symbols("M, Me")
    dM = -M*h*(Ap*(M*a**2*d*h*p*w + M*a*d*h*j*p*w + a**2*b*f*p*w**2 + a**2*d*f*p*w**2 + 2*a*b*f*j*p*w**2 + 2*a*d*f*j*p*w**2 + b*f*j**2*p*w**2 + d*f*j**2*p*w**2)/(V*(M**2*a**2*d*h**2 + M*Me*a**2*d*h**2*w)) + (Ap*a*b*p*w + Ap*a*d*p*w + Ap*b*j*p*w + Ap*d*j*p*w)/(M*V*a*d*h))/w - M*vmax/(V*(Km + M)) + (j + u)*(Ap*(M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*V*a**2*d*h) + (Ap*b*p + Ap*d*p)/(V*a*d))
    dM_eq = sy.lambdify(Me, dM)
    steady_states = {"E": (Ap*a*b*p*w + Ap*a*d*p*w + Ap*b*j*p*w + Ap*d*j*p*w)/(Ae*M*a*d*h), 
                     "Ea": (Ap*b*p + Ap*d*p)/(Ae*d*u), 
                     "Em": (Ap*b*p + Ap*d*p)/(Ae*a*d), 
                     "Eu": Ap*p/(Ae*d), 
                     "P": (M*a**2*d*h*p*w + M*a*d*h*j*p*w + a**2*b*f*p*w**2 + a**2*d*f*p*w**2 + 2*a*b*f*j*p*w**2 + 2*a*d*f*j*p*w**2 + b*f*j**2*p*w**2 + d*f*j**2*p*w**2)/(M**2*a**2*d*h**2 + M*Me*a**2*d*h**2*w), 
                     "Pa": (M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*a*d*h*u), 
                     "Pe": (M*a**2*d*h*k2*p*w + M*a*d*h*j*k2*p*w + a**2*b*f*k2*p*w**2 + a**2*d*f*k2*p*w**2 + 2*a*b*f*j*k2*p*w**2 + 2*a*d*f*j*k2*p*w**2 + b*f*j**2*k2*p*w**2 + d*f*j**2*k2*p*w**2)/(M**2*a**2*d*h**2*k1 + M*Me*a**2*d*h**2*k1*w), 
                     "Pm": (M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*a**2*d*h), 
                     "Pu": (M*a*d*h*p + a*b*f*p*w + a*d*f*p*w + b*f*j*p*w + d*f*j*p*w)/(M*a*d*h*n)}
    
    # Set up equations for plotting
    PM_eq = sy.lambdify([M, Me], steady_states['P'] + steady_states['Pa'] + steady_states['Pm'] + steady_states['Pu'])
    Endosome_eq = sy.lambdify([M, Me], steady_states['E'] + steady_states['Ea'] + steady_states["Em"] + steady_states["Eu"])
    eisosome_only = sy.lambdify([M, Me], steady_states['Pe'])
    total = lambda M, Me: PM_eq(M, Me) + Endosome_eq(M, Me) + eisosome_only(M, Me)
    M_eq = lambda Me: bisection_method(Me, dM_eq, params=parameters)

    # plot equations
    plt.plot(Me_vals, [PM_eq(M_eq(m), m) for m in Me_vals], label="Plasma Membrane")
    plt.plot(Me_vals, [Endosome_eq(M_eq(m), m) for m in Me_vals], label="Endosome")
    plt.plot(Me_vals, [eisosome_only(M_eq(m), m) for m in Me_vals], label="Eisosome")
    plt.plot(Me_vals, [total(M_eq(m), m) for m in Me_vals], label="Total")
    plt.title("Mup1 Levels in a Yeast Cell")
    plt.xlabel("Extracellular Methionine")
    plt.ylabel("Mup1")
    plt.legend()
    plt.show()

mup1_methionine_plot()
