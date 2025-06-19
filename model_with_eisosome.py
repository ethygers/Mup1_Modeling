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


def bisection_method(Me, dM, params = [8.3e-5, 135, 32, 100, .25, 47, 314, 1, 1e-5, 1, .002, 0.1, 523, 8.8e3, 2.5, 1, 1],
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

    # unpack params for readability
    p, h, w, j, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, k1, k2 = params

    # save dM with substitutions for all other variables (computed in steady_states_simplification.ipynb)
    dM = sy.simplify(dM(Me))
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


if __name__ == "__main__":
    # get dM equation
    steady_states, dM_symbolic = compute_steady_states()

    # lambdify dM equation
    Me = sy.symbols("Me")
    dM = sy.lambdify(Me, dM_symbolic, 'numpy')

    # use bisection method to graph Me
    met_vals = np.linspace(.1, 25, 50)
    M_vals = np.array([bisection_method(m, dM) for m in met_vals])

    plt.plot(met_vals, M_vals)
    plt.title("Eisosome Methionine\nextracellular vs intracellular")
    plt.xlabel("Me")
    plt.ylabel("M")
    plt.show()


    #### DEBUG BISECTION METHOD ####