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

    


def substitution():
    """Substitute results from previous function"""
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
Me = .1

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

    print(sy_solve_LU())

    # Lets do some simplifying
    P = -(-Ae*f*(-j*(Ap*M**2*a*b*h**2*n*p/(Ae*w**2*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))**2*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)*(M*a*b*h/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)) - b - d)) 
                     - Ap*M*h*n*p/(Ae*w*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))))/(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j) 
                     - Ap*M*a*b*h*n*p/(Ae*w*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)
                      *(M*a*b*h/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)) - b - d)) 
                      + Ap*n*p/(Ae*(-b - Ap*n/Ae)))/(Ap*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - p 
                      + j*(-Ae*f*(-j*(Ap*M**2*a*b*h**2*n*p/(Ae*w**2*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))**2*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)*(M*a*b*h/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)) - b - d)) 
                                      - Ap*M*h*n*p/(Ae*w*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))))/(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j) - Ap*M*a*b*h*n*p/(Ae*w*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)*(M*a*b*h/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)) - b - d)) 
                                  + Ap*n*p/(Ae*(-b - Ap*n/Ae)))/(Ap*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - p)/a)/(h*(M/w + Me))
    Pm = -(-Ae*f*(-j*(Ap*M**2*a*b*h**2*n*p/(Ae*w**2*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))**2*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)*(M*a*b*h/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)) - b - d)) 
                      - Ap*M*h*n*p/(Ae*w*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))))/(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j) 
                      - Ap*M*a*b*h*n*p/(Ae*w*(-b - Ap*n/Ae)*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)*(M*a*b*h/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))*(-M*h*j/(w*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - a - j)) - b - d)) 
                      + Ap*n*p/(Ae*(-b - Ap*n/Ae)))/(Ap*(-Ae*f/Ap - M*h/w - f*n/(-b - Ap*n/Ae))) - p)/a

