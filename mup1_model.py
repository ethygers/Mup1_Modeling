import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

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
        -f*(Ae/Ap)*E + b*Eu - (h/w)*E*M + j*Em + b*Pu,                    # E
        (h/w)*E*M - a*Em - j*Em,                                          # Em
        a*Em - u*Ea,                                                      # Ea
        n*(Ap/Ae)*Pu - b*Eu + u*Ea - d*Eu,                                # Eu
        -(h/w)*M*P - (h/w)*E*M + (j + u)*(Em + Pm) - vmax*M/(V*(Km + M))  # M
    ]

    return dy

# parameters (filled in the ones I think would be the same or similar to Fur4)
p = .05 #'' # units mup1 per millisecond (mup1 production rate)
h = 1e-5 #'' # per micromolar per millisecond (methionine binding rate)
w = 32 # unitless (scale factor for pH difference)
j = 1e-5 #'' # per millisecond (methionine unbinding rate)
f = .25 #'' # per millisecond (recycling rate)
Ae = 47 # micrometers^3 (endosomal membrane surface area)
Ap = 314 # micrometers^3 (plasma membrane surface area)
u = 1 # per millisecond (ubiquitination rate)
a = 1e-5 #'' # per micromolar per millisecond (art 1 binding rate)
b = 1 # per millisecond (deubiquitination rate)
d = .002 #'' # per millisecond (degradation rate)
n = 0.1 #'' # per millisecond (endocytosis rate)
V = 523 # micrometers^3 (volume of cytoplasm)
vmax = 8.8 #'' # micromolars*micrometers^3 per millisecond (maximal rate of methionine metabolism)
Km = 2.5 #'' # micromolars (methionin michaelis-menten constant)

# methionine (changes)
Me = .1

if __name__ == '__main__':
    # establish initial conditions
    initial = [.1, .1, .1, .1, .1, .1, .1, .1, .1]
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
