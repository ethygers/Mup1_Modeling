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
p = '' # units mup1 per millisecond (mup1 production rate)
h = '' # per micromolar per millisecond (methionine binding rate)
w = 32 # unitless (scale factor for pH difference)
j = '' # per millisecond (methionine unbinding rate)
f = '' # per millisecond (recycling rate)
Ae = 47 # micrometers^3 (endosomal membrane surface area)
Ap = 314 # micrometers^3 (plasma membrane surface area)
u = 1 # per millisecond (ubiquitination rate)
a = '' # per millisecond (art 1 binding rate)
b = '' # per millisecond (deubiquitination rate)
d = '' # per millisecond (degradation rate)
n = '' # per millisecond (endocytosis rate)
V = 523 # micrometers^3 (volume of cytoplasm)
vmax = '' # 
Km = ''