import numpy as np
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.integrate import solve_ivp

def run_single_sensitivity_analysis(param_values, problem, model_func, seed=None):
    """Run a single sensitivity analysis with given parameters.
    
    Parameters
    ----------
    problem : dict
        Problem definition for SALib
    param_values : numpy.ndarray
        Parameter values to evaluate
    model_func : callable
        Function that evaluates the model
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing sensitivity indices and metadata
    """
    start = perf_counter()

    # Run model evaluations
    print("Starting Multiprocessing")
    with Pool() as pool:
        Y = list(tqdm(pool.imap(model_func, param_values),
                        total=len(param_values),
                        desc="Running single iteration in parallel"))
    
    # Perform sensitivity analysis
    print("Begin Analysis")
    Si = morris_analyze.analyze(problem, Y, calc_second_order=True, seed=seed)
    
    # Store results with metadata
    results = {'N_value': len(Y) / 10,
        'time': perf_counter() - start,
        'seed': None,
        'sensitivity_indices': {
            'mu': Si['mu'].tolist(),
            'mu_star': Si['mu_star'].tolist(),
            'mu_star_conf': Si['mu_star_conf'].tolist(),
            'sigma': Si['sigma'].tolist() if 'sigma' in Si else None,
        },
        'parameter_names': problem['names']}
    
    return results

def aggregate_results(results):
    """Aggregate results from multiple sensitivity analyses.
    
    Parameters
    ----------
    results : list
        List of results from individual sensitivity analyses
        
    Returns
    -------
    dict
        Aggregated results with mean and standard deviation
    """
    # Extract parameter names from first result
    param_names = results[0]['parameter_names']
    
    # Initialize arrays for aggregation
    n_params = len(param_names)
    n_runs = len(results)
    
    mu_array = np.zeros((n_runs, n_params))
    mu_star_array = np.zeros((n_runs, n_params))
    sigma_array = np.zeros((n_runs, n_params))
    
    # Collect results
    for i, result in enumerate(results):
        mu_array[i] = result['sensitivity_indices']['mu']
        mu_star_array[i] = result['sensitivity_indices']['mu_star']
        sigma_array[i] = result['sensitivity_indices']['sigma']
    
    # Calculate statistics
    aggregated = {
        'parameter_names': param_names,
        'mu': {
            'mean': np.mean(mu_array, axis=0).tolist(),
            'std': np.std(mu_array, axis=0).tolist(),
            'min': np.min(mu_array, axis=0).tolist(),
            'max': np.max(mu_array, axis=0).tolist()
        },
        'mu_star': {
            'mean': np.mean(mu_star_array, axis=0).tolist(),
            'std': np.std(mu_star_array, axis=0).tolist(),
            'min': np.min(mu_star_array, axis=0).tolist(),
            'max': np.max(mu_star_array, axis=0).tolist()
        },
        'sigma': {
            'mean': np.mean(sigma_array, axis=0).tolist(),
            'std': np.std(sigma_array, axis=0).tolist(),
            'min': np.min(sigma_array, axis=0).tolist(),
            'max': np.max(sigma_array, axis=0).tolist()
        },
        'n_runs': n_runs,
        'timestamp': datetime.now().isoformat()
    }
    
    return aggregated

### This function would need updates to run (switching key names)
# def save_results(results, aggregated, output_dir='sensitivity_results'):
#     """Save results to files.
    
#     Parameters
#     ----------
#     results : list
#         Individual sensitivity analysis results
#     aggregated : dict
#         Aggregated results
#     output_dir : str
#         Directory to save results
#     """
#     import os
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save individual results
#     for i, result in enumerate(results):
#         with open(f'{output_dir}/run_{i}.json', 'w') as f:
#             json.dump(result, f, indent=2)
    
#     # Save aggregated results
#     with open(f'{output_dir}/aggregated_results.json', 'w') as f:
#         json.dump(aggregated, f, indent=2)
    
#     # Create summary DataFrame
#     summary_df = pd.DataFrame({
#         'Parameter': aggregated['parameter_names'],
#         'S1_mean': aggregated['S1']['mean'],
#         'S1_std': aggregated['S1']['std'],
#         'ST_mean': aggregated['ST']['mean'],
#         'ST_std': aggregated['ST']['std']
#     })
    
#     # Save summary to CSV
#     summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

def plot_results(aggregated, barsize=0.15, include_sigma=True, include_mu=False, output_dir='Images/sensitivity_analysis', label=''):
    """Create plots of the aggregated results.
    
    Parameters
    ----------
    aggregated : dict
        Aggregated results
    include_sigma : bool
        Choose whether to include the results for sigma in the plot
    output_dir : str, optional
        Directory to save plots
    label : str, optional
        Additional label to distinguish different results
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 12))
    
    params = aggregated['parameter_names']
    x = np.arange(len(params))
    width = barsize
    
    if include_mu:
        # Plot mu indices
        ax1.bar(x - .15, aggregated['mu']['mean'], width, yerr=aggregated['mu']['std'],
                label='mu', capsize=3)
    
    # Plot mu_star indices
    ax1.bar(x, aggregated['mu_star']['mean'], width, yerr=aggregated['mu_star']['std'],
            label='mu_star', capsize=3)
    
    if include_sigma:
        # Plot sigma indices
        ax1.bar(x + 0.15, aggregated['sigma']['mean'], width, yerr=aggregated['sigma']['std'],
                label='sigma', capsize=3)
    
    # Set figure attributes
    ax1.set_ylabel('Sensitivity Index')
    ax1.set_title('Mu_star Sensitivity Indices')
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=45, ha='right')
    ax1.legend()
    
    plt.tight_layout()
    # plt.savefig(f'{output_dir}/sensitivity_plots{label}.png')
    plt.show()

def scatter_plot(results, aggregated=None, output_dir=None, title=None, show=True):
    """Plot a scatter plot of the mu_star and sigma results from Morris sensitivity analysis (used morris to sample and analyze).
    
    Parameters
    ----------
    results: dict
        dictionary of outputs from sensitivity analysis
    aggregated: dict, optional
        dictionary of aggregated results (from aggregate_results function)
    output_dir: string, optional
        location to save output"""
    
    # get the sigma and mu vals for each parameter
    if aggregated:
        mu_vals = aggregated["mu_star"]["mean"]
        sigma_vals = aggregated["sigma"]["mean"]
        labels = aggregated['parameter_names']
    elif results:
        mu_vals = results["sensitivity_indices"]["mu_star"]
        sigma_vals = results["sensitivity_indices"]["sigma"]
        labels = results['parameter_names']
    else:
        raise ValueError("Must provide results or aggregate dictionary.")

    # create the scatter plot
    plt.scatter(mu_vals, sigma_vals)
    for i, label in enumerate(labels):   # label each point
        plt.annotate(label, (mu_vals[i], sigma_vals[i]), (5, 5), textcoords='offset points',
                     ha='left', va='bottom')
    if title:
        plt.title(title)
    else:
        plt.title("Morris Sensitivity Analysis")
    plt.xlabel("$\mu$")
    plt.ylabel("$\sigma$")
    plt.tight_layout()

    # save and show
    if output_dir:
        plt.savefig(output_dir)
    if show:
        plt.show()

def mup1_model(t, system, parameters):
    """Function coding the Mup1 trafficking model.
    
    Parameters
    ----------
    system : array, list 
        values of P, Pb, Pa, Pu, E, Em, Ea, Eu, M
    parameters: array, list
        values of parameters to run the system with
    """

    P, Pb, Pa, Pu, E, Em, Ea, Eu, M = system 
    y, w, j, kd, f, Ae, Ap, a, h, b, z, g, V, vmax, Km, Me = parameters    # unpack for readability
    k = j / kd

    # define the differential equations
    dy = [
        y - k*Me*P - (k/w)*M*P + j*Pb + f*(Ae/Ap)*E,                       # P
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

### Fur4 Parameters ###
a = 1         # *ubiquitination rate (ms^{-1})
Ae = 47       # endosomal membrane surface area (micrometers^3)
Ap = 314      # plasma membrane surface area (micrometers^3)
b = 1         # *deubiquitination rate (ms^{-1})
f = 0.25      # *fur4 recycling rate (ms^{-1})
g = 0.1       # *fur4 endocytosis rate (ms^{-1})
j = 10**2     # *uracil unbinding rate (ms^{-1})
Kd = 0.74     # uracil dissociation constant (micromolars)
k = j / Kd    # *uracil binding rate (micromolars^{-1}ms^{-1})
Km = 2.5      # uracil Michaelis-Menten constant (micromolars)
V = 523       # volume of cytoplasm (micrometers^3)
vmax = 88 / 6 # maximal rate of uracil metabolism (micrommolars*micrometers^3ms^{-1})
w = 32        # scale factor for pH
y = 5 / 6000  # *fur4 production rate (# Fur4)*ms^{-1}
z = 0.002     # *degradation rate (ms^{-1})

def fur4_model(t, y, parameters):

    # unpack parameters
    P, Pb, Pu, E, Eb, Eu, S = y
    a, b, f, g, j, k, y, z, Se = parameters
    # k = j / Kd
    
    # Define derivatives
    dP = y - (k * Se * P) - ((k / w) * S * P) + (j * Pb) + (f * (Ae / Ap) * E)
    dPb = (k * Se * P) + ((k / w) * S * P) - (j * Pb) - (a * Pb)
    dPu = (a * Pb) - (g * Pu)
    dE = (b * Eu) - ((k / w) * S * E) + (j * Eb) - (f * E)
    dEb = ((k / w) * S * E) - (j * Eb) - (a * Eb)
    dEu = (g * (Ap / Ae) * Pu) - (b * Eu) + (a * Eb) - (z * Eu)
    dS = - ((k / w) * S * ((Ap / V) * P + (Ae / V) * E)) + ((j + a) *
        ((Ap / V) * Pb + (Ae / V) * Eb)) - ((vmax * S) / (V * (Km + S)))
    
    return [dP, dPb, dPu, dE, dEb, dEu, dS]

# helper function to help with parallelization
def run_model_fur4(X):
    y0 = [0, 0, 0, 0, 0, 0, 0]
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 10)

    sol = solve_ivp(fur4_model, t_span, y0, args=(X,), t_eval=t_eval)
    return sum(sol.y[:, -1])  # sum excluding extracellular uracil

# Example usage:
if __name__ == "__main__":
    ### FOR FUR4 ###

    parameter_ranges = [   # (z, k, S_e, b, y, a, f, j, g)
        [0.0002, 0.02],        # z: degradation
        [10/.74, 1000/.74],    # k: uracil binding rate (depends on j and kd)
        [0.0001, 1],           # Se: extracellular uracil
        [0.1, 10],             # b: deubiquitination rate
        [5/60000, 5/600],      # y: production
        [0.1, 10],             # a: ubiquitination
        [0.025, 2.5],          # f: recycling rate
        [10, 1000],            # j: unbinding
        [0.01, 1]              # g: endocytosis rate
    ]

    # set up the problem
    problem = {
        'num_vars': 9,
        'names': ['z', 'k', 'S_e', 'b', 'y', 'a', 'f', 'j', 'g'],
        'bounds': parameter_ranges
    }

    # generate samples and run sensitivity analysis
    for i in range(5):
        param_values = morris_sample.sample(problem, 32768)
        result = run_single_sensitivity_analysis(param_values, problem, run_model_fur4)
        print(result)
        with open("Results/morris_sensitivity.txt", 'a') as f:
            f.write(f"{result}\n")
        print(f"Saving result to Results/morris_sensitivity")

    print("Finished running 5 sensitivity analyses. Results saved to morris_sensitivity.txt.")
    