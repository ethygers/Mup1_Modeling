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
    # Run model evaluations
    with Pool(3) as pool:
        Y = list(tqdm(pool.map(model_func, param_values),
                        total=len(param_values),
                        desc="Running single iteration in parallel"))
    
    # Perform sensitivity analysis
    print("Begin Analysis")
    Si = morris_analyze.analyze(problem, Y, calc_second_order=True, seed=seed)
    
    # Store results with metadata
    results = {'timestamp': datetime.now().isoformat(),
        'seed': None,
        'sensitivity_indices': {
            'mu': Si['mu'].tolist(),
            'mu_star': Si['mu_star'].tolist(),
            'mu_star_conf': Si['mu_star_conf'].tolist(),
            'sigma': Si['sigma'].tolist() if 'sigma' in Si else None,
        },
        'parameter_names': problem['names']}
    
    return results

def run_parallel_sensitivity_analyses(problem, model_func, n_runs=10, n_samples=32, n_processes=2):
    """Run multiple sensitivity analyses in parallel.
    
    Parameters
    ----------
    problem : dict
        Problem definition for SALib
    model_func : callable
        Function that evaluates the model
    n_runs : int
        Number of sensitivity analyses to run
    n_samples : int
        Number of samples per analysis
    n_processes : int, optional
        Number of processes to use (defaults to CPU count)
        
    Returns
    -------
    list
        List of results from each sensitivity analysis
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    # Generate seeds for reproducibility
    seeds = np.random.randint(0, 2**32, size=n_runs)
    
    # Create partial function with fixed arguments
    run_analysis = partial(
        run_single_sensitivity_analysis,
        problem=problem,
        model_func=model_func
    )
    
    # Prepare arguments for parallel processing
    args = []
    for seed in seeds:
        # Generate parameter samples for this run
        param_values = morris_sample.sample(problem, n_samples)
        args.append(param_values)
    
    # Run analyses in parallel
    with ThreadPool(n_processes) as pool:
        print("Begin Multiprocessing")
        results = list(tqdm(
            pool.map(run_analysis, args),
            total=n_runs,
            desc="Running sensitivity analyses"
        ))
    
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

def scatter_plot(results, aggregated=None, output_dir=None):
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
    plt.title("Morris Sensitivity Analysis")
    plt.xlabel("$\mu$")
    plt.ylabel("$\sigma$")
    plt.tight_layout()

    # save and show
    if output_dir:
        plt.savefig(output_dir)
    plt.show()

def mup1_model(t, y, parameters):
    """Function coding the Mup1 trafficking model.
    
    Parameters
    ----------
    y : array, list 
        values of P, Pm, Pa, Pu, E, Em, Ea, Eu, M
    parameters: array, list
        values of parameters to run the system with
    """

    P, Pm, Pa, Pu, E, Em, Ea, Eu, M = y 
    p, w, j, kd, f, Ae, Ap, u, a, b, d, n, V, vmax, Km, Me = parameters    # unpack for readability
    h = j / kd

    # define the differential equations
    dy = [
        p - h*Me*P - (h/w)*M*P + j*Pm + f*(Ae/Ap)*E,                       # P
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

# Example usage:
if __name__ == "__main__":
    # Define your problem
    problem = {
        'num_vars': 16,
        'names': ['p', 'w', 'j', 'kd', 'f', 'Ae', 'Ap', 'u', 'a', 'b', 'd', 'n', 'V', 'vmax', 'Km', 'Me'],
        'bounds': [[8.3e-6, 8.3e-4],      # p: Mup1 production rate – production rates can vary over orders of magnitude due to transcriptional and translational regulation, environmental cues, and promoter strength.
                    [10, 100],            # w: pH scale factor – this is a unitless scaling factor, and while its exact biological interpretation may vary, a range of 1 order of magnitude allows for exploratory analysis without being too speculative.
                    [10, 1000],           # j: Methionine unbinding rate – binding and unbinding kinetics often vary by 1–2 orders of magnitude depending on temperature, affinity, and conformational state.
                    [1000, 90000],        # kd: Dissociation constant – dissociation constants vary widely across protein-ligand systems; range chosen to reflect affinities from high (1 μM) to low (90 μM) binding strength.
                    # h is derived from j/kd, so it should not be varied independently – vary j and kd instead
                    [0.025, 2.5],         # f: Recycling rate – endosomal recycling rates can vary depending on the type of cargo, regulatory proteins, and metabolic state; 1 order of magnitude captures plausible biological fluctuation.
                    [20, 100],            # Ae: Endosomal surface area – endosomal sizes (and hence surface areas) differ based on maturation stage and cell size; this range allows for ~5x variation while remaining realistic for yeast cells.
                    [100, 1000],          # Ap: Plasma membrane surface area – reflects variability in yeast cell size; 3–10 μm diameter cells yield surface areas within this range.
                    [0.1, 10],            # u: Ubiquitination rate – enzymatic tagging rates are context-dependent, influenced by E3 ligase concentration and substrate type; range spans 2 orders of magnitude.
                    [1, 100],             # a: Art1 binding rate – ART protein interactions with transporters can vary widely depending on substrate conformation and signaling state; large range allows for nonlinearity exploration.
                    [0.1, 10],            # b: Deubiquitination rate – affected by availability of deubiquitinases and substrate accessibility; same logic as for ubiquitination rate.
                    [0.0002, 0.02],       # d: Degradation rate – protein degradation is generally slow, but this range captures variation due to stress conditions, proteasome targeting, or trafficking dynamics.
                    [0.01, 1],            # n: Endocytosis rate – strongly regulated and responsive to signaling, nutrient levels, and surface cargo density; up to 100-fold variability is plausible.
                    [200, 1000],          # V: Cytoplasmic volume – yeast cells range from ~30 to ~100 fL; cytoplasmic volume varies with cell cycle stage and environmental conditions.
                    [1e4, 1e6],           # vmax: Max methionine metabolism rate – reflects possible differences in metabolic enzyme expression, post-translational regulation, and methionine flux capacity.
                    [50, 1000],           # Km: Michaelis constant – Km values commonly vary across enzymes and contexts; this range includes both high-affinity (low Km) and low-affinity (high Km) scenarios.
                    [0.1, 500],           # Me: Extracellular methionine concentration – based on reported yeast media compositions (from starvation up to rich media); range spans near-zero to saturating conditions.
                ]
    }
    
    from scipy.integrate import solve_ivp

    # Define your model function
    def model_func(X):
        y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        t_span = (0, 50)
        t_eval = np.linspace(*t_span, 10)
        print('...')
    
        sol = solve_ivp(mup1_model, t_span, y0, args=(X,), t_eval=t_eval)
        print("..")
        return sum(sol.y[:, -1])
    
    # Run parallel sensitivity analyses
    results = run_parallel_sensitivity_analyses(
        problem=problem,
        model_func=model_func,
        n_runs=10,
        n_samples=1024
    )
    
    # Aggregate results
    aggregated = aggregate_results(results)
    
    # # Create plots
    # plot_results(aggregated) 


    # take out parameters we know for sure and then compare it to Fur4 plot