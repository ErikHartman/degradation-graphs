import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from multiprocessing import Pool, cpu_count
import warnings
import networkx as nx
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.util import regex_to_graph, probabilities_to_flows
from src.weight_optimizer import WeightOptimizer

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

sns.set_context("paper")

HBA = "VLSAADKANVKAAWGKVGGQAGAHGAEALERMFLGFPTTKTYFPHFNLSHGSDQVKAHGQKVADALTKAVGHLDDLPGALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHHPDDFNPSVHASLDKFLANVSTVLTSKYR"

def main():
    # Load and process data
    porcine_data = pd.read_csv("/Users/erikhartman/dev/degradation-graphs/data/infection/data.csv")
    design = pd.read_csv("/Users/erikhartman/dev/degradation-graphs/data/infection/design.csv")

    # Filter for HBA protein (HBA) and get peptide data
    porcine_data = porcine_data[porcine_data["Protein"] == "HBA"]
    
    # Get sample columns (exclude metadata columns)
    intensity_cols = [col for col in porcine_data.columns if col.startswith("Sample")]
    peptide_cols = ["Peptide"] + intensity_cols
    enzyme_data = porcine_data[peptide_cols].copy()
    
    # Rename columns to match sample names in design
    enzyme_data = enzyme_data.rename(columns={col: col for col in enzyme_data.columns})
    
    # Get samples for each group
    s_aureus_samples = design[design["group"] == "S. aureus"]["sample"].tolist()
    p_aeruginosa_samples = design[design["group"] == "P. aeruginosa"]["sample"].tolist()
    
    # Set peptide sequence as index and apply log transformation
    enzyme_data = enzyme_data.set_index("Peptide")

    # Process all samples individually with parallelization
    sample_results = {'s_aureus': [], 'p_aeruginosa': []}
    sample_weights = {'s_aureus': [], 'p_aeruginosa': []}
    sample_flows = {'s_aureus': [], 'p_aeruginosa': []}
    underestimation_ratios = {'s_aureus': [], 'p_aeruginosa': []}

    # Get all unique peptides first to create consistent graph
    all_peptides = set()
    for sample in s_aureus_samples + p_aeruginosa_samples:
        if sample in enzyme_data.columns:
            temp_dict = (
                enzyme_data[sample]
                .replace(-np.inf, np.nan)
                .dropna()
                .to_dict()
            )
            for peptide in temp_dict.keys():
                if peptide in HBA:
                    all_peptides.add(peptide)

    all_peptides.add(HBA)

    # Create graph with all peptides
    G = regex_to_graph(HBA, list(all_peptides), "(.)(.)(.)(.)(.)(.)")

    # Prepare arguments for parallel processing
    s_aureus_args = []
    for sample in s_aureus_samples:
        if sample in enzyme_data.columns:
            s_aureus_args.append((sample, enzyme_data[sample], HBA, all_peptides, G))

    p_aeruginosa_args = []
    for sample in p_aeruginosa_samples:
        if sample in enzyme_data.columns:
            p_aeruginosa_args.append((sample, enzyme_data[sample], HBA, all_peptides, G))

    # Process samples in parallel
    print("CPU count:", cpu_count())
    n_cores = min(cpu_count(), 10)  # Use up to 10 cores

    with Pool(n_cores) as pool:
        s_aureus_results = pool.map(optimize_single_sample, s_aureus_args)

    with Pool(n_cores) as pool:
        p_aeruginosa_results = pool.map(optimize_single_sample, p_aeruginosa_args)

    # Collect results
    for result in s_aureus_results:
        if result is not None:
            sample_results['s_aureus'].append(result)
            sample_weights['s_aureus'].append(result['theta'])
            sample_flows['s_aureus'].append(result['flows'])
            underestimation_ratios['s_aureus'].append(result['underestimation_ratio'])

    for result in p_aeruginosa_results:
        if result is not None:
            sample_results['p_aeruginosa'].append(result)
            sample_weights['p_aeruginosa'].append(result['theta'])
            sample_flows['p_aeruginosa'].append(result['flows'])
            underestimation_ratios['p_aeruginosa'].append(result['underestimation_ratio'])

    # Get all unique edges (pathways) across all samples
    all_edges = set()
    for group_weights in sample_weights.values():
        for weights in group_weights:
            all_edges.update(weights.keys())

    # Plot loss histories
    fig, ax = plt.subplots(figsize=(2,2))

    for result in sample_results['s_aureus']:
        ax.plot(result['loss_history'], color='tab:red')
    for result in sample_results['p_aeruginosa']:
        ax.plot(result['loss_history'], color='tab:blue')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:red', label=f'S. aureus (n={len(sample_results["s_aureus"])})'),
        Line2D([0], [0], color='tab:blue',  label=f'P. aeruginosa (n={len(sample_results["p_aeruginosa"])})')
    ]
    ax.legend(handles=legend_elements, frameon=False)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine()
    plt.tight_layout()
    plt.savefig('./results/figures/porcine_combined_loss_histories.svg')
    plt.close()

    # Plot underestimation ratios
    fig, ax = plt.subplots(figsize=(3,3))

    s_aureus_ratios = underestimation_ratios['s_aureus']
    p_aeruginosa_ratios = underestimation_ratios['p_aeruginosa']

    box_data = pd.DataFrame({
        'Underestimation Ratio': s_aureus_ratios + p_aeruginosa_ratios,
        'Group': ['S. aureus'] * len(s_aureus_ratios) + ['P. aeruginosa'] * len(p_aeruginosa_ratios)
    })
    
    sns.boxplot(data=box_data, x='Group', y='Underestimation Ratio', 
                palette={'S. aureus': 'tab:red', 'P. aeruginosa': 'tab:blue'}, ax=ax)
    sns.despine()

    ax.set_ylabel('Underestimation ratio\n(generated / abundance)')
    ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig('./results/figures/porcine_underestimation_ratios.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate pathway statistics for peptidome plot using flows
    pathway_stats = {}
    for edge in all_edges:
        s_aureus_flows = [flows.get(edge, 0) for flows in sample_flows['s_aureus']]
        p_aeruginosa_flows = [flows.get(edge, 0) for flows in sample_flows['p_aeruginosa']]
        
        pathway_stats[edge] = {
            's_aureus_mean': np.mean(s_aureus_flows),
            's_aureus_std': np.std(s_aureus_flows),
            's_aureus_values': s_aureus_flows,
            'p_aeruginosa_mean': np.mean(p_aeruginosa_flows),
            'p_aeruginosa_std': np.std(p_aeruginosa_flows),
            'p_aeruginosa_values': p_aeruginosa_flows,
            'diff_mean': np.mean(s_aureus_flows) - np.mean(p_aeruginosa_flows)
        }

    # Calculate peptide fold changes (S. aureus vs P. aeruginosa)
    peptide_fold_changes = {}
    for peptide in all_peptides:
        if peptide == HBA:
            continue
            
        peptide_pathway_fcs = []
        for edge, stats in pathway_stats.items():
            if edge[1] == peptide:  # This pathway produces this peptide
                if stats['p_aeruginosa_mean'] > 0:
                    fold_change = stats['p_aeruginosa_mean'] / stats['s_aureus_mean']
                    log2_fc = np.log2(fold_change) if fold_change > 0 else 0
                    peptide_pathway_fcs.append(log2_fc)
        
        if peptide_pathway_fcs:
            peptide_fold_changes[peptide] = np.mean(peptide_pathway_fcs)
        else:
            peptide_fold_changes[peptide] = 0
    
    def plot_peptidome(V_Omega: str, P_M: dict, ax, max_range=None, cmap=None):
        """
        Plots peptides along the backbone of V_Omega colored by fold change.
        
        - Each peptide in P_M is drawn as a horizontal line at some 'height' 
          (so that lines do not overlap).
        - The color is determined by P_M[sequence] (fold change via cmap).
        """
        
        # Create custom Red-Blue colormap with white center
        if cmap is None:
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#DC143C", '#FF6B6B', "#ECDAEA", '#87CEEB', "#4169E1"]  # Red to Blue
            cmap = LinearSegmentedColormap.from_list('red_blue', colors, N=256)
        
        # Sort peptides (descending by length) so that longer peptides try to place first
        P_M = dict(sorted(P_M.items(), key=lambda item: len(item[0]), reverse=True))

        # Determine colormap range
        if max_range is None:
            max_range = max(abs(v) for v in P_M.values()) if P_M.values() else 1
        min_range = -max_range  # Center around 0

        # Initialize a 2D "space" array to avoid overlapping lines
        spaces = np.zeros((len(P_M.keys()), len(V_Omega)), dtype=float)

        # Loop over each peptide in P_M (except the parent itself)
        for sequence, fold_change in P_M.items():
            if sequence == V_Omega:
                continue
            
            # Locate the subsequence in V_Omega
            start = V_Omega.find(sequence)
            if start < 0:
                continue
            end = start + len(sequence)

            # Try to place this peptide on the first available "height" row
            for height in range(spaces.shape[0]):
                position_slice = spaces[height, start:end]
                if np.sum(position_slice) == 0:
                    spaces[height, start:end] = 1

                    # Color from colormap based on fold change
                    norm_val = (fold_change - min_range) / (max_range - min_range)
                    color = cmap(norm_val)
                    lw = 1

                    ax.plot(
                        [start+1, end-1],
                        [-height, -height],
                        linewidth=lw,
                        color=color,
                    )
                    break

        # Add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=min_range, vmax=max_range),
            cmap=cmap
        )
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, ax=ax, label=r"$log_2$ Fold change)")
        
        return ax
    
    # Create the peptidome plot
    fig, ax = plt.subplots(figsize=(4,3))
    plot_peptidome(
        V_Omega=HBA,
        P_M=peptide_fold_changes,
        ax=ax
    )
    
    ax.set_yticks([])
    ax.set_xlabel("Backbone index")
    sns.despine(ax=ax, left=True, bottom=True)
    
    plt.tight_layout()
    
    plt.savefig('./results/figures/porcine_peptidome_fold_change.svg', dpi=300, bbox_inches='tight')
    plt.close()

    peptide_descendants = {target: len(nx.descendants(G, target)) for target in peptide_fold_changes.keys() if target!=HBA}
    
    # Create the peptidome inflow plot
    fig, ax = plt.subplots(figsize=(4,3))
    plot_peptidome(
        V_Omega=HBA,
        P_M=peptide_descendants,
        ax=ax
    )
    
    ax.set_yticks([])
    ax.set_xlabel("Backbone index")
    sns.despine(ax=ax, left=True, bottom=True)
    
    plt.tight_layout()
    
    plt.savefig('./results/figures/infection_peptidome_inflow.svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Analysis completed:")
    print(f"S. aureus samples processed: {len(sample_results['s_aureus'])}")
    print(f"P. aeruginosa samples processed: {len(sample_results['p_aeruginosa'])}")
    print(f"Total peptides analyzed: {len(all_peptides)}")
    print(f"Total pathways identified: {len(all_edges)}")

   
def normalize_dict(P_M: dict):
    s = sum(P_M.values())
    return {k: v / s for k, v in P_M.items()}

def process_sample(sample_data, hba_sequence, all_peptides, quantile=0.2):
    """Process a single sample and return peptide abundances with quantile imputation"""
    temp_dict = (
        sample_data
        .replace(0, np.nan)
        .replace(-np.inf, np.nan)
        .dropna()
        .to_dict()
    )
    
    observed_values = list(temp_dict.values())
    if len(observed_values) == 0:
        return None
    
    impute_value = np.quantile(observed_values, quantile)

    filtered_dict = {}
    for peptide in all_peptides:
        if peptide == hba_sequence:
            filtered_dict[peptide] = 1.0 
        elif peptide in temp_dict and peptide in hba_sequence:
            filtered_dict[peptide] = temp_dict[peptide]
        else:
            # Impute missing peptides with quantile value
            filtered_dict[peptide] = impute_value
    
    return filtered_dict

def optimize_single_sample(args):
    """Wrapper function for parallel processing"""
    sample, sample_data, hba_sequence, all_peptides, G = args
    
    Y = process_sample(sample_data, hba_sequence, all_peptides)
    if Y is None:
        return None
    
    # Store the total abundance scale
    total_abundance_scale = sum(Y.values())
    
    # Weight optimization
    weight_optimizer = WeightOptimizer()
    try:
        final_theta = weight_optimizer.gradient_descent(G, Y, hba_sequence, lr=.25, epochs=20)
        flows_normalized = probabilities_to_flows(G, w_dict=final_theta, root=hba_sequence)
        
        # Scale flows to match original abundance scale
        flows = {k: v * total_abundance_scale for k, v in flows_normalized.items()}
        
        # Calculate underestimation ratio with scaled flows
        total_generated = sum(flows.values())
        total_abundance = sum(v for k, v in Y.items() if k != hba_sequence)
        underestimation_ratio = total_generated / total_abundance if total_abundance > 0 else 0
        
        return {
            'sample': sample,
            'Y': Y,
            'theta': final_theta,
            'flows': flows,  # Now scaled to match Y
            'loss_history': weight_optimizer.gd_loss_history,
            'underestimation_ratio': underestimation_ratio
        }
        
    except Exception as e:
        print(f"Error processing sample {sample}: {e}")
        return None


if __name__ == '__main__':
    main()