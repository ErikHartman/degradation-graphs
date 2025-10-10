import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import networkx as nx
from multiprocessing import Pool, cpu_count
import warnings
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

# UMOD protein sequence
umod = "MGQPSLTWMLMVVVASWFITTAATDTSEARWCSECHSNATCTEDEAVTTCTCQEGFTGDGLTCVDLDECAIPGAHNCSANSSCVNTPGSFSCVCPEGFRLSPGLGCTDVDECAEPGLSHCHALATCVNVVGSYLCVCPAGYRGDGWHCECSPGSCGPGLDCVPEGDALVCADPCQAHRTLDEYWRSTEYGEGYACDTDLRGWYRFVGQGGARMAETCVPVLRCNTAAPMWLNGTHPSSDEGIVSRKACAHWSGHCCLWDASVQVKACAGGYYVYNLTAPPECHLAYCTDPSSVEGTCEECSIDEDCKSNNGRWHCQCKQDFNITDISLLEHRLECGANDMKVSLGKCQLKSLGFDKVFMYLSDSRCSGFNDRDNRDWVSVVTPARDGPCGTVLTRNETHATYSNTLYLADEIIIRDLNIKINFACSYPLDMKVSLKTALQPMVSALNIRVGGTGMFTVRMALFQTPSYTQPYQGSSVTLSTEAFLYVGTMLDGGDLSRFALLMTNCYATPSSNATDPLKYFIIQDRCPHTRDSTIQVVENGESSQGRFSVQMFRFAGNYDLVYLHCEVYLCDTMNEKCKPTCSGTRFRSGSVIDQSRVLNLGPITRKGVQATVSRAFSSLGLLKVWLPLLLSATLTLTFQ"

def main():
    # Load and process data
    enzyme_data = pd.read_csv("/Users/erikhartman/dev/degradation-graphs/data/diabetes/peptides.txt", sep="\t")
    design = pd.read_csv("/Users/erikhartman/dev/degradation-graphs/data/diabetes/design.csv")
    enzyme_data = enzyme_data[enzyme_data["Gene names"] == "UMOD"]
    enzyme_data = enzyme_data[["Sequence"] + [col for col in enzyme_data.columns if col.startswith("Intensity ")]]
    enzyme_data = enzyme_data.rename(columns={col: col.split("_")[-1] for col in enzyme_data.columns})

    diabetes_samples = design[design["group"] == "Type 1 diabetes"]["sample"].tolist()
    healthy_samples = design[design["group"] == "Healthy control"]["sample"].tolist()
    enzyme_data = enzyme_data.set_index("Sequence").apply(np.log)


    # Process all samples individually with parallelization
    sample_results = {'diabetes': [], 'healthy': []}
    sample_weights = {'diabetes': [], 'healthy': []}
    sample_flows = {'diabetes': [], 'healthy': []}
    underestimation_ratios = {'diabetes': [], 'healthy': []}


    # Get all unique peptides first to create consistent graph
    all_peptides = set()
    for sample in diabetes_samples + healthy_samples:
        if sample in enzyme_data.columns:
            temp_dict = (
                enzyme_data[sample]
                .replace(0, np.nan)
                .replace(-np.inf, np.nan)
                .dropna()
                .to_dict()
            )
            for peptide in temp_dict.keys():
                if peptide in umod:
                    all_peptides.add(peptide)

    all_peptides.add(umod)


    # Create graph with all peptides
    G = regex_to_graph(umod, list(all_peptides), "(.)(.)(.)(.)(.)(.)")

    # Prepare arguments for parallel processing
    diabetes_args = []
    for sample in diabetes_samples:
        if sample in enzyme_data.columns:
            diabetes_args.append((sample, enzyme_data[sample], umod, all_peptides, G))

    healthy_args = []
    for sample in healthy_samples:
        if sample in enzyme_data.columns:
            healthy_args.append((sample, enzyme_data[sample], umod, all_peptides, G))

    # Process samples in parallel
    n_cores = min(cpu_count(), 8)  # Use up to 8 cores



    with Pool(n_cores) as pool:
        diabetes_results = pool.map(optimize_single_sample, diabetes_args)


    with Pool(n_cores) as pool:
        healthy_results = pool.map(optimize_single_sample, healthy_args)

    # Collect results
    for result in diabetes_results:
        if result is not None:
            sample_results['diabetes'].append(result)
            sample_weights['diabetes'].append(result['theta'])
            sample_flows['diabetes'].append(result['flows'])
            underestimation_ratios['diabetes'].append(result['underestimation_ratio'])

    for result in healthy_results:
        if result is not None:
            sample_results['healthy'].append(result)
            sample_weights['healthy'].append(result['theta'])
            sample_flows['healthy'].append(result['flows'])
            underestimation_ratios['healthy'].append(result['underestimation_ratio'])



    # Get all unique edges (pathways) across all samples
    all_edges = set()
    for group_weights in sample_weights.values():
        for weights in group_weights:
            all_edges.update(weights.keys())

    fig, ax = plt.subplots(figsize=(2,2))

    for result in sample_results['diabetes']:
        ax.plot(result['loss_history'], color='tab:orange')
    for result in sample_results['healthy']:
        ax.plot(result['loss_history'], color='tab:blue')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:orange', label=f'Diabetes (n={len(sample_results["diabetes"])})'),
        Line2D([0], [0], color='tab:blue',  label=f'Healthy (n={len(sample_results["healthy"])})')
    ]
    ax.legend(handles=legend_elements, frameon=False)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine()
    plt.tight_layout()
    plt.savefig('./results/figures/diabetes_combined_loss_histories.svg')
    
    plt.close()

    fig, ax = plt.subplots(figsize=(2.5, 2))

    diabetes_ratios = underestimation_ratios['diabetes']
    healthy_ratios = underestimation_ratios['healthy']

    box_data = pd.DataFrame({
        'Underestimation Ratio': diabetes_ratios + healthy_ratios,
        'Group': ['Diabetes'] * len(diabetes_ratios) + ['Healthy'] * len(healthy_ratios)
    })
    sns.boxplot(data=box_data, x='Group', y='Underestimation Ratio', palette={'Diabetes': 'tab:orange', 'Healthy': 'tab:blue'}, ax=ax)
    sns.despine()

    ax.set_ylabel('Underestimation ratio\n(generated / abundance)')
    ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig('./results/figures/diabetes_underestimation_ratios.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate pathway statistics for peptidome plot using flows
    pathway_stats = {}
    for edge in all_edges:
        diabetes_flows = [flows.get(edge, 0) for flows in sample_flows['diabetes']]
        healthy_flows = [flows.get(edge, 0) for flows in sample_flows['healthy']]
        
        pathway_stats[edge] = {
            'diabetes_mean': np.mean(diabetes_flows),
            'diabetes_std': np.std(diabetes_flows),
            'diabetes_values': diabetes_flows,
            'healthy_mean': np.mean(healthy_flows),
            'healthy_std': np.std(healthy_flows),
            'healthy_values': healthy_flows,
            'diff_mean': np.mean(diabetes_flows) - np.mean(healthy_flows)
        }


    peptide_fold_changes = {}
    for peptide in all_peptides:
        if peptide == umod:
            continue
            
        peptide_pathway_fcs = []
        for edge, stats in pathway_stats.items():
            if edge[1] == peptide:  # This pathway produces this peptide
                if stats['healthy_mean'] > 0:
                    fold_change = stats['diabetes_mean'] / stats['healthy_mean']
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
        
        # Create custom Orange-Blue colormap with white center
        if cmap is None:
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#EE8F19", '#FFA500', "#B2D5E0", '#87CEEB', "#5252F5"]
            cmap = LinearSegmentedColormap.from_list('orange_blue', colors, N=256)
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
        cb = plt.colorbar(sm, cax=cax, ax=ax, label=r"$log_2$ Fold change")
        
        return ax
    
    # Create the peptidome plot
    fig, ax = plt.subplots(figsize=(4,2))
    plot_peptidome(
        V_Omega=umod,
        P_M=peptide_fold_changes,
        ax=ax
    )
    
    ax.set_yticks([])
    ax.set_xlabel("Backbone index")
    sns.despine(ax=ax, left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig('./results/figures/diabetes_peptidome_fold_change.svg', dpi=300, bbox_inches='tight')
    plt.close()

    peptide_descendants = {target: len(nx.descendants(G, target)) for target in peptide_fold_changes.keys() if target!=umod}
    fig, ax = plt.subplots(figsize=(4,2))
    plot_peptidome(
        V_Omega=umod,
        P_M=peptide_descendants,
        ax=ax
    )
    
    ax.set_yticks([])
    ax.set_xlabel("Backbone index")
    sns.despine(ax=ax, left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig('./results/figures/diabetes_peptidome_descendants.svg', dpi=300, bbox_inches='tight')
    plt.close()

   
def normalize_dict(P_M: dict):
    s = sum(P_M.values())
    return {k: v / s for k, v in P_M.items()}

def process_sample(sample_data, umod_sequence, all_peptides, quantile=0.2):
    """Process a single sample and return normalized peptide abundances with quantile imputation"""
    temp_dict = (
        sample_data
        .replace(0, np.nan)
        .replace(-np.inf, np.nan)
        .dropna()
        .to_dict()
    )
    
    observed_values = list(temp_dict.values())
    impute_value = np.quantile(observed_values, quantile)

    filtered_dict = {}
    for peptide in all_peptides:
        if peptide == umod_sequence:
            filtered_dict[peptide] = 1.0 
        elif peptide in temp_dict and peptide in umod_sequence:
            filtered_dict[peptide] = temp_dict[peptide]
        else:
            # Impute missing peptides with quantile value
            filtered_dict[peptide] = impute_value
    
    return filtered_dict

def optimize_single_sample(args):
    """Wrapper function for parallel processing"""
    sample, sample_data, umod_sequence, all_peptides, G = args
    
    Y = process_sample(sample_data, umod_sequence, all_peptides)
    if Y is None:
        return None
    
    # Store the total abundance scale
    total_abundance_scale = sum(Y.values())
    
    # Weight optimization
    weight_optimizer = WeightOptimizer()
    try:
        final_theta = weight_optimizer.gradient_descent(G, Y, umod_sequence, lr=.01, epochs=250)
        flows_normalized = probabilities_to_flows(G, w_dict=final_theta, root=umod_sequence)
        
        # Scale flows to match original abundance scale
        flows = {k: v * total_abundance_scale for k, v in flows_normalized.items()}
        
        # Calculate underestimation ratio with scaled flows
        total_generated = sum(flows.values())
        total_abundance = sum(v for k, v in Y.items() if k != umod_sequence)
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