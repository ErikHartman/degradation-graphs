import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
import torch

# Import custom modules
from src.util import regex_to_graph
from src.proteolysis_simulator import ProteolysisSimulator, Enzyme

warnings.filterwarnings('ignore')
sns.set_context("paper")

# UMOD protein sequence (same as in diabetes analysis)
UMOD = "MGQPSLTWMLMVVVASWFITTAATDTSEARWCSECHSNATCTEDEAVTTCTCQEGFTGDGLTCVDLDECAIPGAHNCSANSSCVNTPGSFSCVCPEGFRLSPGLGCTDVDECAEPGLSHCHALATCVNVVGSYLCVCPAGYRGDGWHCECSPGSCGPGLDCVPEGDALVCADPCQAHRTLDEYWRSTEYGEGYACDTDLRGWYRFVGQGGARMAETCVPVLRCNTAAPMWLNGTHPSSDEGIVSRKACAHWSGHCCLWDASVQVKACAGGYYVYNLTAPPECHLAYCTDPSSVEGTCEECSIDEDCKSNNGRWHCQCKQDFNITDISLLEHRLECGANDMKVSLGKCQLKSLGFDKVFMYLSDSRCSGFNDRDNRDWVSVVTPARDGPCGTVLTRNETHATYSNTLYLADEIIIRDLNIKINFACSYPLDMKVSLKTALQPMVSALNIRVGGTGMFTVRMALFQTPSYTQPYQGSSVTLSTEAFLYVGTMLDGGDLSRFALLMTNCYATPSSNATDPLKYFIIQDRCPHTRDSTIQVVENGESSQGRFSVQMFRFAGNYDLVYLHCEVYLCDTMNEKCKPTCSGTRFRSGSVIDQSRVLNLGPITRKGVQATVSRAFSSLGLLKVWLPLLLSATLTLTFQ"


def create_synthetic_peptidome(protein_sequence, n_peptides, pattern=r"(.)(.)(.)(.)(.)(.)", seed=None):
    """
    Create a synthetic peptidome using the proteolysis simulator.
    
    This generates a more realistic peptidome with proper enzymatic cleavage patterns
    and abundance distributions.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Define a trypsin-like enzyme (cleaves after K or R)
    trypsin = Enzyme([("(.)(.)([RK])([^P])(.)(.)", 1)])
    
    # Create simulator
    simulator = ProteolysisSimulator(
        min_length=6,
        length_params="vitro",
        random_seed=seed,
        verbose=False
    )
    
    # Simulate proteolysis to generate target number of peptides
    # Scale n_succesful_cleaves to get approximately n_peptides unique peptides
    target_cleaves = int(n_peptides * 1.5)  # Generate more events to get diverse peptides
    
    P_Y, _ = simulator.simulate_proteolysis(
        sequence=protein_sequence,
        enzyme=trypsin,
        n_succesful_cleaves=target_cleaves,
        relative_protein_abundance=0.1,
        endo_probability=0.9,
        make_graph=False
    )
    
    # Extract peptides and abundances
    peptides = list(P_Y.keys())
    
    # Create graph
    G = regex_to_graph(protein_sequence, peptides, pattern)
    
    # Use simulated abundances (normalized)
    Y = {p: P_Y.get(p, 0) for p in peptides}
    total = sum(Y.values())
    Y = {k: v / total for k, v in Y.items()}
    
    return G, Y, peptides


def run_gradient_descent_with_init(G, Y, root, lr=0.001, epochs=250, seed=None, init_scale=0.1):
    """
    Run gradient descent with random initialization of parameters.
    
    This wraps the standard gradient descent but allows for random initialization
    of the logits, ensuring different starting points for each run.
    """
    import torch
    import torch.nn.functional as F
    import networkx as nx
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Build topological order
    all_nodes = list(nx.topological_sort(G))
    node_to_children = {n: list(G.successors(n)) for n in all_nodes}
    
    # Create parameters with RANDOM initialization (not zeros!)
    theta_dict = {}
    for n in all_nodes:
        out_deg = len(node_to_children[n])
        # Random initialization instead of zeros
        param = torch.nn.Parameter(
            torch.randn(out_deg + 1, dtype=torch.float) * init_scale
        )
        theta_dict[n] = param
    
    # Create optimizer
    optimizer = torch.optim.Adam(theta_dict.values(), lr=lr)
    
    # Forward pass helper
    def forward_pass():
        p_in = {n: torch.tensor(0.0, dtype=torch.float) for n in all_nodes}
        p_in[root] = torch.tensor(1.0, dtype=torch.float)
        alpha = {}
        
        for j in all_nodes:
            logits_j = theta_dict[j]
            probs_j = F.softmax(logits_j, dim=0)
            children_j = node_to_children[j]
            w_children = probs_j[:-1]
            alpha_j = probs_j[-1]
            alpha[j] = alpha_j
            pj = p_in[j]
            
            for idx, c in enumerate(children_j):
                p_in[c] = p_in[c] + pj * w_children[idx]
        
        Yhat = {j: p_in[j] * alpha[j] for j in all_nodes}
        return Yhat
    
    # Loss computation
    def compute_loss():
        Yhat = forward_pass()
        mse = torch.tensor(0.0, dtype=torch.float)
        for n in all_nodes:
            target = Y.get(n, 0.0)
            diff = Yhat[n] - target
            mse = mse + diff * diff
        return mse, Yhat
    
    # Get edge weights
    def get_edge_weights():
        w_dict = {}
        for j in all_nodes:
            probs_j = F.softmax(theta_dict[j], dim=0)
            w_children = probs_j[:-1]
            child_list = node_to_children[j]
            for idx, c in enumerate(child_list):
                w_dict[(j, c)] = float(w_children[idx])
        return w_dict
    
    # Training loop
    loss_history = []
    theta_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_val, Yhat = compute_loss()
        loss_val.backward()
        optimizer.step()
        
        loss_history.append(loss_val.item())
        theta_history.append(get_edge_weights())
    
    _, final_Yhat = compute_loss()
    final_Yhat = {n: final_Yhat[n].item() for n in all_nodes}
    
    return theta_dict, final_Yhat, loss_history, theta_history


def run_optimization_with_seed(G, Y, root, lr=0.001, epochs=250, seed=None):
    """
    Run a single gradient descent optimization with a specific random seed.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Normalize Y
    Y_norm = {k: v / sum(Y.values()) for k, v in Y.items()}
    
    # Run gradient descent with custom initialization
    try:
        theta_dict, Yhat_dict, loss_history, theta_history = run_gradient_descent_with_init(
            G, Y_norm, root, lr=lr, epochs=epochs, seed=seed
        )
        theta = theta_history[-1]
        
        final_loss = loss_history[-1] if loss_history else np.nan
        
        # Check convergence
        converged = False
        patience = 5
        tol = 1e-5
        if loss_history and len(loss_history) > patience:
            recent = loss_history[-patience:]
            converged = (max(recent) - min(recent)) < tol
        
        return {
            'theta': theta,
            'final_loss': final_loss,
            'converged': converged,
            'loss_history': loss_history
        }
    except Exception as e:
        print(f"Error in optimization with seed {seed}: {e}")
        return None


def compute_edge_weight_statistics(theta_list):
    """
    Compute statistics for edge weights across multiple runs.
    """
    # Get all edges
    all_edges = set()
    for theta in theta_list:
        all_edges.update(theta.keys())
    
    edge_stats = {}
    for edge in all_edges:
        weights = [theta.get(edge, 0.0) for theta in theta_list]
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        cv = std_weight / mean_weight if mean_weight > 0 else np.nan
        
        edge_stats[edge] = {
            'mean': mean_weight,
            'std': std_weight,
            'cv': cv,
            'min': np.min(weights),
            'max': np.max(weights),
            'weights': weights
        }
    
    return edge_stats


def analyze_peptidome_size(n_peptides, n_runs=5, lr=0.01, epochs=500, base_seed=42):
    """
    Analyze robustness for a peptidome of a given size.
    
    """
    print(f"\nAnalyzing peptidome with {n_peptides} peptides...")
    
    # Create a single peptidome (fixed for all runs)
    G, Y, peptides = create_synthetic_peptidome(UMOD, n_peptides, seed=base_seed)
    n_edges = G.number_of_edges()
    n_peptides = len(peptides)
    
    print(f"  Graph has {G.number_of_nodes()} nodes and {n_edges} edges")
    
    # Run optimization multiple times with different seeds
    theta_list = []
    loss_list = []
    loss_histories = []  # Store all loss histories
    converged_count = 0
    
    for run in range(n_runs):
        seed = base_seed + run + 1000 * n_peptides  # Different seed for each run
        result = run_optimization_with_seed(G, Y, UMOD, lr=lr, epochs=epochs, seed=seed)
        
        if result is not None:
            theta_list.append(result['theta'])
            loss_list.append(result['final_loss'])
            loss_histories.append(result['loss_history'])
            if result['converged']:
                converged_count += 1
    
    print(f"  Successfully completed {len(theta_list)}/{n_runs} runs")
    print(f"  Converged: {converged_count}/{n_runs}")
    
    if len(theta_list) == 0:
        return None
    
    # Compute edge weight statistics
    edge_stats = compute_edge_weight_statistics(theta_list)
    
    # Aggregate statistics
    cvs = [stat['cv'] for stat in edge_stats.values() if not np.isnan(stat['cv'])]
    
    return {
        'n_peptides': n_peptides,
        'n_edges': n_edges,
        'n_nodes': G.number_of_nodes(),
        'n_runs': len(theta_list),
        'converged_count': converged_count,
        'edge_stats': edge_stats,
        'mean_cv': np.mean(cvs) if cvs else np.nan,
        'median_cv': np.median(cvs) if cvs else np.nan,
        'std_cv': np.std(cvs) if cvs else np.nan,
        'mean_loss': np.mean(loss_list),
        'std_loss': np.std(loss_list),
        'all_cvs': cvs,
        'loss_histories': loss_histories
    }


def main():
    """s
    Main analysis: evaluate robustness across different peptidome sizes.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Different peptidome sizes to test
    peptidome_sizes = [100, 200, 300, 500, 1000]
    n_runs = 5
    
    print("=" * 60)
    print("ROBUSTNESS ANALYSIS: Edge Weight Stability")
    print("=" * 60)
    print(f"Testing {len(peptidome_sizes)} peptidome sizes")
    print(f"Running {n_runs} independent gradient descent runs per size")
    print()
    
    # Run analysis for each peptidome size
    results = []
    for n_peptides in peptidome_sizes:
        result = analyze_peptidome_size(n_peptides, n_runs=n_runs, lr=0.01, epochs=500, base_seed=42)
        if result is not None:
            results.append(result)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'n_peptides': r['n_peptides'],
        'n_edges': r['n_edges'],
        'n_nodes': r['n_nodes'],
        'mean_cv': r['mean_cv'],
        'median_cv': r['median_cv'],
        'std_cv': r['std_cv'],
        'mean_loss': r['mean_loss'],
        'std_loss': r['std_loss']
    } for r in results])
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    
    # Create visualizations - 4 plots in 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 1. Loss trajectories for all replicates, colored by peptidome size
    ax = axes[0]
    # Use a colormap for different peptidome sizes
    cmap = plt.cm.viridis
    colors = [cmap(i / len(results)) for i in range(len(results))]
    
    for idx, r in enumerate(results):
        for loss_hist in r['loss_histories']:
            ax.plot(loss_hist, color=colors[idx], alpha=0.6, linewidth=1)
    
    # Add legend for peptidome sizes
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, 
                              label=f"{results[i]['n_peptides']} peptides")
                      for i in range(len(results))]
    ax.legend(handles=legend_elements, frameon=False, fontsize=8, ncol=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Trajectories by Peptidome Size')
    ax.set_yscale('log')
    sns.despine(ax=ax)
    
    # 2. Mean CV vs peptidome size
    ax = axes[1]
    ax.plot(summary_df['n_peptides'], summary_df['mean_cv'], 'o-', linewidth=2, markersize=8)
    ax.fill_between(summary_df['n_peptides'], 
                     summary_df['mean_cv'] - summary_df['std_cv'],
                     summary_df['mean_cv'] + summary_df['std_cv'],
                     alpha=0.3)
    ax.set_xlabel('Number of Peptides')
    ax.set_ylabel('Mean Coefficient of Variation')
    ax.set_title('Edge Weight Stability vs. Peptidome Size')
    sns.despine(ax=ax)
    
    # 3. Mean CV vs number of edges
    ax = axes[2]
    ax.plot(summary_df['n_edges'], summary_df['mean_cv'], 'o-', 
            color='tab:orange', linewidth=2, markersize=8)
    ax.fill_between(summary_df['n_edges'], 
                     summary_df['mean_cv'] - summary_df['std_cv'],
                     summary_df['mean_cv'] + summary_df['std_cv'],
                     alpha=0.3, color='tab:orange')
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Mean Coefficient of Variation')
    ax.set_title('Edge Weight Stability vs. Graph Complexity')
    sns.despine(ax=ax)
    
    # 4. Number of edges vs number of peptides
    ax = axes[3]
    ax.plot(summary_df['n_peptides'], summary_df['n_edges'], 'o-', 
            color='tab:green', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Peptides')
    ax.set_ylabel('Number of Edges')
    ax.set_title('Graph Complexity')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig('/Users/erikhartman/dev/degradation-graphs/paper/panels/robustness_analysis.svg', 
                dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(2,2))
    plt.plot(summary_df['n_edges'], summary_df['mean_cv'], 'o-')
    plt.fill_between(summary_df['n_edges'], 
                     summary_df['mean_cv'] - summary_df['std_cv'],
                     summary_df['mean_cv'] + summary_df['std_cv'],
                     alpha=0.3)
    plt.xlabel('Number of edges')
    plt.ylabel('Coefficient of variation')
    sns.despine()
    plt.savefig('/Users/erikhartman/dev/degradation-graphs/paper/panels/robustness_cv_vs_edges.svg', 
                dpi=300, bbox_inches='tight')

    print("Figures saved to paper/panels/")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
