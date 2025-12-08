import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Import custom modules
from src.util import regex_to_graph, probabilities_to_flows
from src.weight_optimizer import WeightOptimizer
from src.proteolysis_simulator import ProteolysisSimulator, Enzyme

warnings.filterwarnings('ignore')
sns.set_context("paper")

# UMOD protein sequence
UMOD = "MGQPSLTWMLMVVVASWFITTAATDTSEARWCSECHSNATCTEDEAVTTCTCQEGFTGDGLTCVDLDECAIPGAHNCSANSSCVNTPGSFSCVCPEGFRLSPGLGCTDVDECAIPGAHNCSANSSCVNTPGSFSCVCPEGFRLSPGLGCTDVDECAEPGLSHCHALATCVNVVGSYLCVCPAGYRGDGWHCECSPGSCGPGLDCVPEGDALVCADPCQAHRTLDEYWRSTEYGEGYACDTDLRGWYRFVGQGGARMAETCVPVLRCNTAAPMWLNGTHPSSDEGIVSRKACAHWSGHCCLWDASVQVKACAGGYYVYNLTAPPECHLAYCTDPSSVEGTCEECSIDEDCKSNNGRWHCQCKQDFNITDISLLEHRLECGANDMKVSLGKCQLKSLGFDKVFMYLSDSRCSGFNDRDNRDWVSVVTPARDGPCGTVLTRNETHATYSNTLYLADEIIIRDLNIKINFACSYPLDMKVSLKTALQPMVSALNIRVGGTGMFTVRMALFQTPSYTQPYQGSSVTLSTEAFLYVGTMLDGGDLSRFALLMTNCYATPSSNATDPLKYFIIQDRCPHTRDSTIQVVENGESSQGRFSVQMFRFAGNYDLVYLHCEVYLCDTMNEKCKPTCSGTRFRSGSVIDQSRVLNLGPITRKGVQATVSRAFSSLGLLKVWLPLLLSATLTLTFQ"


def create_synthetic_peptidome(protein_sequence, n_peptides, pattern=r"(.)(.)([KR])(.)(.)(.)", seed=None):
    """
    Create a synthetic peptidome using the proteolysis simulator.
    
    This generates a more realistic peptidome with proper enzymatic cleavage patterns
    and abundance distributions.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Define a trypsin-like enzyme (cleaves after K or R)
    trypsin = Enzyme([(pattern, 1)])
    
    # Create simulator
    simulator = ProteolysisSimulator(
        min_length=6,
        length_params="vitro",
        random_seed=seed,
        verbose=False
    )
    
    # Simulate proteolysis to generate target number of peptides
    # Scale n_succesful_cleaves to get approximately n_peptides unique peptides
    target_cleaves = int(n_peptides * 1.5)
    
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
    
    # If we got more peptides than requested, take the top n_peptides by abundance
    if len(peptides) > n_peptides + 1:
        sorted_peptides = sorted(P_Y.items(), key=lambda x: x[1], reverse=True)
        peptides = [p for p, _ in sorted_peptides[:n_peptides + 1]]
        # Make sure the full protein is included
        if protein_sequence not in peptides:
            peptides[0] = protein_sequence
    
    # Create graph
    G = regex_to_graph(protein_sequence, peptides, pattern)
    
    # Use simulated abundances (normalized)
    Y = {p: P_Y.get(p, 0) for p in peptides}
    total = sum(Y.values())
    Y = {k: v / total for k, v in Y.items()}
    
    return G, Y, peptides


def compute_underestimation_ratio(G, Y, root, lr=0.05, epochs=100, seed=None):
    """
    Compute the underestimation ratio: flow/abundance for the root node.
    
    The flow (p_in) represents the true protein abundance, while Y represents
    the observed peptide abundance. The ratio shows how much we underestimate
    protein abundance when using peptide-based quantification.
    """
    # Normalize Y
    Y_norm = {k: v / sum(Y.values()) for k, v in Y.items()}
    
    # Run gradient descent optimization
    optimizer = WeightOptimizer()
    theta = optimizer.gradient_descent(
        G, Y_norm, root, lr=lr, epochs=epochs, verbose=False, seed=seed
    )
    
    # Get theta history to extract final edge weights
    theta_history = optimizer.gd_theta_history
    
    # Compute flows using probabilities_to_flows
    flows = probabilities_to_flows(G, w_dict=theta_history[-1], root=root)
    
    ratio = sum(flows.values()) / sum(Y.values())
    
    return ratio


def analyze_peptidome_size(n_peptides, n_runs=5, lr=0.05, epochs=100, base_seed=42):
    """
    Analyze underestimation ratio for a peptidome of a given size.
    """
    print(f"\nAnalyzing peptidome with {n_peptides} peptides...")
    
    # Create peptidome
    G, Y, peptides = create_synthetic_peptidome(UMOD, n_peptides, seed=base_seed)
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    
    print(f"  Graph has {n_nodes} nodes and {n_edges} edges")
    
    # Run optimization multiple times with different seeds
    ratios = []

    for run in range(n_runs):
        seed = base_seed + run + 1000 * n_peptides
        try:
            ratio = compute_underestimation_ratio(
                G, Y, UMOD, lr=lr, epochs=epochs, seed=seed
            )
            ratios.append(ratio)
        except Exception as e:
            print(f"  Error in run {run}: {e}")
    
    print(f"  Successfully completed {len(ratios)}/{n_runs} runs")
    print(f"  Mean underestimation ratio: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    
    if len(ratios) == 0:
        return None
    
    return {
        'n_peptides': n_peptides,
        'n_edges': n_edges,
        'n_nodes': n_nodes,
        'mean_ratio': np.mean(ratios),
        'std_ratio': np.std(ratios),
        'all_ratios': ratios
    }


def main():
    """
    Main analysis: evaluate underestimation ratio across different peptidome sizes.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Different peptidome sizes to test
    peptidome_sizes = [50, 100, 200, 500, 1000, 1500]
    n_runs = 5
    
    print("=" * 60)
    print("UNDERESTIMATION RATIO ANALYSIS")
    print("=" * 60)
    print(f"Testing {len(peptidome_sizes)} peptidome sizes")
    print(f"Running {n_runs} independent runs per size")
    print()
    
    # Run analysis for each peptidome size
    results = []
    for n_peptides in peptidome_sizes:
        result = analyze_peptidome_size(
            n_peptides, n_runs=n_runs, lr=0.01, epochs=500, base_seed=42
        )
        if result is not None:
            results.append(result)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'n_peptides': r['n_peptides'],
        'n_edges': r['n_edges'],
        'n_nodes': r['n_nodes'],
        'mean_ratio': r['mean_ratio'],
        'std_ratio': r['std_ratio'],
    } for r in results])
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
 
    # Create single panel figure
    plt.figure(figsize=(4, 3))
    plt.plot(summary_df['n_edges'], summary_df['mean_ratio'], 'o-', 
             linewidth=2, markersize=4)
    plt.xlabel('Number of edges')
    plt.ylabel(r'$\Delta$')
    plt.tight_layout()
    plt.savefig('/Users/erikhartman/dev/degradation-graphs/paper/panels/underestimation_single.svg', 
                dpi=300, bbox_inches='tight')
    print("Single panel figure saved to paper/panels/underestimation_single.svg")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
