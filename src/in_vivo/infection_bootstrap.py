import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from itertools import product
from collections import defaultdict

# Import custom modules
from src.util import regex_to_graph
from src.weight_optimizer import WeightOptimizer
from src.util import probabilities_to_flows

warnings.filterwarnings('ignore')
sns.set_context("paper")

# HBA protein sequence (hemoglobin alpha)
hba = "VLSAADKANVKAAWGKVGGQAGAHGAEALERMFLGFPTTKTYFPHFNLSHGSDQVKAHGQKVADALTKAVGHLDDLPGALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHHPDDFNPSVHASLDKFLANVSTVLTSKYR"

def process_sample(sample_data: pd.Series, hba_sequence: str, all_peptides: set, quantile: float = 0.2):
    """
    Process a single sample and return peptide abundances with quantile imputation.
    Ensures the root (full-length protein) has abundance 1.0.
    """
    temp_dict = (
        sample_data
        .replace(0, np.nan)
        .replace(-np.inf, np.nan)
        .dropna()
        .to_dict()
    )
    observed_values = list(temp_dict.values()) or [0.0]
    impute_value = np.quantile(observed_values, quantile)

    Y = {}
    for peptide in all_peptides:
        if peptide == hba_sequence:
            Y[peptide] = 1.0  # root mass
        elif peptide in temp_dict and peptide in hba_sequence:
            Y[peptide] = temp_dict[peptide]
        else:
            Y[peptide] = impute_value
    return Y

def topk_bottlenecks(flows: dict, Y: dict, root: str, k: int = 10, use_ratio: bool = False):
    """
    Define bottlenecks as top-k peptides by inflow (or inflow/abundance).
    'flows' is expected to be a dict keyed by edges (u, v) -> flow value.
    """
    inflow = defaultdict(float)
    for (u, v), f in flows.items():
        inflow[v] += f

    scores = {}
    for v, fin in inflow.items():
        if v == root:
            continue
        if use_ratio:
            denom = max(Y.get(v, 0.0), 1e-12)
            scores[v] = fin / denom
        else:
            scores[v] = fin

    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return set(top)

def topk_edges(flows: dict, k: int = 10):
    """
    Define top-k edges by flow magnitude.
    'flows' is expected to be a dict keyed by edges (u, v) -> flow value.
    """
    # Sort edges by flow value and take top k
    top_edges = sorted(flows.items(), key=lambda x: x[1], reverse=True)[:k]
    return set(edge for edge, flow in top_edges)

def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))

def optimize_with_params(G, Y, root, lr, epochs, k=100, use_ratio=False):
    total_abundance_scale = sum(Y.values())
    weight_optimizer = WeightOptimizer()
    try:
        final_theta = weight_optimizer.gradient_descent(
            G, Y, root, lr=lr, epochs=epochs
        )
        # Convert probs -> normalized flows, then scale back to original abundance scale
        flows_norm = probabilities_to_flows(G, w_dict=final_theta, root=root)
        flows = {edge: val * total_abundance_scale for edge, val in flows_norm.items()}

        # Underestimation ratio Δ = total generated flow / total observed abundance (excluding root)
        total_generated = sum(flows.values())
        total_abundance = sum(v for k_, v in Y.items() if k_ != root)
        underestimation_ratio = total_generated / total_abundance if total_abundance > 0 else np.nan

        # Loss diagnostics
        loss_hist = getattr(weight_optimizer, "gd_loss_history", [])
        final_loss = loss_hist[-1] if loss_hist else np.nan

        # Simple "converged" heuristic: last few losses nearly flat
        converged = False
        patience = min(5, max(1, epochs // 10))
        tol = 1e-5
        if loss_hist and len(loss_hist) > patience:
            recent = loss_hist[-patience:]
            converged = (max(recent) - min(recent)) < tol

        # Top-k bottleneck set (peptides)
        topk = topk_bottlenecks(flows, Y, root=root, k=k, use_ratio=use_ratio)
        
        # Top-k edges by flow magnitude
        topk_flows = topk_edges(flows, k=k)

        return {
            'underestimation_ratio': underestimation_ratio,
            'final_loss': final_loss,
            'converged': converged,
            'loss_len': len(loss_hist),
            'loss_hist': loss_hist,
            'topk': topk,
            'topk_flows': topk_flows
        }
    except Exception as e:
        print(f"Error lr={lr}, epochs={epochs}: {e}")
        return {
            'underestimation_ratio': np.nan,
            'final_loss': np.nan,
            'converged': False,
            'loss_len': 0,
            'loss_hist': [],
            'topk': set(),
            'topk_flows': set()
        }

def main():
    # --- Load and process data ---
    enzyme_data = pd.read_csv("/Users/erikhartman/dev/degradation-graphs/data/infection/data.csv")
    design = pd.read_csv("/Users/erikhartman/dev/degradation-graphs/data/infection/design.csv")
    enzyme_data = enzyme_data[enzyme_data["Protein"] == "HBA"]
    
    # Get sample columns (exclude metadata columns)
    intensity_cols = [col for col in enzyme_data.columns if col.startswith("Sample")]
    enzyme_data = enzyme_data[["Peptide"] + intensity_cols]
    enzyme_data = enzyme_data.set_index("Peptide").apply(np.log)

    # One sample (first P. aeruginosa sample)
    p_aeruginosa_samples = design[design["group"] == "P. aeruginosa"]["sample"].tolist()
    sample = p_aeruginosa_samples[15]
    print(f"Using sample: {sample}")

    # Collect peptides present in this sample that occur in HBA sequence
    all_peptides = set()
    if sample in enzyme_data.columns:
        temp_dict = (
            enzyme_data[sample]
            .replace(0, np.nan)
            .replace(-np.inf, np.nan)
            .dropna()
            .to_dict()
        )
        for peptide in temp_dict.keys():
            if peptide in hba:
                all_peptides.add(peptide)
    all_peptides.add(hba)
    print(f"Found {len(all_peptides)} peptides for analysis")

    # Graph over subsequences (pattern can be adjusted)
    G = regex_to_graph(hba, list(all_peptides), r"(.)(.)(.)(.)(.)(.)")

    # Observed abundances with simple quantile imputation
    Y = process_sample(enzyme_data[sample], hba, all_peptides)

    # --- Parameter sweep setup ---
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    epochs_list = [50, 100, 250, 500, 1000]

    # --- Run grid ---
    results = []
    print("Running parameter sweep...")
    for lr, epochs in product(learning_rates, epochs_list):
        print(f"Testing lr={lr}, epochs={epochs}")
        result = optimize_with_params(G, Y, hba, lr, epochs)
        results.append({
            'lr': lr,
            'epochs': epochs,
            **result
        })

    # Results DataFrame
    df = pd.DataFrame(results)

    # --- Aggregates for plotting ---
    df_mean = df.groupby(['lr', 'epochs'], as_index=False).agg({
        'underestimation_ratio': 'mean',
        'final_loss': 'mean',
        'converged': 'mean'  # mean of bool -> fraction converged
    })

    # Pivots
    underestimation_pivot = df_mean.pivot(index='epochs', columns='lr', values='underestimation_ratio')
    loss_pivot = df_mean.pivot(index='epochs', columns='lr', values='final_loss')
    conv_pivot = df_mean.pivot(index='epochs', columns='lr', values='converged')

    # --- Stability vs baseline (best loss) ---
    # pick baseline setting with minimum loss
    idx_min = df['final_loss'].idxmin()
    baseline = df.loc[idx_min]
    baseline_topk = baseline['topk']
    baseline_lr = baseline['lr']
    baseline_epochs = baseline['epochs']
    print(f"Baseline (min loss): lr={baseline_lr}, epochs={baseline_epochs}")

    # Jaccard vs baseline for each setting (peptides)
    df_mean['jaccard_vs_baseline'] = df_mean.apply(
        lambda row: jaccard(
            df[(df.lr == row['lr']) & (df.epochs == row['epochs'])].iloc[0]['topk'],
            baseline_topk
        ), axis=1
    )
    # Convert to percentage
    df_mean['percent_similarity_peptides'] = df_mean['jaccard_vs_baseline'] * 100
    percent_peptides_pivot = df_mean.pivot(index='epochs', columns='lr', values='percent_similarity_peptides')
    
    # Edge-based Jaccard vs baseline for each setting  
    baseline_topk_flows = baseline['topk_flows']
    df_mean['jaccard_edges_vs_baseline'] = df_mean.apply(
        lambda row: jaccard(
            df[(df.lr == row['lr']) & (df.epochs == row['epochs'])].iloc[0]['topk_flows'],
            baseline_topk_flows
        ), axis=1
    )
    # Convert to percentage
    df_mean['percent_similarity_edges'] = df_mean['jaccard_edges_vs_baseline'] * 100
    percent_edges_pivot = df_mean.pivot(index='epochs', columns='lr', values='percent_similarity_edges')

    # --- Combined plots in one figure ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    
    # 1) Δ heatmap
    sns.heatmap(underestimation_pivot, annot=True, fmt='.3f', cmap='Oranges', ax=axes[0], cbar_kws={'label': r'$\Delta$'})
    axes[0].set_title(r'$\Delta$')
    axes[0].set_xlabel('Learning rate')
    axes[0].set_ylabel('Epochs')

    # 2) Final loss heatmap
    sns.heatmap(loss_pivot, annot=True, fmt='.3f', cmap='Purples', ax=axes[1], cbar_kws={'label': 'final loss (MSE)'})
    axes[1].set_title('final loss (MSE)')
    axes[1].set_xlabel('Learning rate')
    axes[1].set_ylabel('Epochs')

    # 3) Percentage similarity vs baseline heatmap (peptides)
    sns.heatmap(percent_peptides_pivot, annot=True, fmt='.1f', cmap='Greens', ax=axes[2], vmin=0, vmax=100, cbar_kws={'label': '% similarity'})
    axes[2].set_title(f'Top-100 peptide stability')
    axes[2].set_xlabel('Learning rate')
    axes[2].set_ylabel('Epochs')
    
    # 4) Percentage similarity vs baseline heatmap (edges)
    sns.heatmap(percent_edges_pivot, annot=True, fmt='.1f', cmap='Blues', ax=axes[3], vmin=0, vmax=100, cbar_kws={'label': '% similarity'})
    axes[3].set_title(f'Top-100 edge stability')
    axes[3].set_xlabel('Learning rate')
    axes[3].set_ylabel('Epochs')
    
    plt.tight_layout()
    plt.savefig('/Users/erikhartman/dev/degradation-graphs/paper/panels/infection_combined_heatmaps.svg', dpi=300, bbox_inches='tight')
    plt.savefig('/Users/erikhartman/dev/degradation-graphs/paper/panels/infection_combined_heatmaps.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()