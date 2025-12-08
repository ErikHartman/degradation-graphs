import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GraphConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

sns.set_context("paper")


class EnzymeGraphConv(nn.Module):
    """Graph neural network for enzyme classification."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.linear(x), dim=1)


def load_real_data(csv_path, protein, treatment_cols):
    """
    Load real peptide data for a specific protein and treatment columns.
    """
    df = pd.read_csv(csv_path)
    
    # Filter for the specific protein
    protein_df = df[df['Protein Accession'] == protein].copy()
    
    if len(protein_df) == 0:
        raise ValueError(f"No data found for protein {protein}")
    
    # Get all columns matching the treatment patterns
    treatment_data = {}
    for treatment in treatment_cols:
        # Match columns like "Area trp1", "Area trp2", etc.
        pattern = f"Area {treatment}\\d+"
        cols = [col for col in protein_df.columns if re.match(pattern, col)]
        if len(cols) == 0:
            raise ValueError(f"No columns found for treatment {treatment}")
        treatment_data[treatment] = cols
    
    print(f"Found protein {protein} with {len(protein_df)} peptides")
    for treatment, cols in treatment_data.items():
        print(f"  Treatment {treatment}: {len(cols)} replicates")
    
    return protein_df, treatment_data


def build_degradation_graph(peptides, parent_sequence):
    G = nx.DiGraph()
    
    # Add parent sequence as root node
    G.add_node(parent_sequence)
    
    # Filter peptides that are actually in the parent sequence
    valid_peptides = [pep for pep in peptides if pep in parent_sequence and pep != parent_sequence]
    
    # Add all valid peptides as nodes
    for pep in valid_peptides:
        G.add_node(pep)
    
    # Add edges from parent to all direct peptides
    for pep in valid_peptides:
        # Check if this peptide could be a direct cleavage from parent
        # (we add edge from parent to all peptides - they all come from the parent)
        G.add_edge(parent_sequence, pep)
    
    # Add edges where one peptide is contained in another
    for i, pep1 in enumerate(valid_peptides):
        for j, pep2 in enumerate(valid_peptides):
            if i != j and pep2 in pep1 and pep2 != pep1:
                # pep1 degrades to pep2 (pep2 is shorter and contained in pep1)
                G.add_edge(pep1, pep2)
    
    return G


def get_position_features(peptide, parent_sequence):
    """Get normalized position features for a peptide."""
    start_pos = parent_sequence.find(peptide)
    if start_pos == -1:
        # If peptide not found, use default values
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    
    end_pos = start_pos + len(peptide) - 1
    normalized_start = start_pos / max(1, len(parent_sequence) - 1)
    normalized_end = end_pos / max(1, len(parent_sequence) - 1)
    normalized_length = len(peptide) / max(1, len(parent_sequence) - 1)
    
    return torch.tensor([normalized_start, normalized_end, normalized_length], dtype=torch.float)


def create_graph_from_sample(peptide_df, sample_cols, parent_sequence, label):
    """
    Create a PyTorch Geometric graph from peptide abundances in one sample.
    """
    # Get peptides and their abundances
    peptides = peptide_df['Peptide'].tolist()
    
    # Average abundance across replicates for this treatment
    abundances = peptide_df[sample_cols].mean(axis=1).fillna(0).values
    
    # Filter for peptides with non-zero abundance
    non_zero_mask = abundances > 0
    peptides_filtered = [p for i, p in enumerate(peptides) if non_zero_mask[i]]
    abundances = abundances[non_zero_mask]
    
    print(f"    Sample has {len(peptides_filtered)} peptides with non-zero abundance (out of {len(peptides)} total)")
    
    # Normalize abundances
    if abundances.max() > 0:
        abundances = abundances / abundances.max()
    
    # Build degradation graph (using filtered peptides and parent)
    G = build_degradation_graph(peptides_filtered, parent_sequence)
    
    # Create node mapping (parent + peptides)
    all_nodes = [parent_sequence] + peptides_filtered
    node_map = {pep: i for i, pep in enumerate(all_nodes)}
    
    # Create node features (abundance + position)
    # Parent sequence has abundance = 1.0 (fully intact protein)
    node_features = []
    
    # Parent sequence features
    parent_features = torch.cat([
        torch.tensor([1.0], dtype=torch.float),  # Full abundance
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.float)  # start=0, end=1, length=1 (normalized)
    ])
    node_features.append(parent_features)
    
    # Peptide features
    # Peptide features
    for pep in peptides_filtered:
        idx = peptides_filtered.index(pep)
        abundance = abundances[idx]
        pos_features = get_position_features(pep, parent_sequence)
        features = torch.cat([
            torch.tensor([abundance], dtype=torch.float),
            pos_features
        ])
        node_features.append(features)
    
    node_features = torch.stack(node_features)
    
    # Create edge index and weights
    edge_list = list(G.edges())
    if len(edge_list) == 0:
        # If no edges, create a self-loop to avoid empty graph
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_weights = torch.tensor([1.0], dtype=torch.float)
    else:
        edge_index = torch.tensor(
            [[node_map[u], node_map[v]] for u, v in edge_list],
            dtype=torch.long
        ).t().contiguous()
        
        # Simple edge weights based on size difference
        edge_weights = torch.ones(len(edge_list), dtype=torch.float)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weights.unsqueeze(1),
        y=torch.tensor([label], dtype=torch.long)
    )
    
    return data


def prepare_dataset(csv_path, protein, treatment_cols, parent_sequence):
    """
    Prepare dataset from real data.
    
    Args:
        csv_path: Path to CSV file
        protein: Protein name
        treatment_cols: List of treatment prefixes (e.g., ["trp", "el", "cht", "lp"])
        parent_sequence: Parent protein sequence
    
    Returns:
        List of PyTorch Geometric Data objects and labels
    """
    peptide_df, treatment_data = load_real_data(csv_path, protein, treatment_cols)
    
    dataset = []
    labels = []
    
    # Create graphs for each treatment
    for label_idx, treatment in enumerate(treatment_cols):
        treatment_sample_cols = treatment_data[treatment]
        for col in treatment_sample_cols:
            data = create_graph_from_sample(peptide_df, [col], parent_sequence, label=label_idx)
            dataset.append(data)
            labels.append(label_idx)
    
    print(f"\nDataset created:")
    for label_idx, treatment in enumerate(treatment_cols):
        n_samples = len(treatment_data[treatment])
        print(f"  Treatment {treatment}: {n_samples} samples (label {label_idx})")
    print(f"  Total: {len(dataset)} samples")
    print(f"  Feature dim: {dataset[0].x.shape[1]}")
    
    return dataset, labels


def train_model(train_dataset, val_dataset, n_classes=4, epochs=100, lr=0.01):
    """Train the GNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    in_channels = train_dataset[0].x.shape[1]
    model = EnzymeGraphConv(in_channels=in_channels, hidden_channels=32, out_channels=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_attr.squeeze(1))
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_attr.squeeze(1))
                loss = F.nll_loss(out, data.y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return model


def evaluate_model(model, val_dataset):
    """Evaluate model and return predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_attr.squeeze(1))
            probs = torch.exp(out).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(data.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_labels


def plot_roc(all_fold_results, treatment_names, save_path='results/figures/real_gnn_roc.svg'):
    """Plot ROC curves with cross-validation folds."""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    n_classes = len(treatment_names)
    
    if n_classes == 2:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        fold_aucs = []
        for fold_idx, (probs, labels) in enumerate(all_fold_results):
            probs_class1 = probs[:, 1]
            fpr, tpr, _ = roc_curve(labels, probs_class1)
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)
            ax.plot(fpr, tpr, color='#3274A1', linewidth=1, alpha=0.3)
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        
        probs, labels = all_fold_results[0]
        probs_class1 = probs[:, 1]
        fpr, tpr, _ = roc_curve(labels, probs_class1)
        ax.plot(fpr, tpr, color='#3274A1', linewidth=2, 
                label=f'Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', frameon=False)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC plot saved to {save_path}")
        plt.close()
        
        return fold_aucs
    else:
        # Multi-class classification - one-vs-rest ROC curves
        colors = sns.palettes.color_palette("viridis", n_classes)
        
        fig, ax = plt.subplots(1, 1, figsize=(3,3))
        
        # Combine all folds for visualization
        all_probs_combined = []
        all_labels_combined = []
        for probs, labels in all_fold_results:
            all_probs_combined.append(probs)
            all_labels_combined.append(labels)
        
        all_probs_combined = np.vstack(all_probs_combined)
        all_labels_combined = np.concatenate(all_labels_combined)
        
        # Binarize labels for one-vs-rest
        labels_bin = label_binarize(all_labels_combined, classes=range(n_classes))
        
        # Calculate ROC curve and AUC for each class
        class_aucs = []
        for i, (color, treatment) in enumerate(zip(colors, treatment_names)):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs_combined[:, i])
            roc_auc = auc(fpr, tpr)
            class_aucs.append(roc_auc)
            
            ax.plot(fpr, tpr, color=color, linewidth=2, 
                   label=f'{treatment} (AUC = {roc_auc:.3f})')
        
        # Calculate macro-average AUC (from fold results)
        fold_aucs = []
        for fold_idx, (probs, labels) in enumerate(all_fold_results):
            labels_bin_fold = label_binarize(labels, classes=range(n_classes))
            fold_auc = roc_auc_score(labels_bin_fold, probs, average='macro', multi_class='ovr')
            fold_aucs.append(fold_auc)
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower right', frameon=False, fontsize=9)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC plot saved to {save_path}")
        plt.close()
        
        return fold_aucs



def plot_peptidome(V_Omega: str, P_M: dict, ax, max_range=None, cmap=None, center_colormap=True):
    """
    Plots peptides along the backbone of V_Omega colored by fold change or intensity.
    """
    
    # Create custom Orange-Blue colormap with white center (for fold changes)
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        colors = ["#EE8F19", '#FFA500', "#B2D5E0", '#87CEEB', "#5252F5"]
        cmap = LinearSegmentedColormap.from_list('orange_blue', colors, N=256)
    
    # Sort peptides (descending by length) so that longer peptides try to place first
    P_M = dict(sorted(P_M.items(), key=lambda item: len(item[0]), reverse=True))

    # Determine colormap range
    if center_colormap:
        # Center around 0 for fold changes
        if max_range is None:
            max_range = max(abs(v) for v in P_M.values()) if P_M.values() else 1
        min_range = -max_range
    else:
        # Use actual min/max for intensities
        if max_range is None:
            max_range = max(P_M.values()) if P_M.values() else 1
        min_range = min(P_M.values()) if P_M.values() else 0

    # Initialize a 2D "space" array to avoid overlapping lines
    spaces = np.zeros((len(P_M.keys()), len(V_Omega)), dtype=float)

    # Loop over each peptide in P_M (except the parent itself)
    for sequence, value in P_M.items():
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

                # Color from colormap based on value
                norm_val = (value - min_range) / (max_range - min_range) if max_range > min_range else 0.5
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
    
    # Label based on whether it's centered or not
    if center_colormap:
        cb_label = r"$log_2$ Fold change"
    else:
        cb_label = "Intensity"
    
    cb = plt.colorbar(sm, cax=cax, ax=ax, label=cb_label)
    
    return ax


def calculate_peptide_fold_changes(peptide_df, treatment_data, treatment_cols, parent_sequence):
    """
    Calculate log2 fold changes for each peptide between two treatments.
    """
    peptides = peptide_df['Peptide'].tolist()
    
    # Get mean abundances for each treatment
    treatment0_cols = treatment_data[treatment_cols[0]]
    treatment1_cols = treatment_data[treatment_cols[1]]
    
    treatment0_mean = peptide_df[treatment0_cols].mean(axis=1).values
    treatment1_mean = peptide_df[treatment1_cols].mean(axis=1).values
    
    peptide_fold_changes = {}
    
    for i, peptide in enumerate(peptides):
        if peptide not in parent_sequence or peptide == parent_sequence:
            continue
        
        t0_abund = treatment0_mean[i]
        t1_abund = treatment1_mean[i]
        
        # Calculate fold change (treatment1 / treatment0)
        # Add small pseudocount to avoid division by zero
        pseudocount = 1e-10
        fold_change = (t1_abund + pseudocount) / (t0_abund + pseudocount)
        log2_fc = np.log2(fold_change)
        
        peptide_fold_changes[peptide] = log2_fc
    
    return peptide_fold_changes


def calculate_peptide_intensities(peptide_df, treatment_data, treatment_name, parent_sequence):
    """
    Calculate mean intensities for each peptide in a specific treatment.
    
    Args:
        peptide_df: DataFrame with peptide sequences
        treatment_data: Dict mapping treatment names to column lists
        treatment_name: Name of the treatment
        parent_sequence: Parent protein sequence
    
    Returns:
        Dict mapping peptide sequences to mean intensities
    """
    peptides = peptide_df['Peptide'].tolist()
    
    # Get mean abundances for this treatment (ignoring NaN values)
    treatment_cols = treatment_data[treatment_name]
    treatment_mean = peptide_df[treatment_cols].mean(axis=1, skipna=True).values
    
    peptide_intensities = {}
    
    for i, peptide in enumerate(peptides):
        if peptide not in parent_sequence or peptide == parent_sequence:
            continue
        
        intensity = treatment_mean[i]
        # Skip peptides with no data (NaN)
        if not np.isnan(intensity):
            peptide_intensities[peptide] = intensity
    
    return peptide_intensities


def main():
    """Main execution function."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    CSV_PATH = 'data/log_processed_area.csv'
    PROTEIN = 'ACTB'
    TREATMENT_COLS = ['trp', 'el', 'cht', 'lp']  # all four enzymes
    N_FOLDS = 5
    
    # Parent sequence for ACTB (from enzyme_gnn.py)
    PARENT_SEQUENCE = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF"
    
    print(f"Training GNN on real data with {N_FOLDS}-fold cross-validation:")
    print(f"  Protein: {PROTEIN}")
    print(f"  Treatments: {', '.join(TREATMENT_COLS)}")
    print()
    
    # Load and prepare data
    dataset, labels = prepare_dataset(CSV_PATH, PROTEIN, TREATMENT_COLS, PARENT_SEQUENCE)
    
    # Convert labels to numpy (dataset stays as list of Data objects)
    labels_array = np.array(labels)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    all_fold_results = []
    fold_metrics = []
    
    # Create dummy X for split (just indices)
    X_dummy = np.arange(len(dataset)).reshape(-1, 1)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dummy, labels_array)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*50}")
        
        # Split data
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        
        train_labels = labels_array[train_idx]
        val_labels = labels_array[val_idx]
        
        print(f"  Train: {len(train_dataset)} samples (class 0: {np.sum(train_labels==0)}, class 1: {np.sum(train_labels==1)}, class 2: {np.sum(train_labels==2)}, class 3: {np.sum(train_labels==3)})")
        print(f"  Val: {len(val_dataset)} samples (class 0: {np.sum(val_labels==0)}, class 1: {np.sum(val_labels==1)}, class 2: {np.sum(val_labels==2)}, class 3: {np.sum(val_labels==3)})")
        
        # Train model
        model = train_model(train_dataset, val_dataset, n_classes=len(TREATMENT_COLS), epochs=100, lr=0.01)
        
        # Evaluate
        probs, true_labels = evaluate_model(model, val_dataset)
        all_fold_results.append((probs, true_labels))
        
        # Calculate metrics
        probs_class1 = probs[:, 1] if len(TREATMENT_COLS) == 2 else None
        
        if len(TREATMENT_COLS) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(true_labels, probs_class1)
            fold_auc = auc(fpr, tpr)
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_auc_score
            labels_bin = label_binarize(true_labels, classes=range(len(TREATMENT_COLS)))
            fold_auc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
        
        fold_metrics.append({'fold': fold_idx + 1, 'auc': fold_auc})
        
        print(f"  Fold {fold_idx + 1} AUC: {fold_auc:.4f}")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("Cross-Validation Results:")
    print(f"{'='*50}")
    aucs = [m['auc'] for m in fold_metrics]
    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Min AUC: {np.min(aucs):.4f}")
    print(f"Max AUC: {np.max(aucs):.4f}")
    print()
    
    for metric in fold_metrics:
        print(f"  Fold {metric['fold']}: AUC = {metric['auc']:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    fold_aucs = plot_roc(all_fold_results, TREATMENT_COLS)
    
    # Calculate and plot peptidome intensities for each treatment
    print("\nGenerating peptidome plots...")
    peptide_df, treatment_data = load_real_data(CSV_PATH, PROTEIN, TREATMENT_COLS)
    
    # Plot for each treatment
    for treatment in TREATMENT_COLS:
        peptide_intensities = calculate_peptide_intensities(
            peptide_df, treatment_data, treatment, PARENT_SEQUENCE
        )
        
        fig, ax = plt.subplots(figsize=(3, 2))
        plot_peptidome(
            V_Omega=PARENT_SEQUENCE,
            P_M=peptide_intensities,
            ax=ax,
            cmap=plt.cm.viridis,
            center_colormap=False
        )
        
        ax.set_yticks([])
        ax.set_xlabel("Backbone index")
        ax.set_title(f'{PROTEIN}: {treatment}')
        sns.despine(ax=ax, left=True, bottom=True)
        
        plt.tight_layout()
        peptidome_path = f'results/figures/real_gnn_peptidome_{treatment}.svg'
        plt.savefig(peptidome_path, dpi=300, bbox_inches='tight')
        print(f"Peptidome plot for {treatment} saved to {peptidome_path}")
        plt.close()
    
    print("\nDone!")



if __name__ == "__main__":
    main()
