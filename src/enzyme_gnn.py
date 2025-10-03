import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os
sns.set_context("paper")
from proteolysis_simulator import Enzyme, ProteolysisSimulator

actb = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF"
hbb = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
thrb = "MAHVRGLQLPGCLALAALCSLVHSQHVFLAPQQARSLLQRVRRANTFLEEVRKGNLERECVEETCSYEEAFEALESSTATDVFWAKYTACETARTPRDKLAACLEGNCAEGLGTNYRGHVNITRSGIECQLWRSRYPHKPEINSTTHPGADLQENFCRNPDSSTTGPWCYTTDPTVRRQECSIPVCGQDQVTVAMTPRSEGSSVNLSPPLEQCVPDRGQQYQGRLAVTTHGLPCLAWASAQAKALSKHQDFNSAVQLVENFCRNPDGDEEGVWCYVAGKPGDFGYCDLNYCEEAVEEETGDGLDEDSDRAIEGRTATSEYQTFFNPRTFGSGEADCGLRPLFEKKSLEDKTERELLESYIDGRIVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIHPRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETWTANVGKGQPSVLQVVNLPIVERPVCKDSTRIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE"
apoa1 = "MKAAVLTLAVLFLTGSQARHFWQQDEPPQSPWDRVKDLATVYVDVLKDSGRDYVSQFEGSALGKQLNLKLLDNWDSVTSTFSKLREQLGPVTQEFWDNLEKETEGLRQEMSKDLEEVKAKVQPYLDDFQKKWQEEMELYRQKVEPLRAELQEGARQKLHELQEKLSPLGEEMRDRARAHVDALRTHLAPYSDELRQRLAARLEALKENGGARLAEYHAKATEHLSTLSEKAKPALEDLRQGLLPVLESFKVSFLSALEEYTKKLNTQ"

def define_proteins():
    """Define all protein sequences."""
    return {
        "ACTB": actb,
        "HBB": hbb,
        "THRB": thrb,
        "APOA1": apoa1
    }

def define_train_enzymes():
    return {
        "trypsin": Enzyme([("(.)(.)([R|K])([^P])(.)(.)", 3), ("(.)(.)(.)(.)(.)(.)", 1)]),
        "elastase": Enzyme([("(.)(.)([V|I|A|S|L|G])(.)(.)(.)", 3), ("(.)(.)(.)(.)(.)(.)", 1)]),
    }




class EnzymeGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


def get_position_features(peptide, parent_sequence):

    start_pos = parent_sequence.find(peptide)
    end_pos = start_pos + len(peptide) - 1
    
    normalized_start = start_pos / (len(parent_sequence) - 1)
    normalized_end = end_pos / (len(parent_sequence) - 1)
    normalized_length = len(peptide) / (len(parent_sequence) - 1)
    
    return torch.tensor([normalized_start, normalized_end, normalized_length], dtype=torch.float)


def generate_dataset_for_training(enzymes, n_samples=100, sequence=actb, protein_name="ACTB"):

    ps = ProteolysisSimulator(verbose=False)
    
    enzyme_indices = {name: i for i, name in enumerate(enzymes.keys())}

    dataset = []
    labels = []
    enzyme_types = []
    protein_names = []
    raw_abundance_list = []

    for enzyme_name, enzyme in enzymes.items():
        print(f"[DATA GENERATION] {enzyme_name}")
        class_idx = enzyme_indices[enzyme_name]
        for i in range(n_samples):
            Y, _ = ps.simulate_proteolysis(
                sequence=sequence,
                n_succesful_cleaves=100,
                enzyme=enzyme,
                make_graph=True,
                endo_probability=0.5,
            )
            G_true = ps.finalize_graph_as_probabilities()

            final_theta = {(u,v):data["v"] for u,v,data in G_true.edges(data=True)}

            node_map = {p: i for i, p in enumerate(G_true.nodes())}
            edge_index = [[node_map[p1], node_map[p2]] for (p1, p2) in final_theta.keys()]
            edge_weights = list(final_theta.values())

            # Get peptide abundances
            abundances = torch.tensor([Y.get(peptide, 0.0) for peptide in G_true.nodes()],
                                     dtype=torch.float)
            if abundances.max() > 0:
                abundances /= abundances.max()
            
            # Get position features for each peptide
            position_features = torch.stack([
                get_position_features(peptide, sequence) 
                for peptide in G_true.nodes()
            ])
            
            node_features = torch.cat([
                abundances.unsqueeze(1),
                position_features
            ], dim=1)  # Shape: [num_nodes, 4] (abundance + 3 position features)

            data = Data(
                x=node_features,
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1),
                y=torch.tensor([class_idx], dtype=torch.long),
            )
            dataset.append(data)
            labels.append(class_idx)
            enzyme_types.append(enzyme_name)
            protein_names.append(protein_name)
            raw_abundance_list.append((abundances.numpy(), class_idx))

    return dataset, labels, enzyme_types, protein_names, raw_abundance_list



def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_with_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out)  # Convert log_softmax to probabilities
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    if all_probs:
        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        return all_probs, all_labels
    return np.array([]), np.array([])


def train_and_evaluate(train_dataset, val_dataset, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    in_channels = train_dataset[0].x.shape[1]
    
    model = EnzymeGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        

        model.eval()
        val_loss = 0
        for data in val_loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    

    probs, labels = evaluate_with_probs(model, val_loader, device)
    

    elastase_probs = probs[:, 1]
    

    fpr, tpr, _ = roc_curve(labels, elastase_probs)
    roc_auc = auc(fpr, tpr)
    
    pr_auc = average_precision_score(labels, elastase_probs)
    
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.plot(fpr, tpr, color='#3274A1', lw=2)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc="lower right", frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('results/figures/roc_curve.png', 
                dpi=300, bbox_inches='tight')
    
    return model, roc_auc, pr_auc


def plot_multi_protein_roc(model, val_dataset, val_proteins, val_enzymes, device):
    """Plot ROC curves for all protein-enzyme combinations."""
    model.eval()
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Get predictions
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(data.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Organize data by protein (not protein-enzyme)
    protein_data = {}
    for i, (protein, enzyme, label) in enumerate(zip(val_proteins, val_enzymes, all_labels)):
        if protein not in protein_data:
            protein_data[protein] = {'probs': [], 'labels': []}
        protein_data[protein]['probs'].append(all_probs[i, 1])  # elastase probability
        protein_data[protein]['labels'].append(label)  # 0=trypsin, 1=elastase
    
    # Debug: Print data distribution
    print("\nValidation set distribution by protein:")
    for protein, data in protein_data.items():
        labels = np.array(data['labels'])
        n_trypsin = np.sum(labels == 0)
        n_elastase = np.sum(labels == 1)
        print(f"  {protein}: {len(labels)} samples (Trypsin: {n_trypsin}, Elastase: {n_elastase})")
    
    # Plot settings
    proteins = list(define_proteins().keys())
    protein_colors = {'ACTB': '#3274A1', 'HBB': "#4AB45A", 'THRB': '#45a884', 'APOA1': "#6b50ce"}

    fig, ax = plt.subplots(1, 1, figsize=(3,3))

    curves_plotted = 0
    
    for protein in proteins:
        if protein in protein_data:
            probs = np.array(protein_data[protein]['probs'])
            labels = np.array(protein_data[protein]['labels'])
            
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, 
                    color=protein_colors[protein], linewidth=2,
                    label=f'{protein} (AUC={roc_auc:.3f})')
            curves_plotted += 1
           
    all_probs_combined = all_probs[:, 1]
    fpr_overall, tpr_overall, _ = roc_curve(all_labels, all_probs_combined)
    auc_overall = auc(fpr_overall, tpr_overall)

    ax.plot(fpr_overall, tpr_overall, linewidth=2,
           color='black',
           label=f'Overall (AUC={auc_overall:.3f})')
    curves_plotted += 1

    ax.set_xlim([-0.05, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    ax.legend(loc='lower right', frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.savefig('results/figures/multi_protein_roc.png', dpi=300, bbox_inches='tight')
    
    print(f"ROC plot saved with {curves_plotted} curves (4 per-protein + 1 overall)")
 


def plot_enzyme_distributions(model, val_dataset, device):
    """Plot KDE distributions in 2x2 subplots for each protein."""
    model.eval()
    
    # Get predictions for all validation data
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(data.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Organize data by protein
    proteins = list(define_proteins().keys())
    colors = {'Trypsin': '#3274A1', 'Elastase': '#45a884'}

    fig, axes = plt.subplots(2, 2, figsize=(3,3), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, protein in enumerate(proteins):
        ax = axes[idx]
        
        # Get indices for this protein from validation set
        protein_indices = [i for i, p in enumerate(val_proteins) if p == protein]
        
        if protein_indices:
            protein_probs = all_probs[protein_indices]
            protein_labels = all_labels[protein_indices]
            
            trypsin_probs = []
            elastase_probs = []
            
            for i, label in enumerate(protein_labels):
                if label == 0:  # Trypsin
                    trypsin_probs.append(protein_probs[i, 1])
                elif label == 1:  # Elastase  
                    elastase_probs.append(protein_probs[i, 1])
            
            # Create DataFrame for this protein
            df = pd.DataFrame({
                'Enzyme': ['Trypsin'] * len(trypsin_probs) + ['Elastase'] * len(elastase_probs),
                'Elastase Probability': trypsin_probs + elastase_probs
            })
            
            # Plot KDE for this protein
            if len(df) > 0:
                sns.kdeplot(
                    data=df, x='Elastase Probability', hue='Enzyme',
                    palette=colors, fill=True, alpha=0.5,
                    common_norm=False, ax=ax
                )
            
            ax.set_title(f'{protein}')
            ax.set_xlabel('Elastase prob.')
            ax.set_ylabel('Density')
            
            # Remove legend from individual subplots
            if ax.get_legend():
                ax.get_legend().remove()
    
    # Add a single legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    plt.savefig('results/figures/enzyme_distributions.png', dpi=300, bbox_inches='tight')
 

def save_data_split(train_dataset, val_dataset, train_proteins, val_proteins, train_enzymes_split, val_enzymes_split, filename="results/outputs/data_split.pkl"):
    """Save train/validation split data."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset, 
        'train_proteins': train_proteins,
        'val_proteins': val_proteins,
        'train_enzymes_split': train_enzymes_split,
        'val_enzymes_split': val_enzymes_split
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data split saved to {filename}")


def load_data_split(filename="results/outputs/data_split.pkl"):
    """Load train/validation split data."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data split loaded from {filename}")
        return data['train_dataset'], data['val_dataset'], data['train_proteins'], data['val_proteins'], data['train_enzymes_split'], data['val_enzymes_split']
    else:
        print(f"No saved data split found at {filename}")
        return None


def save_model_results(model, val_dataset, val_proteins, val_enzymes_split, device, roc_auc, pr_auc, filename="results/outputs/model_results.pkl"):
    """Save model and prediction results."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Get model predictions
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(data.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    data = {
        'model_state_dict': model.state_dict(),
        'predictions': all_probs,
        'true_labels': all_labels,
        'proteins': val_proteins,
        'enzymes': val_enzymes_split,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Model results saved to {filename}")


def load_model_results(filename="results/outputs/model_results.pkl"):
    """Load model and prediction results."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Model results loaded from {filename}")
        return data
    else:
        print(f"No saved model results found at {filename}")
        return None


def plot_roc_from_loaded_data(model_results):
    """Plot ROC curves from loaded model results."""
    predictions = model_results['predictions']
    true_labels = model_results['true_labels']
    proteins = model_results['proteins']
    
    # Calculate per-protein ROC curves
    proteins_list = list(define_proteins().keys())
    protein_colors = {'ACTB': '#3274A1', 'HBB': "#4AB45A", 'THRB': '#45a884', 'APOA1': "#6b50ce"}

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    curves_plotted = 0
    
    print("\nValidation set distribution by protein:")
    # Plot per-protein ROC curves
    for protein in proteins_list:
        protein_indices = [i for i, p in enumerate(proteins) if p == protein]
        if len(protein_indices) > 0:
            protein_probs = predictions[protein_indices, 1]  # elastase probabilities
            protein_labels = true_labels[protein_indices]
            
            n_trypsin = np.sum(protein_labels == 0)
            n_elastase = np.sum(protein_labels == 1)
            print(f"  {protein}: {len(protein_labels)} samples (Trypsin: {n_trypsin}, Elastase: {n_elastase})")
            
            fpr, tpr, _ = roc_curve(protein_labels, protein_probs)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, 
                    color=protein_colors[protein], linewidth=2,
                    label=f'{protein} (AUC={roc_auc:.3f})')
            curves_plotted += 1

    # Add overall ROC curve (all proteins combined)
    all_probs_combined = predictions[:, 1]
    fpr_overall, tpr_overall, _ = roc_curve(true_labels, all_probs_combined)
    auc_overall = auc(fpr_overall, tpr_overall)
    
    ax.plot(fpr_overall, tpr_overall, 
           color='black', 
           linewidth=2,
           linestyle='--',
           label=f'Overall (AUC={auc_overall:.3f})')
    curves_plotted += 1

    ax.set_xlim([-0.05, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.savefig('results/figures/multi_protein_roc.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_kde_from_loaded_data(model_results):
    """Plot KDE distributions from loaded model results."""
    predictions = model_results['predictions']
    true_labels = model_results['true_labels']
    proteins = model_results['proteins']
    
    proteins_list = list(define_proteins().keys())
    colors = {'Trypsin': '#3274A1', 'Elastase': '#45a884'}

    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    axes = axes.flatten()
    
    for idx, protein in enumerate(proteins_list):
        ax = axes[idx]
        
        # Get indices for this protein
        protein_indices = [i for i, p in enumerate(proteins) if p == protein]
        
        if protein_indices:
            protein_probs = predictions[protein_indices]
            protein_labels = true_labels[protein_indices]
            
            trypsin_probs = []
            elastase_probs = []
            
            for i, label in enumerate(protein_labels):
                if label == 0:  # Trypsin
                    trypsin_probs.append(protein_probs[i, 1])
                elif label == 1:  # Elastase  
                    elastase_probs.append(protein_probs[i, 1])
            
            # Create DataFrame for this protein
            df = pd.DataFrame({
                'Enzyme': ['Trypsin'] * len(trypsin_probs) + ['Elastase'] * len(elastase_probs),
                'Elastase Probability': trypsin_probs + elastase_probs
            })
            
            # Plot KDE for this protein
            if len(df) > 0:
                sns.kdeplot(
                    data=df, x='Elastase Probability', hue='Enzyme',
                    palette=colors, fill=True, alpha=0.5,
                    common_norm=False, ax=ax
                )
            
            ax.set_title(f'{protein}')
            ax.set_xlabel('Elastase prob.')
            ax.set_ylabel('Density')
            
            # Remove legend from individual subplots
            if ax.get_legend():
                ax.get_legend().remove()
    
    # Add a single legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    
    sns.despine()
    plt.tight_layout()
    
    plt.savefig('results/figures/enzyme_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple flags - set these to control what to run
    LOAD_DATA = True   # Set to True to load existing data split
    LOAD_MODEL = False  # Set to True to load existing model results and just plot
    
    # Try to load existing data split
    if LOAD_DATA:
        data_split = load_data_split()
        if data_split is not None:
            train_dataset, val_dataset, train_proteins, val_proteins, train_enzymes_split, val_enzymes_split = data_split
        else:
            LOAD_DATA = False  # Fall back to generating new data
    
    # Generate data if not loaded
    if not LOAD_DATA:
        print("Generating synthetic datasets for all proteins...")
        proteins = define_proteins()
        train_enzymes = define_train_enzymes()
        
        all_datasets = []
        all_labels = []
        all_enzyme_types = []
        all_protein_names = []
        
        for protein_name, sequence in proteins.items():
            print(f"Processing {protein_name}...")
            dataset, labels, enzyme_types, protein_names, _ = generate_dataset_for_training(
                train_enzymes, n_samples=500, sequence=sequence, protein_name=protein_name  # Change back to 500 for full run
            )
            all_datasets.extend(dataset)
            all_labels.extend(labels)
            all_enzyme_types.extend(enzyme_types)
            all_protein_names.extend(protein_names)

        print(f"\nGenerated {len(all_datasets)} total samples")
        print(f"Class distribution: {np.bincount(np.array(all_labels))}")
        
        # Split data
        train_dataset, val_dataset, train_labels, val_labels, train_proteins, val_proteins, train_enzymes_split, val_enzymes_split = train_test_split(
            all_datasets, all_labels, all_protein_names, all_enzyme_types, 
            test_size=0.2, random_state=42, stratify=all_labels
        )
        
        print(f"\nTraining set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Feature dimensionality: {train_dataset[0].x.shape[1]}")
        
        # Save the data split
        save_data_split(train_dataset, val_dataset, train_proteins, val_proteins, train_enzymes_split, val_enzymes_split)
    
    # Try to load existing model results
    if LOAD_MODEL:
        model_results = load_model_results()
        if model_results is not None:
            print("\nLoaded model results - generating plots...")
            # Extract data for plotting
            val_proteins = model_results['proteins']
            val_enzymes_split = model_results['enzymes']
            print(f"ROC AUC: {model_results['roc_auc']:.4f}")
            print(f"PR AUC: {model_results['pr_auc']:.4f}")
        else:
            LOAD_MODEL = False  # Fall back to training new model
    
    # Train model if not loaded
    if not LOAD_MODEL:
        print("\nTraining model...")
        model, roc_auc, pr_auc = train_and_evaluate(train_dataset, val_dataset, epochs=50)
        
        # Save model results
        save_model_results(model, val_dataset, val_proteins, val_enzymes_split, device, roc_auc, pr_auc)
    
    # Generate plots (regardless of whether data/model was loaded or trained)
    print("\nGenerating plots...")
    if not LOAD_MODEL:
        # Use fresh model for plotting
        print("Generating ROC curves...")
        plot_multi_protein_roc(model, val_dataset, val_proteins, val_enzymes_split, device)
        
        print("Generating KDE distributions...")
        plot_enzyme_distributions(model, val_dataset, device)
    else:
        # Use loaded data for plotting
        print("Generating ROC curves from loaded data...")
        plot_roc_from_loaded_data(model_results)
        
        print("Generating KDE distributions from loaded data...")
        plot_kde_from_loaded_data(model_results)

