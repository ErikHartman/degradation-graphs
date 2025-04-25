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

from proteolysis_simulator import Enzyme, ProteolysisSimulator

actb = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF"

sns.set_context("paper")


def define_train_enzymes():
    return {
        "trypsin": Enzyme([("(.)(.)([R|K])([^P])(.)(.)", 1)]),
        "elastase": Enzyme([("(.)(.)([V|I|A|S|L|G])(.)(.)(.)", 1)]),
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


def generate_dataset_for_training(enzymes, n_samples=100, sequence=actb):

    ps = ProteolysisSimulator(verbose=False)
    
    enzyme_indices = {name: i for i, name in enumerate(enzymes.keys())}

    dataset = []
    labels = []
    enzyme_types = []
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
            raw_abundance_list.append((abundances.numpy(), class_idx))

    return dataset, labels, enzyme_types, raw_abundance_list



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
    
    precision, recall, _ = precision_recall_curve(labels, elastase_probs)
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
    plt.savefig('roc_pr_curves.png', 
                dpi=300, bbox_inches='tight')

    
    print(f"Final results:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    return model, roc_auc, pr_auc


def plot_enzyme_distributions(model, train_dataset, device):

    model.eval()


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    trypsin_probs = []
    elastase_probs = []


    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out).cpu().numpy()


            for i, label in enumerate(data.y.cpu().numpy()):
                if label == 0:  # Trypsin
                    trypsin_probs.append(probs[i, 1])  
                elif label == 1:  # elastase
                    elastase_probs.append(probs[i, 1])


    df = pd.DataFrame({
        'Enzyme': ['Trypsin'] * len(trypsin_probs) +
                  ['Elastase'] * len(elastase_probs),
        'Elastase Probability': trypsin_probs + elastase_probs
    })

    colors = {
        'Trypsin': '#3274A1',
        'Elastase': '#45a884',
        'Mixed': '#f5b342'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 1]})

    # Plot KDE
    g = sns.kdeplot(
        data=df, x='Elastase Probability', hue='Enzyme',
        palette=colors, fill=True, alpha=0.5, linewidth=2,
        common_norm=False, ax=ax1
    )

    g.legend().remove()

    ax1.set_xlabel('Elastase Probability')
    ax1.set_ylabel('Density')
    
    sns.boxplot(
        data=df, x='Enzyme', y='Elastase Probability', 
        palette=colors, width=0.6, ax=ax2
    )
    
    sns.stripplot(
        data=df, x='Enzyme', y='Elastase Probability',
        color='black', size=3, alpha=0.4, ax=ax2,
        jitter=True
    )
    
    ax2.set_xlabel('Enzyme')
    ax2.set_ylabel('elastase Probability')
    
    sns.despine()
    plt.tight_layout()
    
    plt.savefig('enzyme_distributions.png', 
                dpi=300, bbox_inches='tight')

    return {
        'Trypsin': np.mean(trypsin_probs),
        'Elastase': np.mean(elastase_probs),
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    print("Generating synthetic dataset...")
    train_enzymes = define_train_enzymes()
    dataset, labels, enzyme_types, _ = generate_dataset_for_training(
        train_enzymes, n_samples=50, sequence=actb
    )

    print(f"Generated {len(dataset)} samples")
    print(f"Class distribution: {np.bincount(np.array(labels))}")
    
    train_dataset, val_dataset, train_labels, val_labels = train_test_split(
        dataset, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Feature dimensionality: {train_dataset[0].x.shape[1]}")

    best_model, roc_auc, pr_auc = train_and_evaluate(train_dataset, val_dataset, epochs=10)
    

    mean_probs = plot_enzyme_distributions(best_model, val_dataset, device)

    print("\nMean elastase probabilities by enzyme type:")
    for enzyme, prob in mean_probs.items():
        print(f"{enzyme}: {prob:.4f}")
    

