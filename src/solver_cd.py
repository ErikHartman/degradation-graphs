import networkx as nx
import torch
import torch.nn.functional as F

def run_coordinate_descent(
    G, Y, root,
    num_epochs=1000,
    lr=0.1,
    l1_strength=0.0,
    l2_strength=0.0,
    verbose=True
):
    """
    Coordinate Descent for a DAG absorption problem, storing loss & *edge-weight* history.
    
    We do a block update on theta[j], one node j at a time in topological order,
    for the specified number of epochs.

    :param G:           networkx.DiGraph (DAG)
    :param Y:           dict {node: float} target absorption distribution, sum=1
    :param root:        the root node
    :param num_epochs:  number of passes over the entire set of nodes
    :param lr:          step size for gradient updates
    :param l1_strength: L1 regularization coefficient
    :param l2_strength: L2 regularization coefficient
    :param verbose:     if True, prints periodic losses

    :return:
      theta_dict:   final learned logits (dict of node->Parameter)
      Yhat_dict:    final predicted absorption distribution
      loss_history: list of float, the loss after *each epoch*
      theta_history:list of dict, each dict is {(u,v): w_uv} storing the edge weights
                    at that point in the iteration.
    """

    # 1) Build adjacency info
    all_nodes = list(nx.topological_sort(G))
    node_to_children = {n: list(G.successors(n)) for n in all_nodes}
    
    # 2) Create parameter vectors (logits) for each node
    #    out_deg + 1 => child-edge slots + absorption
    theta_dict = {}
    for n in all_nodes:
        out_deg = len(node_to_children[n])
        param = torch.nn.Parameter(torch.zeros(out_deg + 1, dtype=torch.float))
        theta_dict[n] = param
    
    # 3) A forward pass to compute predicted absorption
    def forward_pass():
        # p_in: inflow to each node
        p_in = {n: torch.tensor(0.0, dtype=torch.float) for n in all_nodes}
        p_in[root] = torch.tensor(1.0, dtype=torch.float)
        
        for j in all_nodes:
            probs_j = F.softmax(theta_dict[j], dim=0)  # shape: (out_deg + 1,)
            w_children = probs_j[:-1]  # outgoing edges
            alpha_j    = probs_j[-1]   # absorption probability
            pj = p_in[j]
            
            # distribute to children
            for idx, c in enumerate(node_to_children[j]):
                p_in[c] += pj * w_children[idx]
        
        # absorption Yhat[j] = p_in[j] * alpha_j
        Yhat = {}
        for j in all_nodes:
            alpha_j = F.softmax(theta_dict[j], dim=0)[-1]  # recompute or reuse
            Yhat[j] = p_in[j] * alpha_j
        return Yhat
    
    # 4) Compute the overall loss = MSE(Yhat vs Y) + L1 + L2
    def compute_loss():
        Yhat = forward_pass()
        mse = torch.tensor(0.0, dtype=torch.float)
        
        # MSE part
        for n in all_nodes:
            target = Y.get(n, 0.0)
            diff = Yhat[n] - target
            mse = mse + diff*diff
        
        # L2 regularization
        if l2_strength > 0:
            l2_reg = torch.tensor(0.0, dtype=torch.float)
            for n in all_nodes:
                l2_reg = l2_reg + torch.sum(theta_dict[n] * theta_dict[n])
            mse = mse + l2_strength*l2_reg
        
        # L1 regularization
        if l1_strength > 0:
            l1_reg = torch.tensor(0.0, dtype=torch.float)
            for n in all_nodes:
                l1_reg = l1_reg + torch.sum(torch.abs(theta_dict[n]))
            mse = mse + l1_strength*l1_reg
        
        return mse
    
    # Helper: convert current logits -> {(u,v): prob_uv}
    def get_edge_weights():
        """
        Returns a dictionary {(u,v): w_uv} for all edges (u->v) in G,
        ignoring absorption. (The absorption is the last entry in each node's softmax.)
        """
        w_dict = {}
        for j in all_nodes:
            probs_j = F.softmax(theta_dict[j], dim=0)
            w_children = probs_j[:-1]  # the first 'out_deg' entries
            child_list = node_to_children[j]
            for idx, c in enumerate(child_list):
                w_dict[(j, c)] = float(w_children[idx])
        return w_dict
    
    # 5) Coordinate Descent main loop
    loss_history = []
    theta_history = []

    for epoch in range(num_epochs):
        # For each node, block update
        for j in all_nodes:
            # Freeze all
            for n in all_nodes:
                theta_dict[n].requires_grad = False
            
            # Unfreeze only j
            theta_dict[j].requires_grad = True
            
            # Zero out existing gradient
            if theta_dict[j].grad is not None:
                theta_dict[j].grad.zero_()
            
            # Compute loss & backprop
            loss_val = compute_loss()
            loss_val.backward()
            
            # Update logits for node j
            with torch.no_grad():
                theta_dict[j] -= lr * theta_dict[j].grad
        
        # End of this epoch: record the new loss & current edge weights
        current_loss = loss_val.item()
        loss_history.append(current_loss)

        # Build a snapshot of (u->v) => prob
        snapshot_w = get_edge_weights()
        theta_history.append(snapshot_w)

        # (Optional) print progress
        if verbose and (epoch+1) % max(1, (num_epochs//10)) == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, loss={current_loss:.6f}")
    
    # 6) Final check
    final_loss = compute_loss().item()
    Yhat_final = forward_pass()
    Yhat_dict = {n: Yhat_final[n].item() for n in all_nodes}

    return theta_dict, Yhat_dict, loss_history, theta_history


def main():
    # Small example usage
    G = nx.DiGraph()
    G.add_nodes_from(["Omega","A","B","C"])
    G.add_edge("Omega","A")
    G.add_edge("Omega","B")
    G.add_edge("B","C")

    root = "Omega"
    # Some random target distribution
    Y = {"Omega":0.1, "A":0.3, "B":0.4, "C":0.2}

    (
        final_theta,
        Yhat,
        loss_history,
        theta_history
    ) = run_coordinate_descent(
        G, Y, root,
        num_epochs=1000,
        lr=0.1,
        l1_strength=0.0,
        l2_strength=0.0,
        verbose=True
    )

    print("\nFinal predicted distribution:")
    for n, val in Yhat.items():
        print(f"  {n}: {val:.4f}")

    print("\nLoss history (last 10):")
    print(loss_history[-10:])

    print("\nFinal edge weights from the last snapshot:")
    print(theta_history[-1])  # e.g. {("Omega","A"): 0.45, ...}


if __name__ == "__main__":
    main()
