import networkx as nx
import torch
import torch.nn.functional as F

def run_gradient_descent(
    G, Y, root,
    num_epochs=1000,
    lr=0.1,
    l1_strength=0.0,
    l2_strength=0.0,
    verbose=True
):
    """
    Solve for node absorption distribution using a gradient descent approach.
    Each node j has (out_degree(j)+1) softmax logits: the last slot is absorption,
    the others are edge probabilities to each child.

    :param G: networkx DiGraph (DAG)
    :param Y: dict {node: float} target absorption distribution, sum=1
    :param root: the root node
    :param num_epochs: number of gradient descent steps
    :param lr: learning rate
    :param reg_strength: L2 regularization on the logits
    :param verbose: if True, prints periodic losses

    :return:
        (theta_dict, Yhat_dict, loss_history, theta_history)
          - theta_dict: {node: torch.nn.Parameter} containing final logits
          - Yhat_dict: {node: float} final predicted absorption
          - loss_history: list of float, one entry per epoch (the MSE+reg at that epoch)
          - theta_history: list of dict snapshots of the parameters.
            Each entry is {node: Tensor} storing a clone of the logits at that epoch.
    """

    # 1) Build a stable topological order
    all_nodes = list(nx.topological_sort(G))
    node_to_children = {n: list(G.successors(n)) for n in all_nodes}
    
    # 2) Create parameters
    theta_dict = {}
    for n in all_nodes:
        out_deg = len(node_to_children[n])
        # out_deg + 1 => child-edge slots + 1 absorption
        param = torch.nn.Parameter(torch.zeros(out_deg + 1, dtype=torch.float))
        theta_dict[n] = param
    
    # 3) We'll collect these parameters into an optimizer
    optimizer = torch.optim.Adam(theta_dict.values(), lr=lr)

    # 4) A helper to do a forward pass
    def forward_pass():
        # p_in[node]: inflow to node
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
            
            # distribute flow to children
            for idx, c in enumerate(children_j):
                p_in[c] = p_in[c] + pj * w_children[idx]
        
        # absorption
        Yhat = {}
        for j in all_nodes:
            Yhat[j] = p_in[j] * alpha[j]
        return Yhat
    
    # 5) Compute a loss
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
        
        return mse, Yhat
    
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

    # 6) Main training loop, now with histories
    loss_history = []
    theta_history = []

    for epoch in range(num_epochs):
        # zero out gradients
        optimizer.zero_grad()

        # forward & compute loss
        loss_val, Yhat = compute_loss()
        
        # backward
        loss_val.backward()
        
        # update
        optimizer.step()
        
        # Store the current loss
        cur_loss = loss_val.item()
        loss_history.append(cur_loss)

        # Store a snapshot of the parameters (logits)
        snapshot_w = get_edge_weights()
        theta_history.append(snapshot_w)

        # optionally print progress
        if verbose and (epoch+1) % max(1, (num_epochs//10)) == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, loss={cur_loss:.6f}")
    
    # 7) Final forward pass
    _, final_Yhat = compute_loss()
    final_Yhat = {n: final_Yhat[n].item() for n in all_nodes}
    
    return theta_dict, final_Yhat, loss_history, theta_history


def main():
    # Example usage
    G = nx.DiGraph()
    G.add_nodes_from(["Omega","A","B","C"])
    G.add_edge("Omega","A")
    G.add_edge("Omega","B")
    G.add_edge("B","C")
    
    root = "Omega"
    Y = {"Omega":0.1, "A":0.3, "B":0.4, "C":0.2}
    
    # run gradient descent
    (
        theta_dict,
        Yhat_dict,
        loss_history,
        theta_history
    ) = run_gradient_descent(
        G, Y, root,
        num_epochs=1000,
        lr=0.1,
        reg_strength=0.0,
        verbose=True
    )
    
    print("\nFinal predicted distribution:", Yhat_dict)
    print("Final loss:", loss_history[-1])
    print("\nFinal edge weights from the last snapshot:")
    print(theta_history[-1]) 
    print("We stored", len(loss_history), "loss entries and", len(theta_history), "theta snapshots.")


if __name__ == "__main__":
    main()
