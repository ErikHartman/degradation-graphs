import networkx as nx
import pulp

def run_lp(G, Y, root, solver_verbose=False):
    """
    Solve for edge probabilities in a DAG via the flow/LP approach.

    :param G: A networkx.DiGraph (assumed DAG).
    :param Y: dict {node: float} specifying target absorption distribution; sum(Y) = 1.
    :param root: The root node from which flow starts (inflow=1).
    :param solver_verbose: If True, prints solver messages.
    :return:
       (edge_probs, alpha):
         - edge_probs: dict {(u,v): probability of transitioning u->v}
         - alpha: dict {node: absorption_probability_at_node}
    :raises ValueError: if no feasible solution.
    """

    # 1) Create the LP problem
    prob = pulp.LpProblem("FlowFeasibility", pulp.LpMinimize)
    
    # 2) Flow variables for each edge
    F_vars = {}
    for (u, v) in G.edges():
        var_name = f"F_{u}_to_{v}"
        F_vars[(u, v)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpContinuous)
    
    # 3) Flow conservation constraints
    for j in G.nodes():
        outflow_terms = [F_vars[(j, c)] for c in G.successors(j)]
        inflow_terms = [F_vars[(p, j)] for p in G.predecessors(j)]
        
        if j == root:
            # sum of outflows + Y[j] = 1
            prob += (pulp.lpSum(outflow_terms) + Y.get(j, 0.0) == 1.0), f"FlowCons_root_{j}"
        else:
            # sum of outflows + Y[j] = sum of inflows
            prob += (pulp.lpSum(outflow_terms) + Y.get(j, 0.0) == pulp.lpSum(inflow_terms)), f"FlowCons_{j}"
    
    # 4) Objective: just find any feasible solution, so minimize 0
    prob.setObjective(pulp.lpSum([]))
    
    # 5) Solve
    msg_level = 1 if solver_verbose else 0
    solver = pulp.PULP_CBC_CMD(msg=msg_level)
    result_status = prob.solve(solver)
    
    if pulp.LpStatus[result_status] != 'Optimal':
        raise ValueError(f"No feasible solution found. LP status = {pulp.LpStatus[result_status]}")
    
    # 6) Extract flows, then compute probabilities
    #    p_in[j] = total inflow to j
    p_in = {}
    for j in G.nodes():
        if j == root:
            p_in[j] = 1.0
        else:
            inflow_val = sum(pulp.value(F_vars[(p, j)]) for p in G.predecessors(j))
            p_in[j] = inflow_val
    
    edge_probs = {}
    alpha = {}
    for j in G.nodes():
        # outflow
        outflow_sum = sum(pulp.value(F_vars[(j, c)]) for c in G.successors(j))
        pj = p_in[j]
        if pj > 1e-12:
            # absorption prob
            alpha_j = (Y.get(j, 0.0)) / pj
            alpha[j] = alpha_j
            
            # edge prob
            for c in G.successors(j):
                F_jc = pulp.value(F_vars[(j, c)])
                edge_probs[(j, c)] = F_jc / pj
        else:
            # if pj ~ 0, but Y[j]>0 => not truly feasible, but we should
            # never land here if solver said "Optimal". Edge case: Y[j]=0
            alpha[j] = 0.0
            for c in G.successors(j):
                edge_probs[(j, c)] = 0.0
    
    return edge_probs, alpha


def main():
    # Simple usage example
    # Build a small DAG
    G = nx.DiGraph()
    G.add_nodes_from(["Omega","A","B","C"])
    G.add_edge("Omega","A")
    G.add_edge("Omega","B")
    G.add_edge("B","C")
    
    root = "Omega"
    Y = {"Omega": 0.1, "A": 0.3, "B":0.4, "C":0.2}
    
    w, alpha = run_lp(G, Y, root, solver_verbose=False)
    print("Edge probabilities:", w)
    print("Absorption probabilities:", alpha)

if __name__ == "__main__":
    main()
