import numpy as np
import re
import networkx as nx
from .proteolysis_simulator import Enzyme


amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

np.seterr(invalid="ignore")

def regex_to_graph(
    V_Omega: str,
    V: list,
    regex_rules,
    exoprotease_threshold=1,
    connect_unconnected=True
):
    """
    Builds a directed graph of proteolysis.
    Each edge (seq_parent -> seq_child) is added if:
      - seq_child is a direct cleavage product of seq_parent
        according to one or more of the regex rules, OR
      - seq_child is obtained by exopeptidase (1 residue trimmed)
        from seq_parent (if exoprotease_threshold == 1).
    
    If `connect_unconnected` is True, then any peptide with
    no incoming edges (and not the parent) will get connected
    from any sequence that contains it as a substring (with a
    very small edge weight labeled 'leftover').

    Parameters
    ----------
    V_Omega : str
        The "parent" or largest sequence. (No incoming edges are desired.)
    V : list of str
        Collection of peptide sequences (including V_Omega).
    regex_rules : str or List[str]
        A single regex pattern or list of regex patterns representing
        possible endoprotease cleavage sites. For example,
        ["(.)(.)([KR])(.)(.)(.)"] for trypsin-like cuts (6 capturing groups).
        The cut is assumed between group 3 and group 4 (see `CUT_GROUP_INDEX`).
    exoprotease_threshold : int
        If == 1, we consider removing exactly 1 residue from either
        N or C terminus as a valid "exopeptidase" step.
    connect_unconnected : bool
        If True, peptides with no incoming edges (and not the parent)
        are connected from any sequence containing them
        with a small edge weight (marked "leftover").

    Returns
    -------
    nx.DiGraph
        Directed graph with edges that represent proteolytic relationships.
        Each edge has attributes:
          - "v": initial (unnormalized) weight
          - "cleavage_type": one of {"exoprotease", "endoprotease", "leftover"}
    """
    # Ensure V_Omega is in the set of nodes.
    if isinstance(regex_rules, Enzyme):
        regex_rules = [cleavage_rules[0] for cleavage_rules in regex_rules.cleavage_rules]
        
    if V_Omega not in V:
        V.append(V_Omega)
    V_set = set(V)

    graph = nx.DiGraph()
    graph.add_nodes_from(V)

    # Allow either a single regex or a list of regexes
    if isinstance(regex_rules, str):
        regex_rules = [regex_rules]

    compiled_rules = [re.compile(r) for r in regex_rules]

    # By default, we'll assume the cleavage is after capturing group #3.
    CUT_GROUP_INDEX = 3

    for seq_parent in V:
        possible_children = dict()  # child_seq -> cleavage_type

        # 1) Exoproteolysis step
        if exoprotease_threshold == 1 and len(seq_parent) > 1:
            left_trim = seq_parent[1:]
            right_trim = seq_parent[:-1]
            if left_trim in V_set:
                # Only add this child if we haven't added it yet (or decide to handle duplicates)
                if left_trim not in possible_children:
                    possible_children[left_trim] = "exoprotease"
            if right_trim in V_set:
                if right_trim not in possible_children:
                    possible_children[right_trim] = "exoprotease"

        # 2) Endoprotease cleavage step using each compiled regex
        for rule in compiled_rules:
            for match_obj in rule.finditer(seq_parent):
                # We assume cleavage is "after group #3" => cut position
                # = match_obj.end(CUT_GROUP_INDEX)
                cut = match_obj.end(CUT_GROUP_INDEX)

                # Build the left and right fragments
                left_frag = seq_parent[:cut]
                right_frag = seq_parent[cut:]

                # If either fragment is in V, it's a valid child
                if left_frag in V_set and left_frag != seq_parent:
                    # Only add if not already present (or unify if you want multiple reasons)
                    if left_frag not in possible_children:
                        possible_children[left_frag] = "endoprotease"

                if right_frag in V_set and right_frag != seq_parent:
                    if right_frag not in possible_children:
                        possible_children[right_frag] = "endoprotease"

        # Add edges for all possible children, marking the cleavage type
        for child, cleavage_type in possible_children.items():
            # We'll attach the cleavage_type attribute here. 
            graph.add_edge(seq_parent, child, v=1.0, cleavage_type=cleavage_type)

    # (Optional) connect unconnected nodes to any sequence containing them with a small weight
    if connect_unconnected:
        for node in graph.nodes():
            if node == V_Omega:
                # The main parent can remain with no incoming edges
                continue

            if graph.in_degree(node) == 0:
                # connect from any parent that contains `node` as a substring
                for pp in V:
                    if pp == node:
                        continue
                    if node in pp:
                        # Add a tiny weight so it doesn't overshadow real cleavages
                        # Mark these edges as "leftover"
                        graph.add_edge(pp, node, v=1e-5, cleavage_type="leftover")

    # Finally, normalize out-edge weights so that sum(out-edges) = (1 - self_prob)
    # (i.e., we imagine there's a 0.1 self-loop probability not explicitly drawn)
    self_prob = 0.1
    for node in graph.nodes():
        out_edges = list(graph.out_edges(node, data=True))
        if not out_edges:
            continue  # no children, skip

        total_v = sum(data["v"] for _, _, data in out_edges)
        if total_v == 0:
            continue  # avoid division by zero if something is off

        for _, child, data in out_edges:
            data["v"] = ((1 - self_prob) * data["v"]) / total_v

    return graph



def probabilities_to_flows(G, w_dict, root):
    """
    Given:
      - G: a NetworkX DiGraph (assumed DAG).
      - w_dict: dict {(u,v): prob} giving the probability of transitioning from u->v.
      - root: node with total inflow = 1 at the start.
      
    Returns:
      A dict {(u,v): flow_uv} of the flow along each edge, computed by
      a forward pass in topological order.
      
    Note:
      If w_dict does not have an entry for (u,v), we treat w(u->v) as 0.
      The sum of outflow from node u is inflow(u)*sum_{v in children(u)} w(u->v).
      The "leftover" inflow is absorbed at u.
    """
    # 1) Topological order
    topo_nodes = list(nx.topological_sort(G))
    
    # 2) Initialize inflows
    p_in = {n: 0.0 for n in G.nodes()}
    p_in[root] = 1.0
    
    # 3) We'll build the flow dictionary
    flow_dict = {}
    for (u,v) in G.edges():
        flow_dict[(u,v)] = 0.0  # initialize to 0
    
    # 4) Forward pass
    for u in topo_nodes:
        inflow_u = p_in[u]
        # For each child v of u, the flow is inflow_u * w(u->v)
        for v in G.successors(u):
            w_uv = w_dict.get((u,v), 0.0)  # 0 if missing
            flow_uv = inflow_u * w_uv
            flow_dict[(u,v)] = flow_uv
            p_in[v] += flow_uv
    
    return flow_dict



# Example usage:
if __name__ == "__main__":
    # A small DAG
    G = nx.DiGraph()
    G.add_nodes_from(["Omega","A","B","C"])
    G.add_edge("Omega","A")
    G.add_edge("Omega","B")
    G.add_edge("B","C")

    root = "Omega"

    # Suppose these are the final probabilities from a GD or CD approach:
    w_est = {
        ("Omega","A"): 0.4,
        ("Omega","B"): 0.6,
        ("B","C"): 0.8
        # No entry for (A-> something), so we treat that as 0 => absorption at A
    }

    # Convert to flow
    flow_est = probabilities_to_flows(G, w_est, root)
    print("Flow dictionary:")
    for e, val in flow_est.items():
        print(f"  {e}: {val:.4f}")
    
    # If you want, you can check absorption at each node as well:
    #    absorption[u] = p_in[u]*(1 - sum_{v in children(u)} w(u->v))
    # We'll do that quickly:
    p_in = {}
    for n in G.nodes():
        p_in[n] = 0.0
    p_in[root] = 1.0

    for u in nx.topological_sort(G):
        # Distribute inflow to children
        for v in G.successors(u):
            w_uv = w_est.get((u,v), 0.0)
            p_in[v] += p_in[u]*w_uv

    absorption = {}
    for u in G.nodes():
        out_sum = 0.0
        for v in G.successors(u):
            out_sum += w_est.get((u,v), 0.0)
        alpha_u = max(0.0, 1 - out_sum)
        absorption[u] = p_in[u]*alpha_u
    
    print("\nAbsorption at each node:")
    for n in G.nodes():
        print(f"  {n}: {absorption[n]:.4f}")
