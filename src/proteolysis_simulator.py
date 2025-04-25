import re
import numpy as np
import networkx as nx  # type: ignore
from scipy.stats import gamma  # type: ignore


class Enzyme:
    """
    Represents an enzyme that defines how endoproteolytic cleavage sites
    are determined via regex patterns.

    Example cleavage_rules: [("(.)(.)([RK])([^P])(.)(.)", 1)]
      This means: for each match of the given regex in the extended sequence
      'XsequenceX', add `amount=1` to the cleavage site near that match.

    Attributes:
    -----------
    cleavage_rules : list of (pattern, weight)
        Each pattern is a regex string; weight is how much that pattern contributes
        to the cleavage probability at each matched site.
    """

    def __init__(self, cleavage_rules):
        self.cleavage_rules = cleavage_rules

    def cleave_probabilities(self, sequence: str) -> np.ndarray:
        """
        Given a sequence, compute the probability of cleavage at each
        position (0..len(sequence)-1). If no cleavage site is found,
        we return a uniform distribution over all positions.

        Returns
        -------
        cleavage_probs : np.ndarray of shape (len(sequence),)
            The probabilities sum to 1 over the possible cut sites.
        """
        n = len(sequence)
        # Probability array for cutting between i and i+1
        # We'll store it in the same indexing as your original approach
        # so that cleavage_probs[i] means "cut occurs after i residues".
        cleavage_probs = np.zeros(n)

        # Extend sequence with dummy chars to handle boundary conditions
        extended_seq = "X" + sequence + "X"

        total_score = 0.0
        # For each pattern, find matches and increment cleavage_probs
        for pattern, weight in self.cleavage_rules:
            regex = re.compile(pattern)
            for match in regex.finditer(extended_seq):
                # your original offset logic was match.start() + 2
                cleavage_index = match.start() + 2
                if 0 <= cleavage_index < n:
                    cleavage_probs[cleavage_index] += weight
                    total_score += weight

        if total_score == 0:
            # fallback: if no cleavage rule matched, use uniform
            cleavage_probs[:] = 1.0
            total_score = float(n)

        return cleavage_probs / total_score


class ProteolysisSimulator:
    """
    Simulates proteolysis (peptide generation) via both endo- and exo-cleavage
    events. Uses a gamma-filter acceptance criterion to model mass-spec detection
    preference for certain peptide lengths.

    Attributes
    ----------
    min_length : int
        Minimum length for a newly generated fragment to be "accepted."
    length_params : str, optional
        Either "vitro" or "vivo"; controls gamma distribution parameters.
    random_seed : int, optional
        If set, ensures reproducible RNG.
    verbose : bool
        If True, prints progress during simulate_proteolysis.
    """

    def __init__(
        self, min_length=6, length_params="vitro", random_seed=None, verbose=True
    ):
        self.min_length = min_length
        self.length_params = length_params
        self.verbose = verbose

        # Handle RNG seeding
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize the gamma distribution used for acceptance
        self._init_gamma_dist()

        # Will store final peptides (P_Y) and number generated
        self.P_Y = {}
        self.n_succesful_cleaves = 0

    def _init_gamma_dist(self):
        """
        Prepare the gamma distribution for acceptance checks, based on
        in vitro or in vivo parameters.
        """
        self.in_vitro_params = (1.438, 6.776, 4.564)  # a  # loc  # scale
        self.in_vivo_params = (1.915, 6.463, 4.557)  # a  # loc  # scale

        if self.length_params == "vitro":
            a, loc, scale = self.in_vitro_params
        else:
            a, loc, scale = self.in_vivo_params

        self.gamma_dist = gamma(a=a, loc=loc, scale=scale)

        # Precompute a maximum PDF for acceptance ratio
        xs = np.linspace(1, 200, 200)
        pdf_vals = self.gamma_dist.pdf(xs)
        self.max_gamma_pdf = pdf_vals.max()

    def simulate_proteolysis(
        self,
        sequence: str,
        enzyme: Enzyme,
        n_succesful_cleaves=100,
        relative_protein_abundance=0.2,
        endo_probability=0.5,
        make_graph=True,
    ):
        """
        Perform a proteolysis simulation.

        Parameters
        ----------
        sequence : str
            The initial full-length protein sequence.
        enzyme : Enzyme
            Defines how endoproteolytic cleavage probabilities are computed.
        n_succesful_cleaves : int
            Target number of peptides to generate (i.e., accepted fragments).
        relative_protein_abundance : float
            Fraction of total peptides that should remain as the main protein.
            (Used to artificially keep the main protein abundant enough.)
        endo_probability : float
            Probability of performing an endo-cleavage event vs. exo-cleavage
            on each iteration.
        make_graph : bool
            If True, track transitions in a NetworkX DiGraph.

        Returns
        -------
        P_Y : dict
            A dictionary of final peptide -> count.
        Gamma : nx.DiGraph
            If make_graph=True, returns the transition graph with 'weight' as
            the number of times a transition occurred. Also includes self-edges
            weighted by the final counts. If make_graph=False, returns None.
        """
        self.P_Y = {sequence: 1}  # Start with the full sequence
        self.n_succesful_cleaves = 0

        # Optionally track transitions in a graph
        if make_graph:
            self.Gamma = nx.DiGraph()
            self.Gamma.add_node(sequence, length=len(sequence))
        else:
            self.Gamma = None

        max_iterations = n_succesful_cleaves * 50  # safety limit
        iteration = 0

        # Pre-generate an array of “endo” vs “exo” decisions
        event_choices = np.random.choice(
            ["endo", "exo"],
            p=[endo_probability, 1 - endo_probability],
            size=max_iterations,
        )

        # Main loop
        while (self.n_succesful_cleaves < n_succesful_cleaves) and (
            iteration < max_iterations
        ):
            # Keep the main protein's abundance at a minimum fraction
            self._enforce_main_protein_abundance(sequence, relative_protein_abundance)

            choice = event_choices[iteration]
            if choice == "endo":
                self._do_endo_cleavage(enzyme)
            else:
                self._do_exo_cleavage()

            iteration += 1
            if self.verbose:
                print(
                    f"\r Succesful leaves performed {self.n_succesful_cleaves} / {n_succesful_cleaves} (iter={iteration})",
                    end="",
                    flush=True,
                )

        if self.verbose:
            total_counts = sum(self.P_Y.values())
            print(
                f"\nSimulation done. {len(self.P_Y)} unique peptides, "
                f"{total_counts} total counts. n_succesful_cleaves={self.n_succesful_cleaves}"
            )

        # Add self-edges to represent “no transition” events
        if make_graph and self.Gamma is not None:
            for pep in self.Gamma.nodes:
                self.Gamma.add_edge(pep, pep, weight=self.P_Y[pep])
            return self.P_Y, self.Gamma
        else:
            return self.P_Y, None

    def _enforce_main_protein_abundance(self, main_protein, rel_abundance):
        """
        Ensures the main protein is kept at or above a minimal threshold
        relative to total peptide count.
        """
        min_constant = 5
        total_count = sum(self.P_Y.values())
        # keep at least 'min_constant' or 'int(total_count * rel_abundance)', whichever is smaller.
        new_count = max(
            min_constant, min(min_constant, int(total_count * rel_abundance))
        )
        self.P_Y[main_protein] = new_count

    def _do_endo_cleavage(self, enzyme: Enzyme):
        """
        Picks a random sequence that has enough length and abundance,
        chooses one cut site using the enzyme cleavage probability,
        then chooses a second cut site weighted by gamma-dist from the first cut.
        Fragments are each tested for acceptance.
        """
        # Filter candidate peptides
        candidates = {
            seq: cnt for (seq, cnt) in self.P_Y.items() if len(seq) > 10 and cnt > 1
        }
        if not candidates:
            # fallback if none found
            seq_to_cut = list(self.P_Y.keys())[0]
            cleavage_probs = enzyme.cleave_probabilities(seq_to_cut)
        else:
            # We'll pick a sequence with weights ~ (# of cut sites * (count-1))
            seqs = []
            weights = []
            cleavage_cache = {}
            for seq, count in candidates.items():
                probs = enzyme.cleave_probabilities(seq)
                # #cut_sites ~ the sum of nonzero probabilities
                n_cut_sites = np.count_nonzero(probs > 0)
                w = n_cut_sites * (count - 1)
                if w > 0:
                    seqs.append(seq)
                    weights.append(w)
                cleavage_cache[seq] = probs

            if len(seqs) == 0:
                # fallback
                seq_to_cut = list(self.P_Y.keys())[0]
                cleavage_probs = enzyme.cleave_probabilities(seq_to_cut)
            else:
                weights = np.array(weights, dtype=float)
                seq_to_cut = np.random.choice(seqs, p=weights / weights.sum())
                cleavage_probs = cleavage_cache[seq_to_cut]

        if cleavage_probs.sum() == 0:
            cleavage_probs[:] = 1.0
        cleavage_probs /= cleavage_probs.sum()

        positions = np.arange(len(seq_to_cut))

        # First cut
        cut1 = np.random.choice(positions, p=cleavage_probs)

        # Second cut, weighted by gamma-dist from cut1
        dist_probs = cleavage_probs * 0.0  # re-init
        for i in positions:
            # Weighted by "enzyme cleavage prob" * "gamma pdf(|i-cut1|)"
            dist_probs[i] = cleavage_probs[i] * self.gamma_dist.pdf(abs(i - cut1))

        # If that all sums to 0, fallback to purely the gamma-dist from cut1
        if dist_probs.sum() == 0:
            for i in positions:
                dist_probs[i] = self.gamma_dist.pdf(abs(i - cut1))

        # Add small noise to avoid zeros
        dist_probs += 1e-8
        dist_probs /= dist_probs.sum()

        cut2 = np.random.choice(positions, p=dist_probs)

        i1, i2 = sorted([cut1, cut2])
        left_frag = seq_to_cut[0:i1]
        mid_frag = seq_to_cut[i1:i2]
        right_frag = seq_to_cut[i2:]

        self._maybe_accept_fragment(seq_to_cut, left_frag)
        self._maybe_accept_fragment(seq_to_cut, mid_frag)
        self._maybe_accept_fragment(seq_to_cut, right_frag)

    def _do_exo_cleavage(self):
        """
        Picks a random peptide (weighted by (count-1)) and removes one residue
        from either the N- or C-terminus with probability 0.5.
        Then tries acceptance on that new fragment.
        """
        peptides = []
        weights = []
        for seq, count in self.P_Y.items():
            w = count - 1
            if w > 0:
                peptides.append(seq)
                weights.append(w)

        if not peptides:
            # no exo cleavage possible if everything has count=1 or no peptides
            return

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        seq_to_chew = np.random.choice(peptides, p=weights)

        # 50-50 chance for N- vs. C-term removal
        if np.random.rand() < 0.5:
            new_seq = seq_to_chew[1:]
        else:
            new_seq = seq_to_chew[:-1]

        self._maybe_accept_fragment(seq_to_chew, new_seq)

    def _maybe_accept_fragment(self, old_seq, new_seq):
        if len(new_seq) < self.min_length:
            return  # skip: below min length

        # Acceptance ratio = pdf(length) / max_pdf
        length_pdf = self.gamma_dist.pdf(len(new_seq))
        ratio = length_pdf / (self.max_gamma_pdf + 1e-12)
        if np.random.rand() > ratio:
            return  # not accepted
        
        # Only decrement if the old_seq has 2 or more counts.
        if self.P_Y[old_seq] < 2:
            # Do NOT degrade it; we want to keep old_seq at minimum 1
            return

        # Otherwise, proceed:
        self.P_Y[old_seq] -= 1
        if new_seq not in self.P_Y:
            self.P_Y[new_seq] = 0
        self.P_Y[new_seq] += 1

        self.n_succesful_cleaves += 1

        # Update transition graph if needed
        if self.Gamma is not None:
            if not self.Gamma.has_node(new_seq):
                self.Gamma.add_node(new_seq, length=len(new_seq))
            if not self.Gamma.has_edge(old_seq, new_seq):
                self.Gamma.add_edge(old_seq, new_seq, weight=1)
            else:
                self.Gamma[old_seq][new_seq]["weight"] += 1


    def finalize_graph_as_probabilities(self):
        """
        Converts the current self.Gamma (if not None) from raw transition counts
        to probabilities. Returns a new DiGraph with edges labeled by 'prob'.

        self-edge probability is computed as:
          P(self-edge) = P_Y[node] / (sum_of_outgoing_counts + P_Y[node])
        Then the remainder is distributed among outgoing edges proportionally.

        Returns
        -------
        Gprob : nx.DiGraph
            The probability graph, or None if self.Gamma is None.
        """
        if self.Gamma is None:
            return None

        Gprob = nx.DiGraph()
        for node in self.Gamma.nodes():
            Gprob.add_node(node, **self.Gamma.nodes[node])

        for node in self.Gamma.nodes():
            out_edges = list(self.Gamma.out_edges(node, data=True))
            sum_out = sum(e[2]["weight"] for e in out_edges if e[0] != e[1])

            # The "self-edge" portion
            node_count = self.P_Y.get(node, 0)
            total_for_node = node_count + sum_out
            if total_for_node == 0:
                # e.g., if node is no longer present in P_Y or has no out edges,
                # we assign self-edge = 1
                Gprob.add_edge(node, node, v=1.0)
                continue

            self_edge_prob = node_count / total_for_node
            Gprob.add_edge(node, node, v=self_edge_prob)

            # Distribute the remaining probability among out edges
            for _, tgt, data in out_edges:
                if tgt == node:
                    # skip self-edge, already handled
                    continue
                raw_count = data["weight"]
                # (1 - self_edge_prob) * (raw_count / sum_out)
                p = (1 - self_edge_prob) * (raw_count / sum_out)
                Gprob.add_edge(node, tgt, v=p)

        return Gprob
