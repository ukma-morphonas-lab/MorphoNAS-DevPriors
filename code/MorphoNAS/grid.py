import hashlib
import math
import random
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import lil_matrix
import networkx as nx

from .weight_channel import get_weight_channel
from .composition_channel import get_composition_channel

class Grid:
    # Define constants at class level
    DIVISION_MORPHOGEN_INDEX = 0
    DIFFERENTIATION_MORPHOGEN_INDEX = 1
    AXON_GUIDANCE_MORPHOGEN_INDEX = 2
    AXON_CONNECT_MORPHOGEN_INDEX = 3
    WEIGHT_ADJUSTMENT_MORPHOGEN_INDEX = 4

    def __init__(self, genome):
        # Initialize random number generator with deterministic hash from genome bytes
        genome_bytes = genome.to_bytes()
        hash_value = int.from_bytes(hashlib.md5(genome_bytes).digest(), 'big')
        self.rng = random.Random(hash_value)
        # Separate, deterministic seed for the bio-inspired weight rules (weight_channel
        # dale/lognormal). Used OUTSIDE the growth RNG stream so weight randomness never
        # perturbs growth -> topology stays invariant (the hard gate). Not consumed by v1.1.
        self._weight_seed = hash_value & 0xFFFFFFFF
        
        self.size_x = genome.size_x
        self.size_y = genome.size_y
        self.diffusion_rate = genome.diffusion_rate
        self.num_morphogens = genome.num_morphogens
        self.division_threshold = genome.division_threshold
        self.cell_differentiation_threshold = genome.cell_differentiation_threshold
        self.axon_growth_threshold = genome.axon_growth_threshold
        self.max_axon_length = genome.max_axon_length
        self.axon_connect_threshold = genome.axon_connect_threshold
        self.self_connect_isolated_neurons_fraction = genome.self_connect_isolated_neurons_fraction
        self.progenitor_secretion_rates = genome.progenitor_secretion_rates
        self.neuron_secretion_rates = genome.neuron_secretion_rates
        self.inhibition_matrix = genome.inhibition_matrix
        self.diffusion_patterns = genome.diffusion_patterns
        self.max_growth_steps = genome.max_growth_steps
        
        # Add new genome parameters for weight adjustment
        self.weight_adjustment_target = genome.weight_adjustment_target
        self.weight_adjustment_rate = genome.weight_adjustment_rate
        
        # Initialize remaining attributes as before
        self._M = np.zeros((self.num_morphogens, self.size_x, self.size_y))
        self._division_morphogen = self._M[self.DIVISION_MORPHOGEN_INDEX]
        self._differentiation_morphogen = self._M[self.DIFFERENTIATION_MORPHOGEN_INDEX]
        self._axon_guidance_morphogen = self._M[self.AXON_GUIDANCE_MORPHOGEN_INDEX]
        self._axon_connect_morphogen = self._M[self.AXON_CONNECT_MORPHOGEN_INDEX if self.num_morphogens > 3 else self.AXON_GUIDANCE_MORPHOGEN_INDEX]
        self._weight_adjustment_morphogen = self._M[
            self.WEIGHT_ADJUSTMENT_MORPHOGEN_INDEX if self.num_morphogens > 4 else (
                self.AXON_CONNECT_MORPHOGEN_INDEX if self.num_morphogens == 4 else self.AXON_GUIDANCE_MORPHOGEN_INDEX
            )
        ]
        
        self.iteration = 0
        self._neurons = np.zeros((self.size_x, self.size_y), dtype=int)
        self._progenitors = np.zeros((self.size_x, self.size_y), dtype=int)
        self._cell_positions = {}
        self._axons = {}
        self._max_cell_id = 0
        self.neuron_connections = lil_matrix((self.size_x * self.size_y, self.size_x * self.size_y), dtype=float)
        self.listeners = []
    
    def get_graph(self):
        """Get the graph of the neural network with connection weights."""
        G = nx.DiGraph()
        
        # Add nodes for all neurons
        neuron_ids = self.get_neuron_ids()
        for cell_id in neuron_ids:
            G.add_node(cell_id)
        
        # Add edges from connectivity matrix with their weights
        source_indices, target_indices = self.neuron_connections.nonzero()
        weights = self.neuron_connections[source_indices, target_indices].toarray().flatten()
        
        for source, target, weight in zip(source_indices + 1, target_indices + 1, weights):
            if source in G.nodes() and target in G.nodes():
                G.add_edge(source, target, weight=float(weight))
        
        return G
    
    def add_cell(self, position, cell_type="progenitor"):
        """Add a new cell at the given position."""
        if (self._neurons[position] == 0 and self._progenitors[position] == 0):
            cell_id = self._max_cell_id + 1
            self._max_cell_id = cell_id
            
            if cell_type == "neuron":
                self._neurons[position] = cell_id
                self._axons[cell_id] = [position]  # Initialize axon for neurons
            else:  # progenitor
                self._progenitors[position] = cell_id
                
            self._cell_positions[cell_id] = position
    
    def _morphogen_diffusion_rate(self, i):
        """Per-morphogen diffusion mix-weight. Canonical (v1.1): the single shared genome
        scalar self.diffusion_rate for every morphogen (byte-identical). Turing (Step 0.5):
        the activator/inhibitor pair gets its own slow/fast rates (D_act << D_inh -- the
        differential-diffusion Turing requirement the verdict found absent); every other
        morphogen keeps the genome scalar. See notes/step0.5-turing-probe-design-2026-06.md."""
        cfg = get_weight_channel()
        if cfg.rd_mode == "turing":
            if i == cfg.turing_act_idx:
                return cfg.turing_D_act
            if i == cfg.turing_inh_idx:
                return cfg.turing_D_inh
        return self.diffusion_rate

    def diffuse(self):
        # Update each morphogen independently with its own kernel
        for i in range(self.num_morphogens):
            # Convolve the matrix with the morphogen's specific kernel
            diffused = convolve2d(self._M[i], self.diffusion_patterns[i], mode='same', boundary='wrap')

            # Update the matrix using the diffusion rate. Under rd_mode="turing" the A/H pair
            # uses per-morphogen rates (differential diffusion); under canonical every morphogen
            # uses the genome's single shared self.diffusion_rate -- byte-identical to v1.1.
            r = self._morphogen_diffusion_rate(i)
            self._M[i] = (1 - r) * self._M[i] + r * diffused

        self.iteration += 1

    def react_morphogens(self):
        """Step 0.5: the Gierer-Meinhardt activator-inhibitor reaction on the (A, H) morphogen
        pair (notes/step0.5-turing-probe-design-2026-06.md; the missing autocatalytic activator
        the RD-Turing verdict identified). Forward-Euler over one growth step, dt = turing_dt:

            dA = rho_a + rho * A^2 / H - mu_A * A     (basal + autocatalytic production - decay)
            dH = rho_h + rho * A^2     - mu_H * H     (basal + A-driven production    - decay)

        The basal rho_a is the standard Meinhardt term that prevents the A->0 absorbing state, so
        a pattern forms from a flat/uniform start (no external source needed). Paired with
        D_H >> D_A in diffuse(), this makes the homogeneous fixed point unstable to a finite-
        wavelength band -> a stable patterned (Turing) steady state. Replaces the pure
        multiplicative cross-damping for the A/H pair (inhibit_morphogens skips them under
        turing). The GM activator peaks well above the canonical morphogen cap (~5x the mean), so
        A/H are clamped only to a generous numerical ceiling (turing_ceiling), NOT the secretion
        cap of 1.0 -- that clamp provably flattens GM. Deterministic in the field (no RNG); only
        runs when rd_mode == 'turing'."""
        cfg = get_weight_channel()
        if cfg.rd_mode != "turing":
            return
        a, h = cfg.turing_act_idx, cfg.turing_inh_idx
        if a >= self.num_morphogens or h >= self.num_morphogens or a == h:
            return
        A = self._M[a]
        H = np.maximum(self._M[h], cfg.turing_H_floor)   # guard A^2/H against ~0 inhibitor
        A2 = A * A
        dA = cfg.turing_rho_a + cfg.turing_rho * A2 / H - cfg.turing_mu_act * A
        dH = cfg.turing_rho_h + cfg.turing_rho * A2 - cfg.turing_mu_inh * H
        ceil = cfg.turing_ceiling
        self._M[a] = np.clip(A + cfg.turing_dt * dA, 0.0, ceil)
        self._M[h] = np.clip(self._M[h] + cfg.turing_dt * dH, 0.0, ceil)

    def inhibit_morphogens(self):
        """Apply morphogen inhibition effects based on the inhibition matrix.

        Under rd_mode == 'turing' the activator/inhibitor pair (turing_act_idx, turing_inh_idx)
        is fully DECOUPLED from the multiplicative cross-inhibition: any pair (i, j) with either
        index in the A/H set is skipped. Their dynamics are governed solely by the Gierer-
        Meinhardt react_morphogens() step + differential diffusion. (The GM activator grows well
        above 1, so leaving it in the (1 - inh*M) damping would drive the other morphogens
        negative and the field would diverge -- the A/H pair must not participate in the
        canonical inhibition at all.) Under canonical (default) the skip-set is empty -> every
        pair is damped exactly as in v1.1 (byte-identical)."""
        cfg = get_weight_channel()
        if cfg.rd_mode == "turing":
            ah = {cfg.turing_act_idx, cfg.turing_inh_idx}
        else:
            ah = set()
        # Iterate over each morphogen pair to apply inhibition
        for i in range(self.num_morphogens):
            for j in range(self.num_morphogens):
                if i != j and self.inhibition_matrix[i, j] > 0:
                    # GM reaction replaces the canonical damping for the A/H pair (turing only):
                    # skip any pair touching A or H, so the GM pair neither inhibits nor is
                    # inhibited by the other morphogens.
                    if i in ah or j in ah:
                        continue
                    # Reduce morphogen `i` by the inhibition effect of morphogen `j`
                    self._M[i] *= (1 - self.inhibition_matrix[i, j] * self._M[j])

    def get_neighbors(self, x, y):
        """Get all neighboring positions around the given coordinates."""
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        
        # Update wrapping to use separate dimensions
        return [((x + dx) % self.size_x, (y + dy) % self.size_y) for dx, dy in offsets]

    def divide_cells(self):
        """Optimized vectorized cell division using matrix operations."""
        occupied = (self._progenitors != 0) | (self._neurons != 0)
        empty = ~occupied

        # Identify eligible progenitor cells for division
        eligible_mask = (self._division_morphogen > self.division_threshold) & (self._progenitors != 0)
        eligible_positions = np.argwhere(eligible_mask)

        if eligible_positions.size == 0:
            return

        # Define neighbor offsets
        offsets = np.array([
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ])
        
        # Precompute all neighbor positions and wrap around grid edges
        neighbors = (eligible_positions[:, None, :] + offsets) % [self.size_x, self.size_y]

        # Extract morphogen concentrations and empty status for neighbors
        neighbor_morphogens = self._division_morphogen[
            neighbors[..., 0], neighbors[..., 1]
        ]
        neighbor_empty = empty[neighbors[..., 0], neighbors[..., 1]]

        # Mask out non-empty positions
        neighbor_morphogens[~neighbor_empty] = -np.inf

        # Find the best neighbor for each eligible progenitor
        best_indices = np.argmax(neighbor_morphogens, axis=1)
        best_neighbors = neighbors[np.arange(len(eligible_positions)), best_indices]

        # Filter positions where a valid neighbor was found
        valid_divisions = neighbor_morphogens.max(axis=1) > -np.inf
        new_cells = best_neighbors[valid_divisions]

        # Update the grid and add new progenitor cells
        for new_x, new_y in new_cells:
            self.add_cell((new_x, new_y), cell_type="progenitor")
            empty[new_x, new_y] = False


    def _composition_secretion_fields(self, comp):
        """Build the per-cell, region-dependent secretion-rate fields for Hox composition
        (notes/structural-composition-design-2026-06.md). Returns (prog_field, neur_field),
        each of shape (num_morphogens, size_x, size_y): the secretion rate at cell (x, y)
        is the genome's secretion vector under the motif selected for the region containing x.

        Region map: K vertical stripes by x, region(x) = min(K-1, floor(x*K/size_x)) -- a
        pure function of position (no extra morphogen field, no growth RNG -> determinism +
        topology-invariance-when-off both hold). Motif library: motif 0 is the genome's OWN
        secretion vectors (encoded once); motif m cyclically shifts the morphogen channels by m
        (np.roll) -- a fixed, deterministic re-parameterization ("reuse a generic motif"). The
        per-region selector sel[r] in [0, M) indexes the library.

        For K=1 with the identity selector (sel=[0]), region(x)=0 everywhere and motif 0 = the
        genome's own program, so prog_field[i] is uniformly progenitor_secretion_rates[i] and
        neur_field[i] is uniformly neuron_secretion_rates[i] -- byte-identical to v1.1."""
        K = max(1, int(comp.K))
        M = max(1, int(comp.M))
        sel = comp.resolved_selector()   # length K, each in [0, M)
        m = self.num_morphogens

        # motif library: (M, num_morphogens) for progenitor and neuron secretion vectors.
        # motif idx 0 == the genome's OWN vectors (so K=1 / identity selector == v1.1).
        if comp.motif == "gain":
            # ROLE-PRESERVING: scale the genome's program by a per-motif scalar gain.
            # gains[0] must be 1.0 (motif 0 == genome's own program -> K=1 byte-identity).
            gains = comp.gains if comp.gains else (1.0,)
            g = [float(gains[mm]) if mm < len(gains) else 1.0 for mm in range(M)]
            prog_lib = np.stack([g[mm] * self.progenitor_secretion_rates for mm in range(M)])
            neur_lib = np.stack([g[mm] * self.neuron_secretion_rates for mm in range(M)])
        else:  # "roll": channel-shift (design-record / ablation; scrambles channel roles)
            prog_lib = np.stack([np.roll(self.progenitor_secretion_rates, mm) for mm in range(M)])
            neur_lib = np.stack([np.roll(self.neuron_secretion_rates, mm) for mm in range(M)])

        # per-x region index, then per-x selected-motif index
        xs = np.arange(self.size_x)
        region_of_x = np.minimum(K - 1, (xs * K) // self.size_x)          # (size_x,)
        motif_of_x = np.array([sel[r] for r in region_of_x], dtype=int)   # (size_x,)

        # per-x secretion vectors: (size_x, num_morphogens) -> broadcast over y
        prog_x = prog_lib[motif_of_x]   # (size_x, num_morphogens)
        neur_x = neur_lib[motif_of_x]   # (size_x, num_morphogens)

        # to (num_morphogens, size_x, size_y): rate depends on x only, constant along y
        prog_field = np.repeat(prog_x.T[:, :, None], self.size_y, axis=2)
        neur_field = np.repeat(neur_x.T[:, :, None], self.size_y, axis=2)
        return prog_field, neur_field

    def secrete_morphogens(self):
        """Vectorized secretion of morphogens."""
        if not self._progenitors.any() and not self._neurons.any():
            return

        # Boolean masks for progenitors and neurons
        progenitor_mask = (self._progenitors != 0)
        neuron_mask = (self._neurons != 0)

        # M9c (notes/mn-v2-sweep-design.md sec. 2.5): the secretion ceiling drives the
        # flat-plateau weight degeneracy. Default cap=1.0 keeps this byte-identical to v1.1;
        # only rd_antiplateau relaxes it (an INTENDED growth change -- NOT topology-invariant).
        cfg = get_weight_channel()
        cap = cfg.rd_secretion_cap if cfg.rd_antiplateau else 1.0

        # Step 0.5: under rd_mode == "turing" the activator/inhibitor pair is GM-governed (its
        # source is the GM basal production rho_a/rho_h, NOT cell secretion), and its activator
        # peaks above the cap, so it must be excluded from the secretion add+clamp here -- the
        # cap=1.0 clamp provably flattens the GM pattern. Every other morphogen secretes exactly
        # as in v1.1. Under canonical (default) this skip-set is empty -> byte-identical to v1.1.
        skip = ({cfg.turing_act_idx, cfg.turing_inh_idx}
                if (cfg.rd_mode == "turing" and cfg.turing_skip_secretion) else set())

        # Genome-gated Hox composition (notes/structural-composition-design-2026-06.md): when
        # ON, the secretion PROGRAM is region-dependent with reuse -- the field is partitioned
        # into K vertical stripes by x, and region r secretes under the reused motif sel[r] (a
        # cyclic morphogen-channel shift of the genome's own secretion vectors). Off (default)
        # AND the K=1 / identity-selector case both reduce to the global program below, so this
        # is byte-identical to v1.1 (the repro gate + prior-preservation-by-construction gate).
        comp = get_composition_channel()
        prog_field = neur_field = None
        if comp.compose:
            prog_field, neur_field = self._composition_secretion_fields(comp)

        # Vectorized secretion for each morphogen
        for i in range(self.num_morphogens):
            if i in skip:
                continue
            if prog_field is not None:
                # per-cell, region-dependent secretion rates (Hox motif reuse)
                secretion = (
                    progenitor_mask * prog_field[i] +
                    neuron_mask * neur_field[i]
                )
            else:
                secretion = (
                    progenitor_mask * self.progenitor_secretion_rates[i] +
                    neuron_mask * self.neuron_secretion_rates[i]
                )
            self._M[i] = np.minimum(cap, self._M[i] + secretion)
    
    def differentiate_cells(self):
        """Vectorized cell differentiation based on the differentiation morphogen concentration."""
        if not self._progenitors.any():
            return

        # Create a mask for cells that meet the differentiation criteria
        differentiation_mask = (
            (self._differentiation_morphogen > self.cell_differentiation_threshold) & 
            (self._progenitors != 0)
        )

        # Identify positions where differentiation occurs
        differentiation_positions = np.argwhere(differentiation_mask)

        for x, y in differentiation_positions:
            cell_id = self._progenitors[x, y]
            self._progenitors[x, y] = 0
            self._neurons[x, y] = cell_id

            # Update the cell data and initialize axon
            self._axons[cell_id] = [self._cell_positions[cell_id]]
            
    def initialize_weight(self, source_id, target_id):
        """Initialize weight between two neurons based on morphogen concentration and distance."""
        x, y = self._cell_positions[target_id]
        source_x, source_y = self._cell_positions[source_id]
        
        # Calculate distance-based scaling
        distance = np.linalg.norm(np.array([source_x, source_y]) - np.array([x, y]))
        weight = max(0.01, self._weight_adjustment_morphogen[x, y] / (1 + distance))
        
        # Update weight in the connections matrix
        self.neuron_connections[source_id - 1, target_id - 1] = weight

    def _channel_weight(self, source_id, target_id, cfg, mode):
        """Pure per-edge weight that `mode` assigns to edge (source->target), or None
        when the target's morphogen neighborhood sums to 0 (no competitive update -- the
        v1.1 guard, leave the existing weight as-is).

        Reads morphogen fields, cell positions, and self._weight_seed ONLY -- never the
        growth RNG -- so it is deterministic and topology-preserving for the signed
        readout modes (notes/mn-v2-sweep-design.md sec. 2). update_weight() calls it with
        cfg.mode; temporal_weight_update() calls it with cfg.base_mode. The "saturating"
        branch is the literal original expression, so the default path is byte-for-byte
        v1.1 (the repro gate)."""
        x, y = self._cell_positions[target_id]
        neighbors = self.get_neighbors(x, y)

        # Compute total morphogen concentration in the neighborhood
        total_morphogen = sum(self._weight_adjustment_morphogen[nx, ny] for nx, ny in neighbors)

        if total_morphogen <= 0:
            return None

        c_t = self._weight_adjustment_morphogen[x, y]
        if mode == "saturating":
            return max(0.01, c_t / total_morphogen)                        # ORIGINAL line -- unchanged

        sx, sy = self._cell_positions[source_id]                          # distance, as initialize_weight computes it
        d = np.linalg.norm(np.array([sx, sy]) - np.array([x, y]))
        w_comp = c_t / total_morphogen                                    # the saturating competitive value
        w_dist = c_t / (1.0 + d)                                          # the distance-graded value (post-transpose orientation preserved)
        if mode == "informative":
            return max(0.01, (w_comp ** (1.0 - cfg.alpha)) * (w_dist ** cfg.alpha))
        elif mode == "contrast":                                         # mechanistic control (no distance)
            return max(0.01, 0.125 + cfg.gamma * (w_comp - 0.125))
        elif mode == "dale":                                             # M4 E/I sign (Dale's law, signed)
            # per-SOURCE sign (genome-seeded, ~p_inh inhibitory); distance-graded magnitude
            src_h = (self._weight_seed * 73856093) ^ (int(source_id) * 19349663)
            sign = -1.0 if (src_h % 1000000) < int(cfg.p_inh * 1000000) else 1.0
            mag = min(cfg.w_max, max(0.01, w_dist * cfg.dale_scale))
            return sign * mag
        elif mode == "lognormal":                                       # M2 heavy-tailed positive (Buzsaki)
            # per-EDGE multiplicative lognormal noise around the v1.1 competitive value
            e_seed = ((self._weight_seed * 73856093) ^ (int(source_id) * 19349663) ^ (int(target_id) * 83492791)) & 0x7FFFFFFF
            z = random.Random(int(e_seed)).gauss(0.0, 1.0)
            return min(cfg.w_max, max(0.01, w_comp * math.exp(cfg.ln_sigma * z)))
        elif mode == "chemo":                                           # M3 chemoaffinity tag product (Sperry)
            c_src = self._weight_adjustment_morphogen[sx, sy]
            return min(cfg.w_max, max(0.01, c_src * c_t * cfg.chemo_scale))
        elif mode == "chemo_signed":                                    # M5 multi-tag chemoaffinity w/ repulsion
            # score = sum_k a_k * t_src[k] * t_tgt[k]; sign by tag mismatch (repulsive channels).
            if cfg.chemo_sign_mode == "genome":
                signs = [(-1.0 if (((self._weight_seed * 73856093) ^ (int(k) * 19349663)) % 2) else 1.0)
                         for k in cfg.chemo_tags]
            else:
                signs = cfg.chemo_signs
            score = 0.0
            for k, a in zip(cfg.chemo_tags, signs):
                score += float(a) * self._M[k][sx, sy] * self._M[k][x, y]
            sign = 1.0 if score >= 0 else -1.0
            return sign * min(cfg.w_max, max(0.01, abs(score) * cfg.chemo_scale))
        elif mode == "dog":                                            # M6 center-surround / Mexican-hat
            d2 = d * d
            dog = (cfg.dog_a * math.exp(-d2 / (2.0 * cfg.dog_s1 * cfg.dog_s1))
                   - cfg.dog_b * math.exp(-d2 / (2.0 * cfg.dog_s2 * cfg.dog_s2)))
            if cfg.dog_invert:
                dog = -dog
            val = c_t * dog
            sign = 1.0 if val >= 0 else -1.0
            return sign * min(cfg.w_max, max(0.01, abs(val) * cfg.dog_scale))
        elif mode == "dale_field":                                     # M9d-inv sign by dominant existing field
            c_sign = self._M[cfg.sign_field_idx][x, y]
            sign = 1.0 if c_t >= c_sign else -1.0
            mag = min(cfg.w_max, max(0.01, w_dist * cfg.dale_scale))
            return sign * mag
        else:
            raise ValueError(f"unknown weight_channel mode {mode!r}")

    def update_weight(self, source_id, target_id):
        """Update weight between neurons using the configured weight channel.

        v1.1 ("saturating") writes the competitive c_local/c_total ratio -- the canonical
        rule. The weight channel (weight_channel.get_weight_channel) selects the readout;
        the default mode "saturating" keeps this byte-for-byte identical to canonical (the
        repro gate), since _channel_weight's saturating branch is the literal original
        expression. See notes/v2-weight-channel-design.md sec. 2.2 and
        notes/mn-v2-sweep-design.md sec. 2."""
        cfg = get_weight_channel()
        new_weight = self._channel_weight(source_id, target_id, cfg, cfg.mode)
        if new_weight is not None:
            self.neuron_connections[source_id - 1, target_id - 1] = new_weight

    def temporal_weight_update(self):
        """M7 (notes/mn-v2-sweep-design.md sec. 2.3): nudge every existing edge's realized
        weight toward the base_mode target by temporal_rate, so the realized weight is an
        integral of the (signed) rule over the growth trajectory rather than a single field
        snapshot. The sign-preserving floor keeps |w| >= 0.01 > 0, so nnz -- and therefore
        every growth decision -- is invariant. Called from step() only when cfg.temporal;
        consumes no growth RNG."""
        cfg = get_weight_channel()
        source_indices, target_indices = self.neuron_connections.nonzero()
        if len(source_indices) == 0:
            return
        for si, ti in zip(source_indices, target_indices):
            w_star = self._channel_weight(int(si) + 1, int(ti) + 1, cfg, cfg.base_mode)
            if w_star is None:
                continue
            w = self.neuron_connections[si, ti]
            w = w + cfg.temporal_rate * (w_star - w)
            sign = 1.0 if w >= 0 else -1.0
            self.neuron_connections[si, ti] = sign * min(cfg.w_max, max(0.01, abs(w)))

    def refine_weights(self):
        """M8 (notes/mn-v2-sweep-design.md sec. 2.4): post-growth activity-dependent
        developmental refinement (Avenue A). Build the dense weight matrix from the grown
        graph (forward-flow W[post, pre], matching the propagator), run intrinsic
        task-agnostic activity, and refine each EXISTING edge by refine_rule. Writes only
        existing nonzero entries with a sign-preserving floor (|w| >= 0.01), so nnz -- and
        topology -- is invariant. Activity is seeded from self._weight_seed ONLY (never the
        growth RNG). Called from final_step() when cfg.refine."""
        cfg = get_weight_channel()
        source_indices, target_indices = self.neuron_connections.nonzero()
        E = len(source_indices)
        if E == 0:
            return

        node_ids = sorted(set(int(s) for s in source_indices) | set(int(t) for t in target_indices))
        pos = {nid: i for i, nid in enumerate(node_ids)}
        K = len(node_ids)

        pre = np.array([pos[int(s)] for s in source_indices], dtype=int)    # source = presynaptic
        post = np.array([pos[int(t)] for t in target_indices], dtype=int)   # target = postsynaptic
        w0 = np.array([self.neuron_connections[s, t] for s, t in zip(source_indices, target_indices)], dtype=float)

        Wd = np.zeros((K, K), dtype=float)
        Wd[post, pre] = w0   # forward flow: y = tanh(Wd @ x) sums each neuron's presynaptic inputs

        rng = np.random.default_rng(self._weight_seed)
        patterns = None
        if cfg.refine_drive == "fixed":
            Q, _ = np.linalg.qr(rng.standard_normal((K, K)))   # orthonormal input ensemble
            patterns = Q
        eta = cfg.refine_eta

        corr = np.zeros(E, dtype=float)   # used by the "sign" rule
        for step in range(cfg.refine_steps):
            x = patterns[:, step % K] if patterns is not None else rng.standard_normal(K)
            y = np.tanh(Wd @ x)
            yi, yj = y[pre], y[post]   # pre- / post-synaptic activity per edge
            if cfg.refine_rule == "hebb":
                Wd[post, pre] = Wd[post, pre] + eta * yi * yj
            elif cfg.refine_rule == "antihebb":
                Wd[post, pre] = Wd[post, pre] - eta * yi * yj
            elif cfg.refine_rule == "oja":
                wij = Wd[post, pre]
                Wd[post, pre] = wij + eta * yj * (yi - yj * wij)
            elif cfg.refine_rule == "sign":
                corr += yi * yj
            else:
                raise ValueError(f"unknown refine_rule {cfg.refine_rule!r}")

        if cfg.refine_rule == "sign":
            # anti-correlated pre/post -> inhibitory; magnitude = |base weight| (explicit sign assignment)
            new_w = np.where(corr >= 0, 1.0, -1.0) * np.abs(w0)
        else:
            new_w = Wd[post, pre]

        # sign-preserving floor + clamp; write back ONLY existing edges (nnz invariant)
        new_w = np.where(new_w >= 0, 1.0, -1.0) * np.clip(np.abs(new_w), 0.01, cfg.w_max)
        for si, ti, w in zip(source_indices, target_indices, new_w):
            self.neuron_connections[si, ti] = float(w)

    def adjust_all_weights(self):
        """Vectorized homeostatic adjustment of all weights in the network.
        Weights will tend towards the local morphogen concentration, scaled by the adjustment rate."""
        MIN_WEIGHT = 0.01
        
        # Get non-zero indices once
        source_indices, target_indices = self.neuron_connections.nonzero()
        if len(source_indices) == 0:
            return
        
        # Get all target positions at once
        target_positions = np.array([self._cell_positions[tid + 1] for tid in target_indices])
        
        # Get morphogen values for all target positions at once
        target_morphogens = self._weight_adjustment_morphogen[target_positions[:, 0], target_positions[:, 1]]
        
        # Get current weights
        current_weights = self.neuron_connections[source_indices, target_indices].toarray().flatten()
        
        # Calculate adjustments - weights will move towards the morphogen concentration
        # scaled by weight_adjustment_target (as maximum) and weight_adjustment_rate (as speed)
        target_weights = target_morphogens * self.weight_adjustment_target
        adjustments = self.weight_adjustment_rate * (target_weights - current_weights)
        
        # Update weights with new adjustments
        new_weights = np.clip(current_weights + adjustments, MIN_WEIGHT, 1.0)
        
        # Update weights one by one in the sparse matrix
        for i in range(len(source_indices)):
            self.neuron_connections[source_indices[i], target_indices[i]] = new_weights[i]

    def grow_axon(self, cell_id, morphogen_concentration):
        """Grow the axon of a neuron cell using precomputed morphogen concentrations."""
        cell_pos = self._cell_positions[cell_id]
        
        # Check if the axon has reached its maximum length
        if len(self._axons[cell_id]) >= self.max_axon_length:
            return

        current_tip = self._axons[cell_id][-1] if self._axons[cell_id] else cell_pos

        # Get neighboring positions of the current axon tip
        x, y = current_tip
        neighbors = self.get_neighbors(x, y)

        # Filter out positions already part of the axon
        available_positions = [pos for pos in neighbors if pos not in self._axons[cell_id]]

        if available_positions:
            # Get morphogen concentrations at available positions
            concentrations = [morphogen_concentration[pos] for pos in available_positions]

            # Filter positions based on the axon growth threshold
            valid_positions = [
                (pos, conc) for pos, conc in zip(available_positions, concentrations)
                if conc >= self.axon_growth_threshold
            ]

            if valid_positions:  # Only proceed if there are valid positions
                # Find the position with the highest morphogen concentration
                max_pos, _ = max(valid_positions, key=lambda x: x[1])

                # Add the new position to the axon
                if not self._axons[cell_id]:
                    self._axons[cell_id] = [cell_pos, max_pos]
                else:
                    self._axons[cell_id].append(max_pos)

    def connect_neurons(self, source_id, target_pos):
        """Connect neurons if conditions are met and initialize weight."""
        target_id = self._neurons[target_pos]

        if target_id in self._cell_positions:
            if self._axon_connect_morphogen[target_pos] >= self.axon_connect_threshold:
                # Update weight initialization
                self.initialize_weight(source_id, target_id)

                # Reset the axon to just the cell position
                self._axons[source_id] = [self._cell_positions[source_id]]
                return True
        return False

    def grow_axons(self):
        """Optimized axon growth and connection for all neurons."""
        if not self._neurons.any():
            return

        morphogen_concentration = self._axon_guidance_morphogen

        for cell_id in self.get_neuron_ids():
            if len(self._axons[cell_id]) > 1:  # Axon exists and has grown
                axon_tip = self._axons[cell_id][-1]
                if self.connect_neurons(cell_id, axon_tip):
                    # Update weight after successful connection
                    target_id = self._neurons[axon_tip]
                    self.update_weight(cell_id, target_id)
                    continue

            self.grow_axon(cell_id, morphogen_concentration)


    def add_listener(self, listener):
        """Add a listener that will be notified after each step."""
        self.listeners.append(listener)
    
    def step(self):
        """Perform one step of the simulation."""
        self.secrete_morphogens()
        self.inhibit_morphogens()
        # Step 0.5 (notes/step0.5-turing-probe-design-2026-06.md): the Gierer-Meinhardt
        # activator-inhibitor reaction, inserted between inhibit and diffuse (the standard
        # reaction-then-diffusion split). No-op under rd_mode == "canonical" (the default),
        # so step() stays byte-identical to v1.1 (the repro gate).
        self.react_morphogens()
        self.diffuse()
        self.divide_cells()
        self.differentiate_cells()
        self.grow_axons()
        # Homeostatic adjustment of all weights
        #self.adjust_all_weights()
        # M7 (notes/mn-v2-sweep-design.md sec. 2.3): integrate the base_mode rule over the
        # growth trajectory. Default temporal=False -> step() byte-identical (the repro gate).
        if get_weight_channel().temporal:
            self.temporal_weight_update()

        # Notify all listeners
        for listener in self.listeners:
            listener.on_step()

    def final_step(self):
        """Perform the final step of the simulation."""
        if self.self_connect_isolated_neurons_fraction > 0:
            self.self_connect_fraction_of_no_input_neurons()
        # M8 (notes/mn-v2-sweep-design.md sec. 2.4): activity-dependent refinement (Avenue A),
        # after self-connection so it sees the full graph. Default refine=False -> byte-identical.
        if get_weight_channel().refine:
            self.refine_weights()

    def no_input_neurons(self):
        """Get the IDs of all neurons with no input connections."""
        # Find columns with no non-zero values (no inputs)
        # and convert to 1-based cell IDs
        return [i + 1 for i in range(self.neuron_connections.shape[1]) 
                if self.neuron_connections[:, i].nnz == 0 and self.is_neuron(i + 1)]
    
    def self_connect_fraction_of_no_input_neurons(self):
        """Self connect a fraction of "no input" neurons."""
        no_input_ids = self.no_input_neurons()
        target_count = int(len(no_input_ids) * self.self_connect_isolated_neurons_fraction)
        
        # Use the seeded RNG to randomly sort the neurons
        selected_ids = self.rng.sample(no_input_ids, len(no_input_ids))
        
        # Take only the fraction we need
        for cell_id in selected_ids[:target_count]:
            self.connect_neurons(cell_id, self._cell_positions[cell_id])

    def get_morphogen_sum(self, morphogen_index):
        """Get the total sum of a specific morphogen."""
        return np.sum(self._M[morphogen_index])

    def get_morphogen_array(self, morphogen_index):
        """Get the concentration data for a specific morphogen as a numpy array."""
        return self._M[morphogen_index]

    def get_cell(self, cell_id):
        """Get data for a specific cell."""
        return self._cell_positions.get(cell_id)

    def get_axon(self, cell_id):
        """Get the axon for a specific cell."""
        return self._axons.get(cell_id)

    def is_neuron(self, cell_id):
        """Check if a cell is a neuron."""
        return cell_id in np.unique(self._neurons[self._neurons > 0])

    def is_progenitor(self, cell_id):
        """Check if a cell is a progenitor."""
        return cell_id in np.unique(self._progenitors[self._progenitors > 0])
    
    def get_neuron_ids(self):
        """Get the IDs of all neurons."""
        return np.unique(self._neurons[self._neurons > 0])
    
    def get_progenitor_ids(self):
        """Get the IDs of all progenitors."""
        return np.unique(self._progenitors[self._progenitors > 0])
    
    def get_cell_ids(self):
        """Get the IDs of all cells."""
        neuron_ids = self._neurons[self._neurons > 0]
        progenitor_ids = self._progenitors[self._progenitors > 0]
        return np.unique(np.concatenate([neuron_ids, progenitor_ids]))
    
    def neuron_count(self):
        """Get the number of neurons."""
        return np.sum(self._neurons > 0)
    
    def progenitor_count(self):
        """Get the number of progenitors."""
        return np.sum(self._progenitors > 0)

    def cell_count(self):
        """Get the total number of cells."""
        return np.sum(self._neurons > 0) + np.sum(self._progenitors > 0)

    def get_cell_position(self, cell_id):
        """Get the position of a specific cell."""
        neuron_pos = np.where(self._neurons == cell_id)
        if neuron_pos[0].size > 0:
            return (neuron_pos[0][0], neuron_pos[1][0])
        
        prog_pos = np.where(self._progenitors == cell_id)
        if prog_pos[0].size > 0:
            return (prog_pos[0][0], prog_pos[1][0])
        
        return None

    def _seed_initial_progenitors(self):
        """Seed the initial progenitor(s). v1.1 (composition off, or on but not multiseed):
        ONE progenitor at the grid center -- byte-identical to the canonical single seed.
        Composition multiseed: ONE progenitor per Hox region, at the region's x-center / y mid,
        so each region nucleates its own growth front (the "k reused copies" the k-replicated
        task rewards). At K=1 the single region center == the grid center, so K=1 multiseed is
        byte-identical to v1.1. Seeds are placed in region order (left-to-right), deterministic,
        consuming no growth RNG."""
        comp = get_composition_channel()
        if not (comp.compose and comp.multiseed):
            self.add_cell((self.size_x // 2, self.size_y // 2), "progenitor")
            return
        K = max(1, int(comp.K))
        cy = self.size_y // 2
        for r in range(K):
            cx = int((r + 0.5) * self.size_x / K)
            cx = min(self.size_x - 1, max(0, cx))
            self.add_cell((cx, cy), "progenitor")

    def run_simulation(self, verbose=True, display_weights=False):
        """Run simulation for specified number of steps.
        
        Args:
            verbose (bool): Whether to print progress and create displays
            display_weights (bool): Whether to print connection weights
            
        Returns:
            Grid: The grid instance after simulation
        """
        self._seed_initial_progenitors()

        if verbose:
            from .morphogen_display import MorphogenDisplay
            from .neuron_graph_display import NeuronGraphDisplay
            # Create displays first
            morphogen_display = MorphogenDisplay(self, update_frequency=10)

        # Run simulation with displays updating each step
        import time
        start_time = time.time()
        for i in range(self.max_growth_steps):
            self.step()  # This will automatically update displays via listeners
            if i % 100 == 0:
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                if verbose:
                    print(f"Step {i}; cells: {self.cell_count()}; elapsed: {elapsed_ms:.2f} ms")
        self.final_step()
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        if verbose:
            print(f"Simulation completed in {elapsed_ms:.2f} ms")
            source_indices, target_indices = self.neuron_connections.nonzero()
            weights = self.neuron_connections[source_indices, target_indices].toarray().flatten()
            if display_weights:
                weight_strings = [f"({s},{t})={w:.4f}" for s, t, w in zip(source_indices + 1, target_indices + 1, weights)]
                print("Weights:", " ".join(weight_strings))
            # Get diagonal elements (self-connections) and count non-zero ones
            self_connections = self.neuron_connections.diagonal()
            print(f"Number of neurons connected to themselves: {(self_connections > 0).sum()}")
            print(f"Sum of weights: {self.neuron_connections.sum()}")
            print(f"Neuron connections number: {self.neuron_connections.nnz}")
            morphogen_display.show(block=False)
            neuron_display = NeuronGraphDisplay(self)
            neuron_display.show(block=True)
        
        return self
