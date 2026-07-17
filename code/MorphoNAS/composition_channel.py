"""Genome-gated Hox-like composition config switch
(notes/structural-composition-design-2026-06.md; engine-design decision in
notes/structural-composition-findings-2026-06.md sec. 1).

A process-level config consulted by ``Grid.secrete_morphogens`` to optionally make the
developmental *secretion program* region-dependent with reuse – the segmentation/Hox-
stripe analog. It is **not** part of the genome, so ``Genome.to_bytes`` / the md5 RNG
seed / hashing / serialization are all untouched, which guarantees topology invariance
when off and a byte-identical K=1 reduction (the repro gate + the prior-preservation-by-
construction gate).

The default is ``compose=False`` (= v1.1). Any code path that grows a genome without
touching this config (the repro gate, every existing runner / analysis) gets exact v1.1
behavior, byte-for-byte. Every field defaults to the inert value, so
``CompositionChannel()`` is v1.1, AND ``CompositionChannel(compose=True, K=1)`` is also
byte-identical to v1.1 (one region, the genome's own motif, applied uniformly).

Mechanism (see findings sec. 1.2):
  * Positional control morphogens = K vertical stripes by grid x-coordinate:
        region(x) = min(K-1, floor(x * K / size_x)).
    A pure function of position – no extra morphogen field, no growth RNG, deterministic.
  * Reused motif library of M secretion-programs. Motif 0 is the genome's OWN
    (progenitor_secretion_rates, neuron_secretion_rates) – encoded ONCE. Motif m is a
    fixed, deterministic re-parameterization of motif 0 (default ``motif="gain"``: scale
    by a per-motif scalar gains[m], ROLE-PRESERVING so growth stays viable; the ``roll``
    channel-shift alternative is kept for the design record but collapses most nets). This
    is "reuse a generic motif" – the per-task cost is the composition (K + selector), not
    M independent secretion vectors; no new free genome bits beyond the O(K log M) selector.
  * Per-region selector ``sel`` of length K, each entry in [0, M): region r secretes
    under motif sel[r]. K regions x M motifs = M^K arrangements from O(K log M) control
    bits – the combinatorial reach the H(G) theorem predicts.

The change lives ENTIRELY in secrete_morphogens(); field dynamics (diffuse/inhibit/react),
division, differentiation, axon growth, and weights are untouched. The weight channel is
held identical across arms (the INNATE read is primary), so this isolates architecture.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompositionChannel:
    compose: bool = False          # master switch; False == v1.1 byte-for-byte
    K: int = 1                     # number of Hox regions (vertical stripes by x); K=1 == v1.1
    M: int = 1                     # motif-library size (M=1 == only the genome's own motif)
    # per-region selector: which motif each region secretes under; length must == K.
    # default () means "identity selector" sel[r] = 0 for all r (every region = motif 0 =
    # the genome's own program), which keeps K>1 inert until a non-trivial selector is set.
    selector: tuple = field(default_factory=tuple)

    # motif transform that builds library entry m from the genome's own program (motif 0):
    #   "gain" -- ROLE-PRESERVING: scale the genome's secretion vectors by a per-motif scalar
    #             gains[m] (channel roles div/diff/axon preserved -> growth stays viable; the
    #             prototype showed roll collapses ~60% of nets, gain ~5%). gains[0] MUST be 1.0
    #             so motif 0 == the genome's own program (the K=1 byte-identity guarantee).
    #   "roll"  -- channel-shift np.roll by m (kept for the design record / ablation; collapses
    #             most nets because it scrambles which morphogen plays which role).
    motif: str = "gain"
    gains: tuple = (1.0,)          # per-motif scalar gains (motif="gain"); len must be >= M; [0]==1.0

    # multiseed: seed ONE progenitor per region (at the region's x-center, y mid) instead of the
    # single grid-center seed, so each Hox domain nucleates its own growth front -> repeated/
    # modular sub-structure (the "k reused copies" the k-replicated task rewards). At K=1 this is
    # exactly the single center seed (region 0 center == grid center), so K=1 stays byte-identical.
    multiseed: bool = False

    def resolved_selector(self):
        """Return the length-K selector, filling the identity (all-0) selector when empty."""
        if not self.selector:
            return tuple(0 for _ in range(self.K))
        if len(self.selector) != self.K:
            raise ValueError(
                f"composition selector length {len(self.selector)} != K {self.K}")
        return tuple(int(s) % max(1, self.M) for s in self.selector)


_CONFIG = CompositionChannel()  # process default = v1.1, byte-for-byte


def get_composition_channel() -> CompositionChannel:
    return _CONFIG


def set_composition_channel(cfg: CompositionChannel) -> None:
    global _CONFIG
    _CONFIG = cfg
