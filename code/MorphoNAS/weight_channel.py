"""v2 weight-channel config switch (notes/v2-weight-channel-design.md sec. 2.1;
MNv2 sweep extensions in notes/mn-v2-sweep-design.md sec. 2).

A process-level config consulted by ``Grid`` to optionally change the realized innate
edge weight (and, for M7/M8, to run a developmental weight *process*). It is **not**
part of the genome, so ``Genome.to_bytes`` / the md5 RNG seed / hashing / serialization
are all untouched -- which is what guarantees topology invariance (sec. 3) and the
bit-for-bit repro gate for every topology-invariant mechanism.

The process default is ``saturating`` (= v1.1). Any code path that grows a genome
without touching this config (the repro gate, every existing runner / analysis) gets
exact v1.1 behavior, byte-for-byte. **Every field added below defaults to the inert
value, so ``WeightChannel()`` is v1.1.**

Per-edge readout modes (computed by ``Grid._channel_weight``):
  * ``saturating`` -- v1.1: the competitive c_local/c_total ratio (the original rule).
  * ``informative`` -- v2.0: a geometric blend reinstating the engine's own distance
    channel, knob ``alpha`` in [0, 1] (0 == v1.1, 1 == pure distance).
  * ``contrast`` -- v2.0 mechanism control: amplify the residual competitive deviation
    from 1/8, knob ``gamma`` >= 1 (1 == v1.1); uses no distance.
  * ``dale`` (M4) -- random ~p_inh per-source sign, distance-graded magnitude.
  * ``lognormal`` (M2) -- per-edge heavy-tailed positive noise around the v1.1 value.
  * ``chemo`` (M3) -- single positive chemoaffinity tag product c_src*c_tgt.
  * ``chemo_signed`` (M5) -- multi-tag chemoaffinity with attraction/repulsion; sign by
    tag mismatch (the principled fix for dale's random sign).
  * ``dog`` (M6) -- center-surround / Mexican-hat signed distance kernel; sign by band.
  * ``dale_field`` (M9d-inv) -- signed by which of two existing morphogen fields dominates
    at the target (topology-invariant inhibitory-morphogen readout).

Cross-cutting process axes (sec. 2.3/2.4):
  * ``temporal`` (M7) -- re-enable a per-step weight-update loop so the realized weight is
    an integral over the growth trajectory; per-step target = ``base_mode``.
  * ``refine`` (M8) -- post-growth activity-dependent refinement (Avenue A); initial weights
    from ``base_mode``, then refined by ``refine_rule`` under intrinsic ``refine_drive``.
  * ``rd_antiplateau`` (M9c) -- relax the morphogen secretion ceiling (NOT topology-invariant).

All signed/stochastic rules are seeded from ``Grid._weight_seed`` (a SEPARATE md5-derived
seed, never the growth RNG), so weight randomness never perturbs growth -> topology stays
invariant for the M5/M6/M7/M8/M9d-inv mechanisms.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightChannel:
    mode: str = "saturating"   # saturating | informative | contrast | dale | lognormal | chemo | chemo_signed | dog | dale_field
    alpha: float = 0.0         # informative blend strength: 0 == v1.1, 1 == pure distance
    gamma: float = 1.0         # contrast gain (control mode only): 1 == v1.1
    # ── bio-inspired deterministic rules (notes/weight-map-expressivity-design.md sec. 3) ──
    # All genome-seeded (a SEPARATE hash, never the growth RNG) and topology-preserving.
    p_inh: float = 0.2         # dale: inhibitory fraction (Dale's law, per-source sign; ~cortical 80/20)
    dale_scale: float = 6.0    # dale: magnitude scale on the distance value (reach the signed [-3,3] ceiling range)
    ln_sigma: float = 1.0      # lognormal: multiplicative log-std around the v1.1 value (heavy tail)
    chemo_scale: float = 3.0   # chemo: scale on the source*target morphogen tag product
    w_max: float = 3.0         # magnitude clamp for the bio rules (= direct's PARAM_SCALE)

    # ── M5: multi-tag chemoaffinity with repulsion (sweep-design sec. 2.1) ──
    # score = sum_k chemo_signs[k] * t_src[k] * t_tgt[k] over the chemo_tags morphogen fields;
    # w = sign(score) * clamp(|score| * chemo_scale). Sign by tag mismatch.
    chemo_tags: tuple = (0, 1, 2)          # morphogen field indices read as tag vectors
    chemo_signs: tuple = (1.0, -1.0, 1.0)  # per-channel attraction(+)/repulsion(-): div attract, diff repel, weight attract
    chemo_sign_mode: str = "fixed"         # "fixed" (use chemo_signs) | "genome" (per-channel sign from _weight_seed)

    # ── M6: center-surround / Mexican-hat signed kernel (sweep-design sec. 2.2) ──
    # dog(d) = dog_a*exp(-d^2/2s1^2) - dog_b*exp(-d^2/2s2^2); w = sign(c_t*dog)*clamp(|c_t*dog|*dog_scale).
    dog_s1: float = 1.0        # near (excitatory) Gaussian width  (s1 < s2)
    dog_s2: float = 3.0        # far (inhibitory) Gaussian width
    dog_a: float = 1.0         # near amplitude
    dog_b: float = 1.0         # far amplitude
    dog_scale: float = 3.0     # magnitude scale on the DoG*c_t product
    dog_invert: bool = False   # M6a center-excite/surround-inhibit (False) | M6b inverted Mexican-hat (True)

    # ── M7: temporal weight trajectory (sweep-design sec. 2.3) ──
    # When temporal, step() nudges every existing edge's realized weight toward the base_mode
    # target by temporal_rate each growth step (sign-preserving), so the weight integrates the
    # signed rule over the developmental trajectory. base_mode is also M8's initial-weight rule.
    temporal: bool = False
    temporal_rate: float = 0.2
    base_mode: str = "saturating"   # per-step target rule (M7) / initial weights before refinement (M8)

    # ── M8: activity-dependent developmental refinement (Avenue A, sweep-design sec. 2.4) ──
    refine: bool = False
    refine_drive: str = "spontaneous"   # "spontaneous" (x ~ N(0,I)) | "fixed" (orthogonal input ensemble)
    refine_rule: str = "sign"           # "sign" (anti-correlated -> inhibitory) | "hebb" | "antihebb" | "oja"
    refine_steps: int = 50
    refine_eta: float = 0.05

    # ── M9: morphogen-substrate enrichment (Avenue B, sweep-design sec. 2.5) ──
    # M9a/b (dedicated/extra morphogens) are driven by --num-morphogens, no field here.
    rd_antiplateau: bool = False        # M9c: relax the secretion ceiling (NOT topology-invariant)
    rd_secretion_cap: float = 1.0       # M9c: secretion clamp when rd_antiplateau (1.0 == v1.1)
    sign_field_idx: int = 1             # M9d-inv (dale_field): the existing field read as the sign tag (1 = differentiation)

    # ── Step 0.5: the minimal Turing extension (notes/step0.5-turing-probe-design-2026-06.md) ──
    # The RD-Turing verdict (rd-turing-capability-findings) proved the canonical RD is a sum-1
    # smoother + cross-damping with NO autocatalytic activator and NO differential diffusion, so
    # it is architecturally pattern-incapable. rd_mode="turing" ADDS the two missing ingredients:
    #   (1) per-morphogen differential diffusion -- the A/H pair gets its own D_act << D_inh
    #       (inhibitor diffuses much faster), replacing the single shared diffusion_rate for that
    #       pair; every other morphogen keeps the genome's diffusion_rate (canonical smoothing);
    #   (2) a Gierer-Meinhardt activator-inhibitor reaction on the (A, H) pair, replacing the pure
    #       multiplicative damping that inhibit_morphogens would apply to that pair:
    #         dA = rho * A^2 / H - mu_A * A ;  dH = rho * A^2 - mu_H * H   (forward Euler, dt).
    # rd_mode="canonical" (default) is byte-identical to v1.1 (the repro gate). All reaction
    # randomness is absent (the GM update is deterministic in the field); determinism stays seeded
    # from Grid._weight_seed / the genome hash, never the growth RNG. Additive, default-off.
    # Standard Gierer-Meinhardt parameters, verified to sit in the Turing-unstable band on this
    # discrete (1-D)I + D*box grid: rho=1, mu_A=1, mu_H=2 (mu_H/mu_A=2 -> inhibitor turns over
    # faster), D_inh/D_act = 0.5/0.025 = 20 >> 1 (the differential-diffusion requirement). The
    # small basal activator production rho_a is the standard Meinhardt term that prevents the
    # A->0 absorbing state and lets a pattern form from a flat/uniform start (so the bare-field
    # uniform-source sanity gate flips). The activator peaks ~5x the mean, so the A/H pair is
    # NOT clamped to the canonical secretion cap (1.0) -- that clamp flattens GM (proven); they
    # are clamped only to a generous numerical ceiling. Both is/are off under canonical.
    rd_mode: str = "canonical"          # "canonical" (v1.1, byte-identical) | "turing"
    turing_act_idx: int = 0             # activator morphogen index (A); slow diffusion, autocatalytic
    turing_inh_idx: int = 1             # inhibitor morphogen index (H); fast diffusion, long-range
    turing_D_act: float = 0.025         # activator diffusion mix-weight (slow); (1-D)I + D*box form
    turing_D_inh: float = 0.50          # inhibitor diffusion mix-weight (fast; D_inh/D_act = 20 >> 1)
    turing_rho: float = 1.0             # GM autocatalysis coefficient (the A^2/H production scale)
    turing_rho_a: float = 0.01          # basal activator production (standard GM; avoids A->0)
    turing_rho_h: float = 0.0           # basal inhibitor production (standard GM; 0 = pure A-driven)
    turing_mu_act: float = 1.0          # activator linear decay mu_A
    turing_mu_inh: float = 2.0          # inhibitor linear decay mu_H (> mu_A: faster turnover)
    turing_dt: float = 0.1              # forward-Euler step for the GM reaction (per growth step)
    turing_H_floor: float = 1e-3        # guard so A^2/H never divides by ~0 (H kept >= floor)
    turing_ceiling: float = 50.0        # numerical safety clamp for the GM pair (>> the ~5x peak)
    turing_skip_secretion: bool = True  # turing: don't add/clamp the A/H pair via secrete (GM is
    #                                     their source); other morphogens secrete as in v1.1


_CONFIG = WeightChannel()  # process default = v1.1, byte-for-byte


def get_weight_channel() -> WeightChannel:
    return _CONFIG


def set_weight_channel(cfg: WeightChannel) -> None:
    global _CONFIG
    _CONFIG = cfg
