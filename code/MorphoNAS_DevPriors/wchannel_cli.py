"""Shared CLI surface for the engine weight channel (MNv2 sweep M5-M9).

One source of truth for the ``--wchannel-*`` flags and the ``WeightChannel`` they build,
so ``run_pool``, ``run_evo_tier1`` and ``verify_topology_invariance`` expose an identical
knob set across the whole M5-M9 mechanism family (notes/mn-v2-sweep-design.md sec. 2, 9).

The full ``WeightChannel`` (a frozen, picklable dataclass) is threaded through each
runner's worker ``initargs`` -- ``set_weight_channel(cfg)`` in the worker -- so every
knob reaches the growing process across fork/spawn. An unset config == ``saturating`` ==
v1.1 byte-for-byte, which is what keeps the repro gate green for the default path.

Usage in a runner::

    from MorphoNAS_DevPriors.wchannel_cli import add_wchannel_args, build_weight_channel
    add_wchannel_args(parser)
    ...
    cfg = build_weight_channel(args)
    Pool(..., initializer=_init_worker, initargs=(cfg, ...))   # _init_worker: set_weight_channel(cfg)
"""

from __future__ import annotations

import argparse
import dataclasses

from MorphoNAS.weight_channel import WeightChannel

# Per-edge readout modes computed by Grid._channel_weight (also the valid base_mode values).
WCHANNEL_MODES = [
    "saturating", "informative", "contrast",   # v1.1 + v2.0
    "dale", "lognormal", "chemo",              # M4 / M2 / M3
    "chemo_signed",                            # M5
    "dog",                                     # M6
    "dale_field",                              # M9d-inv
]


def add_wchannel_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the engine weight-channel knobs (M5-M9). NOT --weight-mode, which is taken
    elsewhere (run_weight_ablation ablation cell; run_evo_tier1 CPPN mode)."""
    g = parser.add_argument_group("engine weight channel (MNv2 M5-M9)")
    g.add_argument("--wchannel-mode", type=str, default="saturating", choices=WCHANNEL_MODES,
                   help="per-edge readout: saturating(v1.1) | informative | contrast | dale | "
                        "lognormal | chemo | chemo_signed(M5) | dog(M6) | dale_field(M9d-inv)")
    g.add_argument("--wchannel-alpha", type=float, default=0.0, help="informative blend in [0,1]")
    g.add_argument("--wchannel-gamma", type=float, default=1.0, help="contrast gain >= 1")
    # M5 chemo_signed
    g.add_argument("--wchannel-chemo-sign-mode", type=str, default="fixed", choices=["fixed", "genome"],
                   help="M5: per-channel attraction/repulsion sign source")
    # M6 dog
    g.add_argument("--wchannel-dog-invert", action="store_true",
                   help="M6b: inverted Mexican-hat (center-inhibit/surround-excite); default M6a")
    # M7 temporal
    g.add_argument("--wchannel-temporal", action="store_true",
                   help="M7: integrate base_mode over the growth trajectory (per-step update)")
    g.add_argument("--wchannel-temporal-rate", type=float, default=0.2)
    g.add_argument("--wchannel-base-mode", type=str, default="saturating", choices=WCHANNEL_MODES,
                   help="M7 per-step target rule / M8 initial-weight rule")
    # M8 refine
    g.add_argument("--wchannel-refine", action="store_true",
                   help="M8: post-growth activity-dependent refinement (Avenue A)")
    g.add_argument("--wchannel-refine-drive", type=str, default="spontaneous", choices=["spontaneous", "fixed"])
    g.add_argument("--wchannel-refine-rule", type=str, default="sign", choices=["sign", "hebb", "antihebb", "oja"])
    g.add_argument("--wchannel-refine-steps", type=int, default=50)
    g.add_argument("--wchannel-refine-eta", type=float, default=0.05)
    # M9c / M9d-inv
    g.add_argument("--wchannel-rd-antiplateau", action="store_true",
                   help="M9c: relax the secretion ceiling (NOT topology-invariant)")
    g.add_argument("--wchannel-rd-secretion-cap", type=float, default=1.0,
                   help="M9c secretion clamp when --wchannel-rd-antiplateau (1.0 == v1.1)")
    g.add_argument("--wchannel-sign-field-idx", type=int, default=1,
                   help="M9d-inv (dale_field): existing field read as the sign tag (1=differentiation)")
    return parser


def build_weight_channel(args: argparse.Namespace) -> WeightChannel:
    """Construct the WeightChannel from parsed --wchannel-* args. Knobs not present on
    `args` fall back to the frozen WeightChannel defaults, so a runner that only added a
    subset of the surface still works."""
    def g(name, default):
        return getattr(args, name, default)

    base = WeightChannel()  # frozen design defaults for anything a runner did not expose
    return WeightChannel(
        mode=g("wchannel_mode", base.mode),
        alpha=g("wchannel_alpha", base.alpha),
        gamma=g("wchannel_gamma", base.gamma),
        chemo_sign_mode=g("wchannel_chemo_sign_mode", base.chemo_sign_mode),
        dog_invert=g("wchannel_dog_invert", base.dog_invert),
        temporal=g("wchannel_temporal", base.temporal),
        temporal_rate=g("wchannel_temporal_rate", base.temporal_rate),
        base_mode=g("wchannel_base_mode", base.base_mode),
        refine=g("wchannel_refine", base.refine),
        refine_drive=g("wchannel_refine_drive", base.refine_drive),
        refine_rule=g("wchannel_refine_rule", base.refine_rule),
        refine_steps=g("wchannel_refine_steps", base.refine_steps),
        refine_eta=g("wchannel_refine_eta", base.refine_eta),
        rd_antiplateau=g("wchannel_rd_antiplateau", base.rd_antiplateau),
        rd_secretion_cap=g("wchannel_rd_secretion_cap", base.rd_secretion_cap),
        sign_field_idx=g("wchannel_sign_field_idx", base.sign_field_idx),
    )


def wchannel_to_dict(cfg: WeightChannel) -> dict:
    """Full config -> JSON-able dict for the run-record metadata (audit trail)."""
    d = dataclasses.asdict(cfg)
    d["chemo_tags"] = list(cfg.chemo_tags)
    d["chemo_signs"] = list(cfg.chemo_signs)
    return d
