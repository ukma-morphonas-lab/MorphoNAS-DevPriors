"""MorphoNAS-DevPriors shared libraries (the reach-map: developmental priors).

Experiment libraries layered on top of the vendored MorphoNAS engine without
modifying its core logic: the Acrobot evaluation harness, the task/stratum
registry, the matched-random graph generator, ratio statistics, reward
wrappers, and run/parallel/logging utilities. Evaluation is structural – no plasticity
and no learning; propagators are built with no edge hook.
"""
