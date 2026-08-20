"""
Clone the rule, warm the critic, then fine-tune with RL.

    PYTHONPATH=. python part_preparation/solvers/bilevel/rl/main_cloned.py

Why this order. Policy gradient from scratch fails on this objective: three seeds land at
~1,134,000 against ~1,266,000 for random and ~491,000 for a one-line rule, and a run eight times
longer plateaus at 1,063,000. Cloning the same rule reaches ~480,000 in 70 seconds and beats its
own teacher. So the network can represent a good policy, supervised learning finds it, and only
the gradient-policy search fails -- which makes "learn it, then improve it" the sensible order
rather than "discover it from nothing".

The three stages are separate on purpose, and each is measured, so a regression can be
attributed: cloning, critic warm-up (actor frozen), then joint fine-tuning.
"""
import contextlib
import io
import time

import numpy as np
import torch

from part_preparation import instances
from part_preparation.instance_generator import generate_instance
from part_preparation.solvers.bilevel.rl import cloning
from part_preparation.solvers.bilevel.rl.agent import (
    ActorCriticNetwork, ENGINE_FEATURES, MSN_FEATURES, FACTORY_BASE_FEATURES)
from part_preparation.solvers.bilevel.rl.environnement import Env
from part_preparation.solvers.bilevel.rl.features import FeatureExtractor
from part_preparation.solvers.bilevel.rl.training import run_episode, train

TRAIN_SEEDS = range(60)
EVAL_INSTANCE = "medium"
SELECT_REPLICATES = range(90, 94)     # drives the checkpoint
TEST_REPLICATES = range(110, 130)     # disjoint, drives the reported number
WEIGHTS = "part_preparation/solvers/bilevel/rl/policy_cloned.pt"


def subproblems(problems):
    envs, extractors = [], []
    for problem in problems:
        for engine_type in sorted({e.engine_type for e in problem.engines_list}):
            env = Env(problem, engine_type)
            envs.append(env)
            extractors.append(FeatureExtractor(env))
    return envs, extractors


def cost(envs, extractors, network):
    """Cost per INSTANCE, the two engine types summed back together, argmax rollout."""
    per = []
    for env, extractor in zip(envs, extractors):
        with torch.no_grad():
            per.append(-sum(run_episode(env, network, extractor, greedy=True)[2]))
    return np.array([sum(per[i:i + 2]) for i in range(0, len(per), 2)])


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        train_envs, train_ex = subproblems(
            [generate_instance(instances.varied_params(s, n_msns_per_fal=40))
             for s in TRAIN_SEEDS])
        select_envs, select_ex = subproblems(
            [instances.build(EVAL_INSTANCE, replicate_id=r) for r in SELECT_REPLICATES])
        test_envs, test_ex = subproblems(
            [instances.build(EVAL_INSTANCE, replicate_id=r) for r in TEST_REPLICATES])
    print(f"{len(train_envs)} training sub-problems, "
          f"{len(select_envs)} for selection, {len(test_envs)} for test")

    network = ActorCriticNetwork(
        engine_input_dim=ENGINE_FEATURES,
        msn_input_dim=MSN_FEATURES,
        factory_input_dim=FACTORY_BASE_FEATURES + max(
            f.default_capacity for f in train_envs[0].problem.factories_list),
        state_input_dim=len(train_ex[0].state_features(train_envs[0].reset())),
        hidden_dim=64,
    )
    rng = np.random.default_rng(0)

    print("\n=== 1. cloning ===")
    started = time.perf_counter()
    samples, expert_cost = cloning.collect(train_envs, train_ex)
    print(f"{len(samples):,} expert samples, teacher scores {expert_cost:,.0f} on training")
    cloning.clone(network, samples, epochs=12, lr=1e-3, seed=0)
    cloned = cost(test_envs, test_ex, network)
    print(f"  cloned policy (test) : {cloned.mean():11,.0f} +/- {cloned.std():9,.0f}"
          f"   [{time.perf_counter() - started:.0f}s]")

    print("\n=== 2. critic warm-up (actor frozen) ===")
    cloning.warm_critic(train_envs, train_ex, network, rng, iterations=30, episodes=4)
    warmed = cost(test_envs, test_ex, network)
    # Must be identical to the line above: the actor was frozen, so the policy cannot have moved.
    # Any difference here is a bug in the freezing.
    print(f"  after warm-up (test) : {warmed.mean():11,.0f}   "
          f"(expected unchanged, delta {abs(warmed.mean() - cloned.mean()):,.0f})")

    print("\n=== 3. RL fine-tuning ===")
    best_before = cost(select_envs, select_ex, network).mean()
    history, best = train(
        train_envs, network, train_ex,
        iterations=150, episodes_per_batch=8, lr=1e-4,   # small lr: refine, do not overwrite
        gamma=1.0, lam=0.95, entropy_coef=0.0,           # no exploration bonus on a good policy
        seed=0, log_every=10,
        eval_fn=lambda: cost(select_envs, select_ex, network).mean(),
    )

    final = cost(test_envs, test_ex, network)
    print(f"\n  after fine-tuning    : {final.mean():11,.0f} +/- {final.std():9,.0f}")
    if best["state"] is not None and best["eval"] < best_before:
        network.load_state_dict(best["state"])
        restored = cost(test_envs, test_ex, network)
        print(f"  best checkpoint      : {restored.mean():11,.0f} (iter {best['iteration']})")
    else:
        print(f"  fine-tuning did not improve on the clone -- keeping the cloned weights")

    torch.save(network.state_dict(), WEIGHTS)
    print(f"\n  saved to {WEIGHTS}")


if __name__ == "__main__":
    main()