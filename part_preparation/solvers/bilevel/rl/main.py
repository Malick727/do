"""
Pipeline entry point: build the sub-problems, train one shared network, evaluate, plot.

    PYTHONPATH=. python part_preparation/solvers/bilevel/rl/main.py

One episode is one (instance, engine type) sub-problem. The problem splits exactly by type --
factories are type-exclusive, every MSN has one type, and the FALs carry no capacity -- so
score(solution) = score(type 1) + score(type 2) and two episodes cover one instance.

The MODEL is not split: the heads score candidates through their features and never through
their identity, so a single network serves both types and both instance sizes.

Episode cost is directly comparable to the project's objective. Summing (travel_cost + delay)
over a type gives aggregate_score for that type: storage has weight 0, and the coverage term is
0 by construction since the episode ends exactly when no engine or no MSN is left, which is the
maximum matching.
"""
import contextlib
import io
import time

import numpy as np
import torch

from part_preparation import instances
from part_preparation.instance_generator import generate_instance
from part_preparation.solvers.bilevel.rl.agent import (
    ActorCriticNetwork, ENGINE_FEATURES, MSN_FEATURES, FACTORY_BASE_FEATURES)
from part_preparation.solvers.bilevel.rl.environnement import Env
from part_preparation.solvers.bilevel.rl.features import FeatureExtractor
from part_preparation.solvers.bilevel.rl.training import run_episode, train

# Training on VARIED instances, evaluation on a NAMED one. Varied instances make the policy
# learn a rule instead of memorising one congestion profile: takts, supplier regularity and the
# engine/MSN ratios are drawn, so the delivery-to-deadline spread goes from 2.1 days of standard
# deviation (fixed medium) up to ~5, which is what makes choosing the MSN a real decision.
# Instance SIZE is drawn too (20 to 120 MSN per FAL, i.e. 90 to 500 engines), because several
# features -- normalised ranks, pool fractions -- do not behave the same on a pool of 80 and one
# of 500. The previous run trained only around 80 engines per type and had to generalise to 500.
# Evaluation stays on the named instance, the only one comparable to the GA, CP-SAT and dataset.
TRAIN_SEEDS = range(1200)
EVAL_INSTANCE = "medium"
EVAL_REPLICATES = range(90, 94)   # held out: a policy scored on what it trained on proves nothing
WEIGHTS = "part_preparation/solvers/bilevel/rl/policy.pt"
CURVE = "part_preparation/solvers/bilevel/rl/training_curve.png"
HISTORY = "part_preparation/solvers/bilevel/rl/training_history.csv"


def _subproblems(problems):
    """One (Env, FeatureExtractor) pair per engine type, types kept in a stable order so that
    consecutive pairs belong to the same instance and can be summed back together."""
    envs, extractors = [], []
    for problem in problems:
        for engine_type in sorted({e.engine_type for e in problem.engines_list}):
            env = Env(problem, engine_type)
            envs.append(env)
            extractors.append(FeatureExtractor(env))
    return envs, extractors


def build_varied(seeds):
    with contextlib.redirect_stdout(io.StringIO()):       # the generator prints a report
        problems = [generate_instance(instances.varied_params(s)) for s in seeds]
    return _subproblems(problems)


def build_named(instance, replicates):
    with contextlib.redirect_stdout(io.StringIO()):
        problems = [instances.build(instance, replicate_id=r) for r in replicates]
    return _subproblems(problems)


def total_cost(envs, extractors, network, greedy=True, rng=None):
    """
    Cost per INSTANCE, the two engine types summed back together.

    rng given -> uniform random over the legal candidates, the floor to beat.
    """
    per_subproblem = []
    for env, extractor in zip(envs, extractors):
        if rng is None:
            with torch.no_grad():
                rewards = run_episode(env, network, extractor, greedy=greedy)[2]
        else:
            rewards = _random_rollout(env, extractor, rng)
        per_subproblem.append(-sum(rewards))
    return np.array([sum(per_subproblem[i:i + 2]) for i in range(0, len(per_subproblem), 2)])


def _random_rollout(env, extractor, rng):
    state, rewards = env.reset(), []
    while True:
        _, engine_idx = extractor.engine_candidates(state)
        if len(engine_idx) == 0:
            break
        engine_i = int(rng.choice(engine_idx))
        _, msn_idx = extractor.msn_candidates(state, engine_i)
        if len(msn_idx) == 0:
            break
        msn_i = int(rng.choice(msn_idx))
        _, factory_idx = extractor.factory_candidates(state, engine_i, msn_i)
        factory_i = int(rng.choice(factory_idx))
        state, reward, done, _ = env.step((
            env.engines[engine_i].id, env.factories[factory_i].id, env.msns[msn_i].id))
        rewards.append(reward)
        if done:
            break
    return rewards


def save_history(history, path):
    """
    Dump the per-iteration records to CSV.

    The PNG alone is a dead end: it cannot be re-plotted, rescaled, or compared against another
    run. Keeping the numbers costs one file and makes every later question answerable without
    retraining.
    """
    import csv
    fields = ["iteration", "cost", "cost_normalised", "entropy", "eval"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: record.get(k) for k in fields} for record in history)
    print(f"  historique           : {path}")


def plot(history, baseline, path):
    """
    Two curves, because they answer different questions.

    Sampled cost is normalised by a per-sub-problem reference: raw batch means would mostly show
    which sub-problems were drawn (type 1 costs ~187k, type 2 ~151k on medium), not whether the
    policy improved. Greedy cost is the one to trust -- fixed held-out set, no sampling noise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [h["iteration"] for h in history]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    top.plot(iterations, [h["cost_normalised"] for h in history],
             lw=1, alpha=.5, label="echantillonne (x reference)")
    evaluated = [(h["iteration"], h["eval"]) for h in history if h["eval"] is not None]
    if evaluated:
        twin = top.twinx()
        twin.plot(*zip(*evaluated), color="tab:red", marker="o", ms=3, label="argmax, hors echantillon")
        twin.axhline(baseline, color="gray", ls="--", lw=1, label="aleatoire")
        twin.set_ylabel("cout (euros)")
        twin.legend(loc="upper right")
    top.set_ylabel("cout normalise")
    top.legend(loc="lower left")

    bottom.plot(iterations, [h["entropy"] for h in history], color="tab:green")
    bottom.axhline(1.0, color="gray", ls="--", lw=1)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("entropie normalisee")
    bottom.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  courbe sauvegardee   : {path}")


def main():
    started_build = time.perf_counter()
    train_envs, train_extractors = build_varied(TRAIN_SEEDS)
    eval_envs, eval_extractors = build_named(EVAL_INSTANCE, EVAL_REPLICATES)
    print(f"{len(train_envs)} sous-problemes d'entrainement ({len(list(TRAIN_SEEDS))} instances "
          f"variees), {len(eval_envs)} d'evaluation ({EVAL_INSTANCE}) "
          f"[{time.perf_counter() - started_build:.0f}s]")

    max_capacity = max(f.default_capacity for f in train_envs[0].problem.factories_list)
    state_dim = len(train_extractors[0].state_features(train_envs[0].reset()))
    network = ActorCriticNetwork(
        engine_input_dim=ENGINE_FEATURES,
        msn_input_dim=MSN_FEATURES,
        factory_input_dim=FACTORY_BASE_FEATURES + max_capacity,
        state_input_dim=state_dim,
        hidden_dim=64,
    )
    print(f"state_dim={state_dim}  factory_features={FACTORY_BASE_FEATURES + max_capacity}  "
          f"{sum(p.numel() for p in network.parameters()):,} parametres\n")

    rng = np.random.default_rng(0)
    random_cost = total_cost(eval_envs, eval_extractors, network, rng=rng)
    print(f"  aleatoire            : {random_cost.mean():11,.0f} +/- {random_cost.std():9,.0f}")
    print(f"  non entraine (argmax): {total_cost(eval_envs, eval_extractors, network).mean():11,.0f}\n")

    started = time.perf_counter()
    history, best = train(
        train_envs, network, train_extractors,
        iterations=1500, episodes_per_batch=32, lr=1e-3,
        gamma=1.0, lam=0.95, entropy_coef=0.01, log_every=25,
        eval_fn=lambda: total_cost(eval_envs, eval_extractors, network).mean(),
    )

    final = total_cost(eval_envs, eval_extractors, network)
    print(f"\n  RL final  (iter {len(history)-1:3d}) : {final.mean():11,.0f} "
          f"+/- {final.std():9,.0f}   [{time.perf_counter() - started:.0f}s]")

    # Restore the best checkpoint before saving: the final weights are not the run's result.
    if best["state"] is not None:
        network.load_state_dict(best["state"])
        restored = total_cost(eval_envs, eval_extractors, network)
        print(f"  RL meilleur (iter {best['iteration']:3d}) : {restored.mean():11,.0f} "
              f"+/- {restored.std():9,.0f}")

    torch.save(network.state_dict(), WEIGHTS)
    print(f"  poids sauvegardes    : {WEIGHTS}")
    save_history(history, HISTORY)
    plot(history, random_cost.mean(), CURVE)


if __name__ == "__main__":
    main()