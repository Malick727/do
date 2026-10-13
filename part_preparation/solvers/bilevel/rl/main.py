import copy
import csv

import numpy as np
import torch

from part_preparation import instances
from part_preparation.instance_generator import generate_instance
from part_preparation.paths import DATA_DIR
from part_preparation.solvers.bilevel.rl import features
from part_preparation.solvers.bilevel.rl.agent import Agent
from part_preparation.solvers.bilevel.rl.environnement import Env

TRAIN_SEEDS = range(1200)
EVAL_INSTANCE = "medium"
EVAL_REPLICATES = range(90, 94)
ITERATIONS = 1500
EPISODES_PER_BATCH = 32
EVAL_EVERY = 25
WEIGHTS_DIR = DATA_DIR / "datasets" / "rl" / "weights"
WEIGHTS = WEIGHTS_DIR / "policy.pt"
CURVE = "part_preparation/solvers/bilevel/rl/training_curve.png"
HISTORY = "part_preparation/solvers/bilevel/rl/training_history.csv"


def subproblems(problems):
    envs = []
    for problem in problems:
        engine_types = sorted(set(e.engine_type for e in problem.engines_list))
        for engine_type in engine_types:
            envs.append(Env(problem, engine_type))
    return envs


def build_train_envs():
    problems = []
    for seed in TRAIN_SEEDS:
        problems.append(generate_instance(instances.varied_params(seed)))
    return subproblems(problems)


def build_eval_envs():
    problems = []
    for replicate in EVAL_REPLICATES:
        problems.append(instances.build(EVAL_INSTANCE, replicate_id=replicate))
    return subproblems(problems)


def step(env, action):
    engine_i, msn_i, factory_i = action
    engine_id = env.engines[engine_i].id
    factory_id = env.factories[factory_i].id
    msn_id = env.msns[msn_i].id
    return env.step((engine_id, factory_id, msn_id))


def random_episode(env, rng):
    state = env.reset()
    cost = 0.0
    while True:
        engine_feats, engine_idx = features.engine_candidates(env, state)
        if len(engine_idx) == 0:
            break
        engine_i = int(rng.choice(engine_idx))

        msn_feats, msn_idx = features.msn_candidates(env, state, engine_i)
        if len(msn_idx) == 0:
            break
        msn_i = int(rng.choice(msn_idx))

        factory_feats, factory_idx = features.factory_candidates(env, state, engine_i, msn_i)
        factory_i = int(rng.choice(factory_idx))

        state, reward, done, info = step(env, (engine_i, msn_i, factory_i))
        cost = cost - reward
        if done:
            break
    return cost


def per_instance(costs):
    totals = []
    for i in range(0, len(costs), 2):
        totals.append(costs[i] + costs[i + 1])
    return np.array(totals)


def evaluate(envs, agent):
    costs = []
    with torch.no_grad():
        for env in envs:
            state = env.reset()
            cost = 0.0
            while True:
                action = agent.act(env, state, deterministic=True)
                if action is None:
                    break
                state, reward, done, info = step(env, action)
                cost = cost - reward
                if done:
                    break
            costs.append(cost)
    return per_instance(costs)


def random_baseline(envs, rng):
    costs = []
    for env in envs:
        costs.append(random_episode(env, rng))
    return per_instance(costs)


def save_history(history, path):
    fields = ["iteration", "cost", "value_loss", "entropy", "eval"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow(record)


def plot(history, baseline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_iterations = []
    eval_costs = []
    for record in history:
        if record["eval"] is not None:
            eval_iterations.append(record["iteration"])
            eval_costs.append(record["eval"])

    iterations = []
    entropies = []
    for record in history:
        iterations.append(record["iteration"])
        entropies.append(record["entropy"])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    top.plot(eval_iterations, eval_costs, color="tab:red", marker="o", ms=3,
             label="argmax, hors echantillon")
    top.axhline(baseline, color="gray", ls="--", lw=1, label="aleatoire")
    top.set_ylabel("cout (euros)")
    top.legend(loc="upper right")

    bottom.plot(iterations, entropies, color="tab:green")
    bottom.axhline(1.0, color="gray", ls="--", lw=1)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("entropie normalisee")
    bottom.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(path, dpi=120)


def train_one_iteration(agent, train_envs, rng):
    costs = []
    entropies = []
    value_losses = []
    for episode in range(EPISODES_PER_BATCH):
        index = int(rng.integers(len(train_envs)))
        env = train_envs[index]
        state = env.reset()
        while True:
            action = agent.act(env, state)
            if action is None:
                break
            state, reward, done, info = step(env, action)
            agent.rewards.append(reward)
            if done:
                break
        stats = agent.finish_episode()
        if stats is not None:
            costs.append(stats["cost"])
            entropies.append(stats["entropy"])
            value_losses.append(stats["value_loss"])
    agent.update()
    return np.mean(costs), np.mean(entropies), np.mean(value_losses)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    train_envs = build_train_envs()
    eval_envs = build_eval_envs()

    agent = Agent(train_envs[0], hidden_dim=64, lr=1e-3, gamma=1.0, lam=0.95,
                  entropy_coef=0.01)
    baseline = random_baseline(eval_envs, rng).mean()

    history = []
    best_score = float("inf")
    best_weights = None

    for iteration in range(ITERATIONS):
        cost, entropy, value_loss = train_one_iteration(agent, train_envs, rng)

        score = None
        if iteration % EVAL_EVERY == 0:
            score = evaluate(eval_envs, agent).mean()
            if score < best_score:
                best_score = score
                best_weights = copy.deepcopy(agent.network.state_dict())

        history.append({"iteration": iteration, "cost": cost, "value_loss": value_loss,
                        "entropy": entropy, "eval": score})

    agent.network.load_state_dict(best_weights)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(WEIGHTS)
    agent.save_logs("a2c_without_imitation")
    save_history(history, HISTORY)
    plot(history, baseline, CURVE)
    print("eval :", round(best_score))


if __name__ == "__main__":
    main()
