import argparse
import copy

import numpy as np
import torch

from part_preparation import instances
from part_preparation.instance_generator import generate_instance
from part_preparation.paths import DATA_DIR
from part_preparation.solvers.bilevel.rl.agent import Agent
from part_preparation.solvers.bilevel.rl.expert import DidpExpert
from part_preparation.solvers.bilevel.rl.main import evaluate, step, subproblems

TRAIN_SEEDS = range(60)
N_MSNS_PER_FAL = 40
EXPERT_TIME_LIMIT = 10.0
IMITATION_DIR = DATA_DIR / "datasets" / "imitation"
EXPERT = DidpExpert(time_limit=EXPERT_TIME_LIMIT)
TAG = EXPERT.name
WEIGHTS_DIR = IMITATION_DIR / "weights"
EVAL_INSTANCE = "medium"
SELECT_REPLICATES = range(90, 94)
TEST_REPLICATES = range(110, 130)
IMITATION_EPOCHS = 12
WARMUP_ITERATIONS = 30
WARMUP_EPISODES = 4
FINETUNE_ITERATIONS = 150
FINETUNE_EPISODES = 8


def build_train_envs():
    problems = []
    for seed in TRAIN_SEEDS:
        params = instances.varied_params(seed, n_msns_per_fal=N_MSNS_PER_FAL)
        problems.append(generate_instance(params))
    return subproblems(problems)


def build_eval_envs(replicates):
    problems = []
    for replicate in replicates:
        problems.append(instances.build(EVAL_INSTANCE, replicate_id=replicate))
    return subproblems(problems)


def random_env(envs, rng):
    index = int(rng.integers(len(envs)))
    return envs[index]


def warm_up_critic(agent, train_envs, rng):
    agent.phase = "warmup"
    agent.set_actor_trainable(False)
    agent.optimizer = torch.optim.Adam(agent.critic_params, lr=1e-3)
    for iteration in range(WARMUP_ITERATIONS):
        for episode in range(WARMUP_EPISODES):
            env = random_env(train_envs, rng)
            state = env.reset()
            while True:
                action = agent.act(env, state)
                if action is None:
                    break
                state, reward, done, info = step(env, action)
                agent.rewards.append(reward)
                if done:
                    break
            agent.finish_episode()
        agent.update()
    agent.set_actor_trainable(True)


def fine_tune(agent, train_envs, select_envs, rng):
    agent.phase = "finetune"
    agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-4)
    agent.entropy_coef = 0.0

    best_score = evaluate(select_envs, agent).mean()
    best_weights = copy.deepcopy(agent.network.state_dict())

    for iteration in range(FINETUNE_ITERATIONS):
        for episode in range(FINETUNE_EPISODES):
            env = random_env(train_envs, rng)
            state = env.reset()
            while True:
                action = agent.act(env, state)
                if action is None:
                    break
                state, reward, done, info = step(env, action)
                agent.rewards.append(reward)
                if done:
                    break
            agent.finish_episode()
        agent.update()

        if iteration % 10 == 0:
            score = evaluate(select_envs, agent).mean()
            if score < best_score:
                best_score = score
                best_weights = copy.deepcopy(agent.network.state_dict())

    agent.network.load_state_dict(best_weights)


def save_weights(agent, stage):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    state = agent.network.state_dict()
    policy = {name: value for name, value in state.items() if not name.startswith("value")}
    critic = {name: value for name, value in state.items() if name.startswith("value")}
    torch.save(policy, WEIGHTS_DIR / f"policy_{stage}_{TAG}.pt")
    torch.save(critic, WEIGHTS_DIR / f"critic_{stage}_{TAG}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "train"])
    mode = parser.parse_args().mode

    expert = EXPERT
    if mode == "generate":
        expert.build_dataset(TRAIN_SEEDS, N_MSNS_PER_FAL, IMITATION_DIR)
        return

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    train_envs = build_train_envs()
    select_envs = build_eval_envs(SELECT_REPLICATES)
    test_envs = build_eval_envs(TEST_REPLICATES)

    agent = Agent(train_envs[0], hidden_dim=64, lr=1e-3)
    dataset = expert.load_dataset(IMITATION_DIR)

    expert.train(agent, dataset, IMITATION_EPOCHS, rng)
    warm_up_critic(agent, train_envs, rng)
    save_weights(agent, "imitation")

    fine_tune(agent, train_envs, select_envs, rng)
    save_weights(agent, "final")
    agent.save_logs(f"a2c_with_imitation_{TAG}")

    print("test :", round(evaluate(test_envs, agent).mean()))


if __name__ == "__main__":
    main()
