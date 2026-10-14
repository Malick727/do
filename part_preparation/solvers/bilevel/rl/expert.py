import json
from pathlib import Path
from multiprocessing import Pool

import didppy as dp
import numpy as np
import torch
import torch.nn.functional as F

from part_preparation import instances
from part_preparation.instance_generator import generate_instance
from part_preparation.solvers.bilevel.dp import adapter
from part_preparation.solvers.bilevel.rl import features
from part_preparation.solvers.bilevel.rl.environnement import Env
from part_preparation.solvers.bilevel.rl.main import step


def build_problem(seed, n_msns_per_fal):
    return generate_instance(instances.varied_params(seed, n_msns_per_fal=n_msns_per_fal))


def engine_types(problem):
    return sorted({e.engine_type for e in problem.engines_list})


def position(candidates, chosen):
    return int(np.flatnonzero(candidates == chosen)[0])


def replay_cost(env, plan):
    env.reset()
    cost = 0.0
    for action in plan:
        state, reward, done, info = step(env, tuple(action))
        cost -= reward
    return cost


class DidpExpert:

    def __init__(self, time_limit=10.0, solver="CABS"):
        self.time_limit = time_limit
        self.solver = solver

    def solve(self, env):
        sub = next(s for s in adapter.split_by_engine_type(env.problem) if s.engine_type == env.engine_type)
        model = adapter.build_didp_model(env.problem, sub)
        result = getattr(dp, self.solver)(model.model, time_limit=self.time_limit, quiet=True).search()
        plan = [[e, m, f] for e, f, m, k in adapter.assignments_from_transitions(result.transitions)]
        return plan, result.cost, result.is_optimal

    def solve_instance(self, seed, n_msns_per_fal):
        problem = build_problem(seed, n_msns_per_fal)
        entries = []
        for engine_type in engine_types(problem):
            env = Env(problem, engine_type)
            plan, cost, is_optimal = self.solve(env)
            entries.append({
                "seed": seed,
                "n_msns_per_fal": n_msns_per_fal,
                "engine_type": engine_type,
                "plan": plan,
                "cost": cost,
                "is_optimal": is_optimal,
            })
        return entries

    @property
    def name(self):
        return f"didp_{self.solver.lower()}_{self.time_limit:g}s"

    def dataset_path(self, directory):
        return Path(directory) / f"{self.name}.json"

    def build_dataset(self, seeds, n_msns_per_fal, directory, processes=None):
        jobs = [(seed, n_msns_per_fal) for seed in seeds]
        if processes == 1:
            results = [self.solve_instance(*job) for job in jobs]
        else:
            with Pool(processes) as pool:
                results = pool.starmap(self.solve_instance, jobs)

        dataset = {
            "solver": self.solver,
            "time_limit": self.time_limit,
            "entries": [entry for entries in results for entry in entries],
        }
        path = self.dataset_path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(dataset, handle)
        return dataset

    def load_dataset(self, directory):
        with open(self.dataset_path(directory)) as handle:
            return json.load(handle)

    def env_of(self, entry, problems):
        key = (entry["seed"], entry["n_msns_per_fal"])
        if key not in problems:
            problems[key] = build_problem(*key)
        return Env(problems[key], entry["engine_type"])

    def demonstrations(self, dataset):
        problems = {}
        demos = []
        for entry in dataset["entries"]:
            env = self.env_of(entry, problems)
            state = env.reset()
            for engine_i, msn_i, factory_i in entry["plan"]:
                engine_feats, engine_idx = features.engine_candidates(env, state)
                msn_feats, msn_idx = features.msn_candidates(env, state, engine_i)
                factory_feats, factory_idx = features.factory_candidates(env, state, engine_i, msn_i)
                demos.append((
                    engine_feats, position(engine_idx, engine_i),
                    msn_feats, position(msn_idx, msn_i),
                    factory_feats, position(factory_idx, factory_i),
                ))
                state, reward, done, info = step(env, (engine_i, msn_i, factory_i))
        return demos

    def train(self, agent, dataset, epochs, rng):
        demos = self.demonstrations(dataset)
        net = agent.network
        history = []

        for epoch in range(epochs):
            correct = np.zeros(3)
            for i in rng.permutation(len(demos)):
                engine_feats, k_e, msn_feats, k_m, factory_feats, k_f = demos[i]

                engine_logits = net.score_engines(torch.as_tensor(engine_feats))
                msn_logits = net.score_msns(torch.as_tensor(msn_feats))
                factory_logits = net.score_factories(torch.as_tensor(factory_feats))

                loss = (F.cross_entropy(engine_logits, torch.tensor(k_e))
                        + F.cross_entropy(msn_logits, torch.tensor(k_m))
                        + F.cross_entropy(factory_logits, torch.tensor(k_f)))
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                correct += [int(engine_logits.argmax()) == k_e,
                            int(msn_logits.argmax()) == k_m,
                            int(factory_logits.argmax()) == k_f]

            history.append(correct / max(len(demos), 1))
        return history
