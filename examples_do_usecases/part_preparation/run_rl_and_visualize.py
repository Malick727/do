"""
Run the RL policy on the engine preparation problem and launch the visualization dashboards.

Usage:
    uv run python examples_do_usecases/part_preparation/run_rl_and_visualize.py --instance medium
    uv run python examples_do_usecases/part_preparation/run_rl_and_visualize.py --instance xlarge --no-dashboard

The policy decides which factory prepares which engine, and which MSN it serves. The problem
splits exactly by engine type -- factories are type-exclusive, every MSN has a single type, and
the FALs carry no capacity -- so one episode runs per type and the two are merged back into a
single EnginePreparationSolution.

Same objective as every other solver here: score = travel_cost + arrival_delays, storage having
weight 0 and the coverage term being 0 by construction, since an episode ends exactly when no
engine or no MSN is left, which is the maximum matching.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import pickle
import subprocess
import time

import numpy as np
import torch

from part_preparation import instances
from part_preparation.paths import solution_file, ensure_dirs
from part_preparation.problem import aggregate_score, EnginePreparationSolution
from part_preparation.solvers.bilevel.rl.agent import (
    ActorCriticNetwork, ENGINE_FEATURES, MSN_FEATURES, FACTORY_BASE_FEATURES)
from part_preparation.solvers.bilevel.rl.environnement import Env
from part_preparation.solvers.bilevel.rl.features import FeatureExtractor
from part_preparation.solvers.bilevel.rl.training import run_episode

# Absolute, so the script runs from any working directory. A path relative to the cwd only
# worked when launched from the repository root.
DEFAULT_WEIGHTS = PROJECT_ROOT / "part_preparation/solvers/bilevel/rl/policy_cloned.pt"


def build_solution(problem, states):
    """
    Merge the per-type episode states into one EnginePreparationSolution.

    This is the piece the other run_*_and_visualize scripts do not need: a solver returns a
    Solution, an episode returns a State. The State only records the preparation START, so
    prep_end and fal_arrival are DERIVED here with the same formula the reward used -- keeping
    one definition rather than storing a second copy that could drift.

    Engines left without an MSN simply do not appear in the dicts. problem.evaluate treats a
    missing key as unallocated, and their only cost is storage, whose weight is 0.
    """
    engine_to_msn, engine_to_factory = {}, {}
    prep_start, prep_end, fal_arrival = {}, {}, {}
    delivery = {engine: engine.delivery_time for engine in problem.engines_list}

    for state in states:
        for engine_id, factory_id in state.engineToFactory.items():
            engine = problem.engines[engine_id]
            factory = problem.factories[factory_id]
            msn = problem.msns[state.engineToMsn[engine_id]]
            line = problem.assembly_lines[msn.assembly_line_id]
            duration = factory.get_preparation_duration(engine.engine_type)
            start = state.engineStart[engine_id]

            engine_to_msn[engine] = msn
            engine_to_factory[engine] = factory
            prep_start[engine] = start
            prep_end[engine] = start + duration
            fal_arrival[engine] = start + duration + problem.get_travel_time(factory, line)

    return EnginePreparationSolution(
        problem, engine_to_msn, engine_to_factory,
        prep_start, prep_end, fal_arrival, delivery)


def policy_rollout(env, extractor, network):
    """
    Deterministic rollout: argmax at each of the three levels.

    Note the vocabulary clash: run_episode's `greedy=True` means DETERMINISTIC (argmax instead
    of sampling), which is unrelated to the myopic rule below. Sampling exists for training,
    where the policy-gradient estimator needs actions drawn from pi_theta; for reporting a
    score, argmax is both better and reproducible.
    """
    with torch.no_grad():
        rewards = run_episode(env, network, extractor, greedy=True)[2]
    return env.state, rewards


def myopic_rule(env, extractor):
    """
    The one-line baseline: cheapest reachable MSN, then the factory with the best IMMEDIATE
    reward (feature 5, which is exactly what env.step will pay).

    Reported alongside the policy on purpose. Over three training seeds the policy averages
    76,018 against 69,375 for this rule on held-out medium replicates, and the spread between
    seeds is 21% -- so a policy score quoted alone, on one instance, cannot tell a good policy
    from a lucky seed.
    """
    state, rewards = env.reset(), []
    while True:
        _, engine_idx = extractor.engine_candidates(state)
        if len(engine_idx) == 0:
            break
        engine_i = int(engine_idx[0])
        msn_feats, msn_idx = extractor.msn_candidates(state, engine_i)
        if len(msn_idx) == 0:
            break
        msn_i = int(msn_idx[int(np.argmin(msn_feats[:, 2]))])
        factory_feats, factory_idx = extractor.factory_candidates(state, engine_i, msn_i)
        factory_i = int(factory_idx[int(np.argmax(factory_feats[:, 5]))])
        state, reward, done, _ = env.step((
            env.engines[engine_i].id, env.factories[factory_i].id, env.msns[msn_i].id))
        rewards.append(reward)
        if done:
            break
    return state, rewards


def solve(problem, network=None):
    """
    One episode per engine type, merged into a single solution.

    network=None runs the myopic rule instead, so both decision rules go through exactly the
    same environment, reward and solution reconstruction -- the comparison then measures the
    rule and nothing else.
    """
    states, cost = [], 0.0
    started = time.perf_counter()
    for engine_type in sorted({e.engine_type for e in problem.engines_list}):
        env = Env(problem, engine_type)
        extractor = FeatureExtractor(env)
        state, rewards = (myopic_rule(env, extractor) if network is None
                          else policy_rollout(env, extractor, network))
        states.append(state)
        cost += -sum(rewards)
    return build_solution(problem, states), cost, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description="Run the RL policy and visualize results")
    parser.add_argument('--instance', choices=sorted(instances.INSTANCES),
                        default=instances.DEFAULT_INSTANCE)
    parser.add_argument('--replicate', type=int, default=None,
                        help='Replicate id of the named instance. Use a HELD-OUT one (>= 110) '
                             'for the score to mean anything: the shipped policy was '
                             'checkpointed on replicates 90-93')
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--msns-per-fal', type=int, default=None,
                        help='Override the named instance size. Total MSNs = 4 x this value. '
                             'The horizon is derived, never given (see instances.sized_params)')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--greedy', type=bool, default=False,
                        help='Override the named instance size. Total MSNs = 4 x this value. '
                             'The horizon is derived, never given (see instances.sized_params)')
    args = parser.parse_args()

    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)

    if args.msns_per_fal is not None:
        from part_preparation.instance_generator import generate_instance
        problem = generate_instance(instances.sized_params(args.msns_per_fal))
    else:
        problem = instances.build(args.instance, replicate_id=args.replicate)

    greedy = args.greedy

    print(f"\nProblem size:")
    print(f"  - {len(problem.assembly_lines_list)} assembly lines")
    print(f"  - {len(problem.msns_list)} MSNs")
    print(f"  - {len(problem.engines_list)} engines")
    print(f"  - {len(problem.factories_list)} factories")
    print(f"  - Time horizon: {problem.time_horizon} days")
    print(f"  - Structurally uncoverable MSNs: {problem.uncoverable_msns}")

    print("\n" + "=" * 80)
    print(f"SOLVING WITH THE RL POLICY ({args.weights})")
    print("=" * 80)

    network = ActorCriticNetwork(
        engine_input_dim=ENGINE_FEATURES,
        msn_input_dim=MSN_FEATURES,
        factory_input_dim=FACTORY_BASE_FEATURES + max(
            f.default_capacity for f in problem.factories_list),
        state_input_dim=44,
        hidden_dim=64,
    )
    network.load_state_dict(torch.load(args.weights))
    network.eval()
    print(f"{sum(p.numel() for p in network.parameters()):,} parameters")

    solution, episode_cost, elapsed = solve(problem, network)
    solution2, myopic_cost, myopic_elapsed = solve(problem, None)
    
    if greedy:
        solution = solution2
    kpis = problem.evaluate(solution)
    feasible = problem.satisfy(solution)
    score = aggregate_score(kpis)
    print("\n" + "=" * 80)
    print("SOLUTION")
    print("=" * 80)
    print(f"\nFeasible: {'yes' if feasible else 'NO'}")
    # These two must match. The episode reward IS the objective, not a proxy for it; a gap means
    # the environment and problem.evaluate have drifted apart and every reported score is wrong.
    print(f"\n  - aggregate score  : {score:,.2f}")
    print(f"  - episode reward   : {episode_cost:,.2f}   (gap {abs(score - episode_cost):,.4f})")
    print(f"  - solve time       : {elapsed:.2f}s")
    print(f"\n  - myopic baseline  : {myopic_cost:,.2f}   [{myopic_elapsed:.2f}s]")
    print(f"  - RL vs myopic     : {100 * (score - myopic_cost) / myopic_cost:+.1f}%")

    print(f"\nObjective values:")
    for name, value in kpis.items():
        print(f"  - {name}: {value:.2f}")

    allocated = sum(1 for e in problem.engines_list
                    if solution.engine_to_msn.get(e) is not None)
    print(f"\nAllocation:")
    print(f"  - Allocated engines: {allocated}/{len(problem.engines_list)}")
    print(f"  - Unallocated MSNs: {kpis['unallocated_msns']:.0f}/{len(problem.msns_list)} "
          f"({problem.uncoverable_msns} structurally uncoverable)")

    ensure_dirs()
    # Suffixed so the RL run never overwrites the CP-SAT or GA pickle of the same instance --
    # the dashboards would otherwise show a solution that is not the one they name.
    # Le réplicat entre dans le nom, sinon deux réplicats d'une même instance nommée -- qui sont
    # deux problèmes différents -- écrivent dans le même pickle et le second efface le premier.
    label = (args.instance if args.msns_per_fal is None
             else f"msns{args.msns_per_fal * 4}")
    if args.replicate is not None:
        label = f"{label}_r{args.replicate}"
    label = f"{label}_rl"
    data_file = solution_file(label)
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            # One entry: unlike CP-SAT, a policy rollout has no intermediate solutions. The
            # dashboard expects the key, so it gets the final one.
            'solution_history': [{'index': 0, 'fitness': score, 'kpis': kpis}],
            # Lu par le dashboard : plusieurs solveurs ecrivent pour la meme instance.
            'solver': f"RL ({Path(args.weights).name})",
        }, f)
    print(f"\nSolution saved to: {data_file}")

    if args.no_dashboard:
        return

    print("\n" + "=" * 80)
    print("LAUNCHING DASHBOARD")
    print("=" * 80)

    dashboard_launcher = Path(__file__).parent / "launch_unified_dashboard.py"
    with open(dashboard_launcher, 'w') as f:
        f.write(f"""
import streamlit as st
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

data_file = Path(r"{data_file}")
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])
st.session_state.solver = data.get('solver', 'solveur inconnu')

from part_preparation.unified_dashboard import main
main()
""")
    print(f"Dashboard launcher created: {dashboard_launcher}")
    subprocess.run(["streamlit", "run", str(dashboard_launcher)])


if __name__ == "__main__":
    main()