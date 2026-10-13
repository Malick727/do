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
from part_preparation.solvers.bilevel.rl import features
from part_preparation.solvers.bilevel.rl.agent import Agent
from part_preparation.solvers.bilevel.rl.environnement import Env
from part_preparation.solvers.bilevel.rl.main import step
from part_preparation.solvers.bilevel.rl.main_imitation import TAG, WEIGHTS_DIR

DEFAULT_WEIGHTS = WEIGHTS_DIR / f"policy_final_{TAG}.pt"


def build_solution(problem, states):
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


def policy_rollout(env, agent):
    state, cost = env.reset(), 0.0
    with torch.no_grad():
        while True:
            action = agent.act(env, state, deterministic=True)
            if action is None:
                break
            state, reward, done, info = step(env, action)
            cost -= reward
            if done:
                break
    return state, cost


def myopic_rule(env):
    state, cost = env.reset(), 0.0
    while True:
        _, engine_idx = features.engine_candidates(env, state)
        if len(engine_idx) == 0:
            break
        engine_i = int(engine_idx[0])
        msn_feats, msn_idx = features.msn_candidates(env, state, engine_i)
        if len(msn_idx) == 0:
            break
        msn_i = int(msn_idx[int(np.argmin(msn_feats[:, 2]))])
        factory_feats, factory_idx = features.factory_candidates(env, state, engine_i, msn_i)
        factory_i = int(factory_idx[int(np.argmax(factory_feats[:, 5]))])
        state, reward, done, _ = step(env, (engine_i, msn_i, factory_i))
        cost -= reward
        if done:
            break
    return state, cost


def solve(problem, agent=None):
    states, cost = [], 0.0
    started = time.perf_counter()
    for engine_type in sorted({e.engine_type for e in problem.engines_list}):
        env = Env(problem, engine_type)
        state, episode_cost = (myopic_rule(env) if agent is None
                               else policy_rollout(env, agent))
        states.append(state)
        cost += episode_cost
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
    parser.add_argument('--greedy', action='store_true',
                        help='Save and display the myopic greedy rule instead of the RL policy')
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

    first_type = sorted({e.engine_type for e in problem.engines_list})[0]
    agent = Agent(Env(problem, first_type), hidden_dim=64)
    agent.network.load_state_dict(torch.load(args.weights), strict=False)
    agent.network.eval()
    print(f"{sum(p.numel() for p in agent.network.parameters()):,} parameters")

    solution, episode_cost, elapsed = solve(problem, agent)
    solution2, myopic_cost, myopic_elapsed = solve(problem, None)
    
    method = "greedy" if greedy else "rl"
    if greedy:
        solution, episode_cost, elapsed = solution2, myopic_cost, myopic_elapsed
    kpis = problem.evaluate(solution)
    feasible = problem.satisfy(solution)
    score = aggregate_score(kpis)
    print("\n" + "=" * 80)
    print("SOLUTION")
    print("=" * 80)
    print(f"\nFeasible: {'yes' if feasible else 'NO'}")
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
    label = (args.instance if args.msns_per_fal is None
             else f"msns{args.msns_per_fal * 4}")
    if args.replicate is not None:
        label = f"{label}_r{args.replicate}"
    label = f"{label}_{method}"
    data_file = solution_file(label)
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': [{'index': 0, 'fitness': score, 'kpis': kpis}],
            'solver': "Greedy (myopic rule)" if greedy else f"RL ({Path(args.weights).name})",
        }, f)
    print(f"\nSolution saved to: {data_file}")

    if args.no_dashboard:
        return

    print("\n" + "=" * 80)
    print("LAUNCHING DASHBOARD")
    print("=" * 80)

    dashboard_launcher = Path(__file__).parent / f"launch_{method}_dashboard.py"
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