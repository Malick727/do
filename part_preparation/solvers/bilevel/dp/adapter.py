import math
import re
from dataclasses import dataclass

import didppy as dp

from part_preparation.problem import EnginePreparationProblem, EnginePreparationSolution
from part_preparation.solvers.bilevel.dp.model import DidpModel

TRANSITION_NAME = re.compile(r"assign e=(\d+) f=(\d+) m=(\d+) k=(\d+)")
SELECT_ENGINE = re.compile(r"select_engine e=(\d+)")
SELECT_MSN = re.compile(r"select_msn m=(\d+)")
SELECT_FACTORY = re.compile(r"select_factory f=(\d+) k=(\d+)")


@dataclass
class TypeSubproblem:
    engine_type: str
    engines: list
    factories: list
    msns: list


def split_by_engine_type(problem: EnginePreparationProblem) -> list[TypeSubproblem]:
    for factory in problem.factories_list:
        if len(factory.compatible_engine_types) != 1:
            raise ValueError(f"Factory {factory.id} prepares {factory.compatible_engine_types}, types are no longer independent")

    engine_types = sorted({e.engine_type for e in problem.engines_list})
    subproblems = []
    for engine_type in engine_types:
        subproblem = TypeSubproblem(
            engine_type=engine_type,
            engines=[e for e in problem.engines_list if e.engine_type == engine_type],
            factories=[f for f in problem.factories_list if f.can_prepare(engine_type)],
            msns=[m for m in problem.msns_list if m.engine_type == engine_type],
        )
        if not subproblem.factories:
            raise ValueError(f"No factory can prepare {engine_type}")
        subproblems.append(subproblem)
    return subproblems


def factory_prep_duration(factory):
    durations = {float(d) for d in factory.preparation_durations.values()}
    if len(durations) != 1:
        raise ValueError(f"Factory {factory.id} has several preparation durations {durations}, the DIDP model expects one p_f")
    return durations.pop()


def build_didp_model(problem: EnginePreparationProblem, sub: TypeSubproblem, multi_phase=True) -> DidpModel:
    msn_lines = [problem.assembly_lines[msn.assembly_line_id] for msn in sub.msns]

    didp_model = DidpModel(
        nb_engine=len(sub.engines),
        nb_factory=len(sub.factories),
        nb_msn=len(sub.msns),
        release_engine=[float(e.delivery_time or 0) for e in sub.engines],
        msn_inst_date=[float(msn.engine_installation_date) for msn in sub.msns],
        factory_msn_distance=[
            [float(problem.get_travel_time(f, line)) for line in msn_lines] for f in sub.factories
        ],
        factory_capacity=[f.default_capacity or 1 for f in sub.factories],
        factory_prep_duration=[factory_prep_duration(f) for f in sub.factories],
        factory_msn_travel_cost=[
            [float(problem.get_travel_cost(f, line)) for line in msn_lines] for f in sub.factories
        ],
        engine_storage_cost=[float(e.engine_type_object.cost_storage_per_time) for e in sub.engines],
        engine_msn_compatible=[[True] * len(sub.msns) for _ in sub.engines],
        engine_factory_compatible=[[True] * len(sub.factories) for _ in sub.engines],
        horizon=float(problem.time_horizon),
    )
    didp_model.add_tables()
    didp_model.add_variables()
    if multi_phase:
        didp_model.add_multi_phase_transition()
    else:
        didp_model.add_transition()
    return didp_model


def assignments_from_transitions(transitions):
    engine, msn = None, None
    for transition in transitions:
        name = transition.name
        if match := TRANSITION_NAME.fullmatch(name):
            yield tuple(map(int, match.groups()))
        elif match := SELECT_ENGINE.fullmatch(name):
            engine = int(match.group(1))
        elif match := SELECT_MSN.fullmatch(name):
            msn = int(match.group(1))
        elif match := SELECT_FACTORY.fullmatch(name):
            f, k = map(int, match.groups())
            yield engine, f, msn, k
            engine, msn = None, None
        else:
            raise ValueError(f"Unknown transition name: {name!r}")


def replay_transitions(sub: TypeSubproblem, didp_model: DidpModel, transitions, solution: EnginePreparationSolution):
    station_free = [[0.0] * capacity for capacity in didp_model.factory_capacity]

    for e, f, m, k in assignments_from_transitions(transitions):
        d_m = didp_model.msn_inst_date[m]
        theta = didp_model.factory_msn_distance[f][m]
        p_f = didp_model.factory_prep_duration[f]
        r_e = didp_model.release_engine[e]

        start = max(d_m - theta - p_f, math.ceil(station_free[f][k]), r_e)
        station_free[f][k] = start + p_f

        engine = sub.engines[e]
        solution.engine_to_msn[engine] = sub.msns[m]
        solution.engine_to_factory[engine] = sub.factories[f]
        solution.prep_start_times[engine] = start
        solution.prep_end_times[engine] = start + p_f
        solution.fal_arrival_times[engine] = start + p_f + theta


def empty_solution(problem: EnginePreparationProblem) -> EnginePreparationSolution:
    engines = problem.engines_list
    return EnginePreparationSolution(
        problem=problem,
        engine_to_msn={engine: None for engine in engines},
        engine_to_factory={engine: None for engine in engines},
        prep_start_times={engine: None for engine in engines},
        prep_end_times={engine: None for engine in engines},
        fal_arrival_times={engine: None for engine in engines},
        delivery_time={engine: engine.delivery_time for engine in engines},
    )


def solve_with_didp(problem: EnginePreparationProblem, solver_class=dp.CABS, time_limit=60, quiet=True, multi_phase=True, **solver_kwargs):
    subproblems = split_by_engine_type(problem)
    time_per_type = time_limit / len(subproblems)
    solution = empty_solution(problem)
    results = {}

    for sub in subproblems:
        didp_model = build_didp_model(problem, sub, multi_phase=multi_phase)
        result = solver_class(didp_model.model, time_limit=time_per_type, quiet=quiet, **solver_kwargs).search()
        results[sub.engine_type] = result
        if result.is_infeasible or not result.transitions:
            return None, results
        replay_transitions(sub, didp_model, result.transitions, solution)

    return solution, results
