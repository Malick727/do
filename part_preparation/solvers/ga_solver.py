import random
import math
import functools
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
from ortools.sat.python import cp_model

# Assure-toi que ces imports correspondent à l'emplacement de tes fichiers
from part_preparation.problem import EnginePreparationSolution

# ==========================================
# 1. DEAP INITIALIZATION (Must be global)
# ==========================================
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# ==========================================
# 2. NEURAL NETWORK & FEATURE ENGINEERING
# ==========================================
class SiameseAllocationNet(nn.Module):
    def __init__(self, input_size):
        super(SiameseAllocationNet, self).__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, alloc_a, alloc_b):
        score_a = self.scorer(alloc_a)
        score_b = self.scorer(alloc_b)
        return torch.sigmoid(score_b - score_a)

    def predict_score(self, alloc_features):
        return self.scorer(alloc_features)


def extract_features_from_allocation(allocation_dict, problem):
    """
    Extrait le vecteur de features directement depuis le dictionnaire d'allocation 
    généré par l'Algorithme Génétique.
    """
    features = []

    # Dictionnaires pour retrouver rapidement l'index (0, 1, 2...) à partir de l'ID
    msn_to_idx = {m.id: i for i, m in enumerate(problem.msns_list)}
    factory_to_idx = {f.id: i for i, f in enumerate(problem.factories_list)}

    # On parcourt les moteurs DANS UN ORDRE FIXE
    for engine in problem.engines_list:
        data = allocation_dict.get(engine.id, {})
        msn_id = data.get("msn_id")
        factory_id = data.get("factory_id")

        # Conversion en index. -1 représente une non-allocation.
        m_idx = msn_to_idx.get(msn_id, -1.0)
        f_idx = factory_to_idx.get(factory_id, -1.0)

        # On ajoute ces deux valeurs brutes aux features
        features.extend([float(m_idx), float(f_idx)])

    return features


# ==========================================
# 3. GA OPERATORS
# ==========================================
def create_engine_genes(problem):
    msn_idx = random.randint(-1, len(problem.msns_list) - 1)
    factory_idx = random.randint(0, len(problem.factories_list) - 1)
    return [msn_idx, factory_idx]


def create_individual(problem):
    chromosome = []
    for _ in range(len(problem.engines_list)):
        chromosome.extend(create_engine_genes(problem))
    return creator.Individual(chromosome)


def cx_blocks(ind1, ind2, problem):
    nb_engines = len(problem.engines_list)
    if nb_engines < 2: return ind1, ind2
    cut_engine = random.randint(1, nb_engines - 1)
    cut_idx = cut_engine * 2
    ind1[cut_idx:], ind2[cut_idx:] = ind2[cut_idx:], ind1[cut_idx:]
    return ind1, ind2


def mut_blocks(individual, indpb, problem):
    nb_msns = len(problem.msns_list)
    nb_factories = len(problem.factories_list)

    for i in range(0, len(individual), 2):
        if random.random() < indpb:
            individual[i] = random.randint(-1, nb_msns - 1)
        if random.random() < indpb:
            individual[i + 1] = random.randint(0, nb_factories - 1)
    return individual,


def sel_rank_based(individuals, k):
    """Sélection par rang (Rank-Based Selection) privilégiée pour le Surrogate Model."""
    valid_inds = [ind for ind in individuals if ind.fitness.valid]
    sorted_inds = sorted(valid_inds, key=lambda ind: ind.fitness.values[0])

    N = len(sorted_inds)
    if N == 0:
        return []

    weights = [N - i for i in range(N)]
    chosen = random.choices(sorted_inds, weights=weights, k=k)
    return chosen


# ==========================================
# 4. ROUTING DECODER
# ==========================================
def get_allocation_from_chromosome(chromosome, problem) -> dict:
    """Extrait le routage brut depuis le chromosome en appliquant les réparations."""
    allocation = {}
    for i, engine in enumerate(problem.engines_list):
        msn_idx = chromosome[i * 2]
        factory_idx = chromosome[i * 2 + 1]

        # Réparation de l'usine
        factory = problem.factories_list[factory_idx]
        if not factory.can_prepare(engine.engine_type):
            compatible_factories = problem.engine_compatible_factories.get(engine, [])
            if compatible_factories:
                factory = random.choice(compatible_factories)

        # Réparation du MSN
        msn = None
        if msn_idx != -1:
            candidate_msn = problem.msns_list[msn_idx]
            if candidate_msn.engine_type == engine.engine_type:
                msn = candidate_msn

        allocation[engine.id] = {
            "msn_id": msn.id if msn else None,
            "factory_id": factory.id if factory else None,
            "engine_obj": engine,
            "msn_obj": msn,
            "factory_obj": factory
        }
    return allocation


# ==========================================
# 5. ÉVALUATEURS (Greedy, CP-SAT, IA)
# ==========================================

# --- MODE IA (Surrogate Model) ---
def evaluate_individual_ia(chromosome, problem, model):
    allocation = get_allocation_from_chromosome(chromosome, problem)
    features = extract_features_from_allocation(allocation, problem)
    feature_tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        score = model.predict_score(feature_tensor)

    return (float(score.item()),)


# --- MODE GREEDY ---
def decode_and_simulate_greedy(chromosome, problem) -> EnginePreparationSolution:
    allocation = get_allocation_from_chromosome(chromosome, problem)
    engine_to_msn, engine_to_factory = {}, {}
    prep_start_times, prep_end_times, fal_arrival_times, delivery_times = {}, {}, {}
    engine_jobs = []

    for engine_id, data in allocation.items():
        engine = data["engine_obj"]
        msn = data["msn_obj"]
        factory = data["factory_obj"]

        engine_to_factory[engine] = factory
        engine_to_msn[engine] = msn
        prep_duration = math.ceil(factory.get_preparation_duration(engine.engine_type))

        if msn is not None:
            assembly_line = problem.assembly_lines[msn.assembly_line_id]
            travel_time = math.ceil(problem.get_travel_time(factory, assembly_line))
            ideal_start_time = msn.engine_installation_date - travel_time - prep_duration
        else:
            travel_time = 0
            ideal_start_time = float('inf')

        engine_jobs.append({
            'engine': engine, 'factory': factory, 'msn': msn,
            'ideal_start': ideal_start_time, 'duration': prep_duration, 'travel': travel_time
        })

    engine_jobs.sort(key=lambda job: job['ideal_start'])
    factory_queues = {f: [0] * (f.default_capacity or 1) for f in problem.factories_list}

    for job in engine_jobs:
        factory, engine, msn = job['factory'], job['engine'], job['msn']
        ideal_start, prep_duration = job['ideal_start'], job['duration']

        factory_queues[factory].sort()
        delivery_time = engine.delivery_time if engine.delivery_time is not None else 0
        delivery_times[engine] = engine.delivery_time

        earliest_start = math.ceil(max(factory_queues[factory][0], delivery_time))

        if msn is not None:
            start_time = max(earliest_start, ideal_start)
            prep_start_times[engine] = start_time
            prep_end = start_time + prep_duration
            fal_arrival_times[engine] = prep_end + job['travel']
        else:
            start_time = earliest_start
            prep_start_times[engine] = start_time
            prep_end = start_time + prep_duration
            fal_arrival_times[engine] = None

        prep_end_times[engine] = prep_end
        factory_queues[factory][0] = prep_end

    return EnginePreparationSolution(
        problem=problem, engine_to_msn=engine_to_msn, engine_to_factory=engine_to_factory,
        prep_start_times=prep_start_times, prep_end_times=prep_end_times,
        fal_arrival_times=fal_arrival_times, delivery_time=delivery_times
    )


def evaluate_individual_greedy(chromosome, problem):
    solution = decode_and_simulate_greedy(chromosome, problem)
    if not problem.satisfy(solution):
        return (9999999.0,)

    kpis = problem.evaluate(solution)
    score = (
            kpis["arrival_delays"] * 1.0 +
            kpis["travel_cost"] * 1.0 +
            kpis["unallocated_msns"] * 100000.0
    )
    return (score,)


# --- MODE CP-SAT ---
def simulate_with_cpsat(chromosome, problem, return_solution=False):
    allocation = get_allocation_from_chromosome(chromosome, problem)
    model = cp_model.CpModel()
    max_time = int(problem.time_horizon + 200)

    start_vars, end_vars, interval_vars = {}, {}, {}
    delay_vars = {}

    factory_intervals = {f.id: [] for f in problem.factories_list}
    factory_demands = {f.id: [] for f in problem.factories_list}

    fixed_travel_cost = 0.0
    allocated_msns = set()

    for engine_id, data in allocation.items():
        engine = data["engine_obj"]
        msn = data["msn_obj"]
        factory = data["factory_obj"]

        if msn is None or factory is None:
            continue

        allocated_msns.add(msn.id)
        assembly_line = problem.assembly_lines[msn.assembly_line_id]

        min_start = int(round(engine.delivery_time if engine.delivery_time is not None else 0))
        S = model.NewIntVar(min_start, max_time, f"S_{engine_id}")
        E = model.NewIntVar(min_start, max_time, f"E_{engine_id}")

        prep_duration = int(round(factory.get_preparation_duration(engine.engine_type)))
        I = model.NewIntervalVar(S, prep_duration, E, f"I_{engine_id}")

        start_vars[engine_id] = S
        end_vars[engine_id] = E

        factory_intervals[factory.id].append(I)
        factory_demands[factory.id].append(1)

        travel_time = int(round(problem.get_travel_time(factory, assembly_line)))
        fixed_travel_cost += problem.get_travel_cost(factory, assembly_line)

        arrival_time = E + travel_time
        target_date = int(round(msn.engine_installation_date))

        D = model.NewIntVar(0, max_time, f"D_{engine_id}")
        model.AddMaxEquality(D, [0, arrival_time - target_date])
        delay_vars[engine_id] = D

    for f in problem.factories_list:
        intervals = factory_intervals[f.id]
        if intervals and f.default_capacity is not None:
            model.AddCumulative(intervals, factory_demands[f.id], int(f.default_capacity))

    unallocated_count = len(problem.msns_list) - len(allocated_msns)
    unallocated_penalty = unallocated_count * 100000.0

    total_delay_penalty = sum(D for D in delay_vars.values())
    model.Minimize(total_delay_penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        if not return_solution:
            total_fitness = solver.ObjectiveValue() + fixed_travel_cost + unallocated_penalty
            return (float(total_fitness),)

        engine_to_msn, engine_to_factory = {}, {}
        prep_start_times, prep_end_times, fal_arrival_times, delivery_times = {}, {}, {}, {}

        for engine_id, data in allocation.items():
            engine = data["engine_obj"]
            engine_to_msn[engine] = data["msn_obj"]
            engine_to_factory[engine] = data["factory_obj"]
            delivery_times[engine] = engine.delivery_time

            if engine_id in start_vars:
                start_t = solver.Value(start_vars[engine_id])
                end_t = solver.Value(end_vars[engine_id])
                prep_start_times[engine] = start_t
                prep_end_times[engine] = end_t

                msn = data["msn_obj"]
                factory = data["factory_obj"]
                al = problem.assembly_lines[msn.assembly_line_id]
                travel_time = problem.get_travel_time(factory, al)
                fal_arrival_times[engine] = end_t + travel_time
            else:
                prep_start_times[engine] = None
                prep_end_times[engine] = None
                fal_arrival_times[engine] = None

        return EnginePreparationSolution(
            problem=problem, engine_to_msn=engine_to_msn, engine_to_factory=engine_to_factory,
            prep_start_times=prep_start_times, prep_end_times=prep_end_times,
            fal_arrival_times=fal_arrival_times, delivery_time=delivery_times
        )
    else:
        return (9999999.0,) if not return_solution else None


def evaluate_individual_cpsat(chromosome, problem):
    return simulate_with_cpsat(chromosome, problem, return_solution=False)


# ==========================================
# 6. SOLVER CLASS
# ==========================================
class GAEngineSolver:
    """Genetic Algorithm Solver for Engine Preparation Problem"""

    def __init__(self, problem, pop_size=100, ngen=100, cxpb=0.7, mutpb=0.2, evaluation_mode="cpsat", ia_model=None):
        self.problem = problem
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ia_model = ia_model

        self.evaluation_mode = evaluation_mode
        self.toolbox = self._build_toolbox()

    def _build_toolbox(self):
        tb = base.Toolbox()
        tb.register("individual", functools.partial(create_individual, problem=self.problem))
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate", functools.partial(cx_blocks, problem=self.problem))
        tb.register("mutate", functools.partial(mut_blocks, problem=self.problem), indpb=0.1)

        if self.evaluation_mode == "cpsat":
            tb.register("evaluate", functools.partial(evaluate_individual_cpsat, problem=self.problem))
            tb.register("select", tools.selTournament, tournsize=3)

        elif self.evaluation_mode == "ia":
            if self.ia_model is None:
                raise ValueError("Le mode 'ia' requiert un modèle PyTorch (ia_model) entraîné.")
            tb.register("evaluate",
                        functools.partial(evaluate_individual_ia, problem=self.problem, model=self.ia_model))
            tb.register("select", sel_rank_based)

        else:
            tb.register("evaluate", functools.partial(evaluate_individual_greedy, problem=self.problem))
            tb.register("select", tools.selTournament, tournsize=3)

        return tb

    def solve(self):
        pop = self.toolbox.population(n=self.pop_size)
        # On garde les 10 meilleurs individus en mémoire (l'élite globale)
        hof = tools.HallOfFame(10)

        # 1. Évolution (L'IA évalue la population à la vitesse de l'éclair)
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.ngen,
            halloffame=hof,
            verbose=True
        )

        # =================================================================
        # 2. POST-PROCESSING : Le tribunal de CP-SAT
        # =================================================================
        print(f"\n[Post-Processing] Ré-évaluation des {len(hof)} meilleures solutions par CP-SAT...")

        true_best_solution = None
        best_true_score = float('inf')
        solution_history = []

        # On passe les 10 finalistes au crible du solveur exact
        for idx, ind in enumerate(hof):
            # Calcul du planning exact
            sol = simulate_with_cpsat(ind, self.problem, return_solution=True)

            # Mécanisme de sécurité si CP-SAT timeout
            if sol is None:
                sol = decode_and_simulate_greedy(ind, self.problem)

            # Évaluation des vrais KPIs mathématiques
            sol_kpis = self.problem.evaluate(sol)

            # Recalcul du score exact (même formule que le greedy/cp-sat)
            true_score = (
                    sol_kpis["arrival_delays"] * 1.0 +
                    sol_kpis["travel_cost"] * 1.0 +
                    sol_kpis["unallocated_msns"] * 100000.0
            )

            # Sauvegarde dans l'historique
            solution_history.append({
                'index': idx,
                'ia_predicted_fitness': ind.fitness.values[0],  # Ce que l'IA avait prédit
                'true_score': true_score,  # La réalité
                'kpis': sol_kpis
            })

            # Mise à jour du vrai vainqueur absolu
            if true_score < best_true_score:
                best_true_score = true_score
                true_best_solution = sol

        print(f"✅ Vrai vainqueur trouvé avec un score exact de : {best_true_score:.2f}")

        return true_best_solution, solution_history