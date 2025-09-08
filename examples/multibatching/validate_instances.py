from multibatching.problem import MultibatchingProblem, MultibatchingSolution
from multibatching.parse_config_data import parse_json_to_problem, parse_config_file_to_problem
import os


def validate():
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    problem = parse_config_file_to_problem(config_file)

    check = problem.check_feasibility_per_product(verbose=True)
    print(check)


if __name__ == "__main__":
    validate()