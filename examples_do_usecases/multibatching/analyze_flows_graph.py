from multibatching.parse_config_data import parse_json_to_problem
from multibatching.problem import MultibatchingProblem, compute_graph_for_product
from multibatching.parsing_airbus_data_ import main_parse


def main():
    problem = main_parse()

    for product in problem.products:
        if product.id == "section2":
            print("product ", product.id)
            compute_graph_for_product(problem=problem, product=product)

def main_config():
    import os, json
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d)
    for product in problem.products:
        # print(product.name)
        # if product.id == "section2":
        #     print("product ", product.id)
        compute_graph_for_product(problem=problem, product=product)

if __name__ == "__main__":
    main_config()
