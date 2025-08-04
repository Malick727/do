from multibatching.problem import MultibatchingProblem, compute_graph_for_product
from multibatching.parsing_airbus_data_ import main_parse


def main():
    problem = main_parse()
    for product in problem.products:
        if product.id == "section2":
            print("product ", product.id)
            compute_graph_for_product(problem=problem, product=product)

if __name__ == "__main__":
    main()
