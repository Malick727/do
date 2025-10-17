import os
import pickle
from multibatching.analyse_solutions import compare_solutions_visually, calculate_multibatch_percentage, compare_solutions_by_route
from multibatching.problem import MultibatchingSolution

this_folder = os.path.dirname(os.path.abspath(__file__))
res_data_folder = os.path.join(this_folder, "res_data/")


def basic_comparison():
    with open(os.path.join(res_data_folder, "solution_single_batch.pk"), "rb") as f:
        single_batch = pickle.load(f)
    with open(os.path.join(res_data_folder, "solution_multibatching.pk"), "rb") as f:
        multi_batch = pickle.load(f)
    multi_batch: MultibatchingSolution = pickle.load(open("solution_multibatching_lazy.pk", "rb"))
    for f in multi_batch.list_flows:
        print([f.product_packing[p] for p in f.product_packing], f.transport_link.transport_type.name)
        print("Capacity", f.transport_link.transport_type.capacity)
        print("Size", [(p.name, p.size) for p in f.product_packing])
    calculate_multibatch_percentage(single_batch)
    calculate_multibatch_percentage(multi_batch)
    compare_solutions_visually(singlebatch_solution=single_batch,
                               multibatch_solution=multi_batch)

    compare_solutions_by_route(singlebatch_solution=single_batch,
                               multibatch_solution=multi_batch)


if __name__ == "__main__":
    basic_comparison()



