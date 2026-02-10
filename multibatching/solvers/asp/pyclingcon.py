
# coding: utf-8
import argparse
import gc
import clingo
from clingo.control import Control
from clingo.backend import Observer
from clingo import Control
from clingcon import ClingconTheory


import matplotlib
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import threading
import sys
import plotly.express as px
from clingo.ast import parse_files, parse_string, ProgramBuilder
from ngo import optimize, auto_detect_input, auto_detect_output

import helper_functions as hf

import subprocess
import shlex
import os
import plotly.graph_objects as go
import time 
import shutil
import csv
import json
import pickle
#import compute_similarity
#import create_best_model_dataframe

#import create_best_model_dataframe

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

YELLOW = "\033[33m"
RESET = "\033[0m"  

showModel = False
showDataframes = False
print_details = True
foundModelIndex = 0
analyzedModelIndex = 0
all_dataframes = {}
costs = []

modelAggregateTemplates = [{'name':'flow_out', 'filter':lambda s: s.name=='flow_out', 'columns':['product' ,'location', 'amount']},
                           {'name':'flow_in', 'filter':lambda s: s.name=='flow_in', 'columns':['product' ,'location', 'amount']},
                         {'name':'req', 'filter':lambda s: s.name=='req', 'columns':['product' ,'location', 'amount']},
                          {'name':'route_cost_total', 'filter':lambda s: s.ame=='route_cost_total', 'columns':['from' ,'to', 'transportresource', 'cost']},
                        {'name':'route_freq', 'filter':lambda s: s.name=='estimated_freq', 'columns':['from' ,'to', 'transportresource', 'freq', 'totalcapacity']},
                        {'name':'load', 'filter':lambda s: s.name=='load', 'columns':['from' ,'to', 'transportresource', 'product', 'amount']},
                        {'name':'total_weight', 'filter':lambda s: s.name=='total_weight', 'columns':['from' ,'to', 'transportresource', 'weight']}
]
 
modelAtomTemplates = [
        {'name':'totalSupply', 'filter':lambda s: s.name=='totalSupply', 'columns':['product' ,'amount']},
        {'name':'route', 'filter':lambda s: s.name=='route', 'columns':['idx','from' ,'to', 'transportresource', 'distance','cost']},
        {'name':'routeCostTotal', 'filter':lambda s: s.name=='routeCostTotal', 'columns':['from' ,'to', 'transportresource', 'cost']},
        {'name':'routeFreq','filter':lambda s: s.name=='routeFreq', 'columns':['from', 'to', 'transportresource','freq','maxcapacity']}, #,'maxcapacity', 'minus1'
        {'name':'activeFlow','filter':lambda s: s.name=='activeFlow', 'columns':['from', 'to', 'transportresource']},
        {'name':'demandOffer','filter':lambda s: s.name=='demandOffer', 'columns':['product', 'location', 'amount']},
        #{'name':'requiredNet','filter':lambda s: s.name=='requiredNet', 'columns':['product', 'location', 'amount']},
        {'name':'productTR','filter':lambda s: s.name=='productTR', 'columns':['product', 'transportresource']},
        {'name':'productSize','filter':lambda s: s.name=='productSize', 'columns':['product', 'size']},
        {'name':'possibleFlow','filter':lambda s: s.name=='possibleFlow', 'columns':['from', 'to', 'transportresource','product']},
        {'name':'transportCapacity','filter':lambda s: s.name=='transportCapacity', 'columns':['transportresource', 'capacity']},
        {'name':'productOnFlow','filter':lambda s: s.name=='productOnFlow', 'columns':['from', 'to', 'transportresource','product','amount']}
    ]


def cprint(Text):
    global print_details
    if print_details:
        print(Text)


def solve_clingcon(programs, max_models=0, ngo=False):
    global start_solving, produced_at_atoms, ctl, timeout
    print('Configuring Clingcon ...')
    print(f'--models={max_models}')

    prg = hf.read_files_to_string(programs)

    thy = ClingconTheory()
    # , "--parallel-mode=12"
    clingo_args = [f"--models={max_models}", "--opt-mode=opt"]

    ctl = clingo.Control(clingo_args)
    ctl.configuration.solve.models = 0  # Search for better models
    #ctl.configuration.solver.opt_strategy = "bb"
    #ctl = Control(["--opt-mode=opt", "--heuristic=Domain","--opt-strategy=usc"]) #, f"--models={max_models}"
    print(f'Configuration Keys: {ctl.configuration.solve.keys}')
    #conf = ctl.configuration.solve
    #print("Solver Config:", ", ".join([f"{key}: {getattr(conf, key)}" for key in conf.keys]))
    print("programs", programs)
    ctl.register_observer(myobs)
    myobs.start()
    thy.register(ctl)
    with ProgramBuilder(ctl) as bld:
        parse_string(prg, lambda ast: thy.rewrite_ast(ast, bld.add))

    start_time_grounding = time.process_time()
    print(f"=== Grounding programs: {programs} ... ===")
    ctl.ground([('base', [])])
    current_best = 1000000
    optimum_model = None

    ctl.configuration.solve.opt_mode = "opt"
    
    end_time_grounding = time.process_time()
    print(f"=== Grounding done ({round((end_time_grounding - start_time_grounding) / 60, 2)} minutes, {myobs.number_rules} rules, {myobs.number_choice_rules} choice rules) ===")
    
    thy.prepare(ctl)
    start_solving = time.process_time()
    
    with ctl.solve(yield_=True, async_=True, on_model=on_model) as hnd:
        for idx, mdl in enumerate(hnd):
            #print(mdl)
            assignment_data = list(thy.assignment(mdl.thread_id))
            print("aggregates selection:")
            print([f'{key}={val}' for key, val in thy.assignment(mdl.thread_id) if key.name == "route_freq" and val!=0])
            processAggregates(assignment_data, idx)   
            processModel(idx)         
    ctl.cleanup()

def processAggregates(assignment_data, idx):
    for modelAggregateTemplate in modelAggregateTemplates:
        showModelAggregates(
            assignment_data=assignment_data,
            dataName=modelAggregateTemplate["name"], 
            columns=modelAggregateTemplate["columns"],
            index=idx
        )
    
def showModelAggregates(assignment_data, dataName, columns, index):
    filtered_data = [[*key.arguments, val] for key, val in assignment_data if key.name==dataName and val!=0]
    df = pd.DataFrame(data=filtered_data, columns=columns)
    all_dataframes[dataName] = df
    output_dir = f"{outputfolder}/model_{index}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{dataName}.csv", index=False)

def check_productTR(modelIndex):
    productTR_df_str = all_dataframes["productTR"].astype(str) 
    productTR_dict = list(zip(productTR_df_str['product'], productTR_df_str['transportresource']))

    load_df = all_dataframes["load"].astype(str) 
    
    count = 0
    for row in load_df.itertuples():
        if  (row.product,row.transportresource) not in productTR_dict:
            print(f"Invalid product / transport resource pair: {row.product,row.transportresource }")
            count += 1
    
    if count==0:
        print(f"Model {modelIndex}: Product / transport resource pairs seem consistent.")
    else:
        print(f"Model {modelIndex}: {count} # invalid product/ transport resource.")
        sys.exit()
    
def check_clingcon_flow(modelIndex):
    demand_supply_df = all_dataframes["demandOffer"]
    req_df = demand_supply_df.copy()
    req_df["amount"] = req_df["amount"].apply(to_int) * -1

    flow_in = all_dataframes["flow_in"]
    flow_out = all_dataframes["flow_out"]

    in_and_out = pd.merge(
    flow_in, 
    flow_out, 
    on=['product', 'location'], 
    how='outer', 
    suffixes=('_in', '_out')
    )

    final_merged_df = pd.merge(
    in_and_out, 
    req_df, 
    on=['product', 'location'], 
    how='outer', 
    suffixes=('_merged', '_net_req')
    )   

    final_merged_df['net_flow'] = final_merged_df['amount_in'] - final_merged_df['amount_out']
    final_merged_df['remaining_req'] = final_merged_df['amount'] - final_merged_df['net_flow']
    
    pd.set_option('future.no_silent_downcasting', True)
    final_merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_merged_df = final_merged_df.fillna(0)
    cols_to_convert = ["net_flow",'amount_in',"amount_out","remaining_req"]
    final_merged_df[cols_to_convert] = final_merged_df[cols_to_convert].astype(int)


    is_consistent = np.all(np.isclose(final_merged_df['remaining_req'], 0))
    if is_consistent != True:
            inconsistent_rows = final_merged_df[~np.isclose(final_merged_df['remaining_req'], 0)]
            print(f"{YELLOW}============================== Model {modelIndex} does not seem corect",  f"=============================={RESET}")
            print("inconsistent_rows")
            print(inconsistent_rows)
            print("all")
            print(final_merged_df)
            sys.exit('exit')
    else:
        print(f"Model {modelIndex}: Flow seems consistent.")

def check_clingcon_load(modelIndex):

    """
    Verifies that aggregated loads (from/to) match the net requirements at each location.
    """
    
    demand_supply_df = all_dataframes["demandOffer"]
    req_df = demand_supply_df.copy()
    req_df["amount"] = req_df["amount"].apply(to_int) * -1
    
    load_df = all_dataframes["load"]

    # 1. Aggregate loads entering locations (Group by product, To)
    load_in_agg = load_df.groupby(['product', 'to'])['amount'].sum().reset_index()
    load_in_agg.rename(columns={'to': 'location', 'amount': 'amount_in_calc'}, inplace=True)

    # 2. Aggregate loads leaving locations (Group by product, From)
    load_out_agg = load_df.groupby(['product', 'from'])['amount'].sum().reset_index()
    load_out_agg.rename(columns={'from': 'location', 'amount': 'amount_out_calc'}, inplace=True)

    # 3. Merge In and Out aggregates
    flow_calc = pd.merge(
        load_in_agg, 
        load_out_agg, 
        on=['product', 'location'], 
        how='outer'
    ).fillna(0)

    # 4. Calculate Net Flow (In - Out) derived purely from loads
    flow_calc['net_flow_calc'] = flow_calc['amount_in_calc'] - flow_calc['amount_out_calc']

    # 5. Merge with Requirements (Demand/Supply)
    # We use suffixes to distinguish the req amount from our calculated flow
    check_df = pd.merge(
        flow_calc,
        req_df,
        on=['product', 'location'],
        how='outer',
        suffixes=('_calc', '_req')
    ).fillna(0)
    
    # 6. Verify: Remaining Req should be 0 (Req - NetFlow == 0)
    # Note: Depending on your sign convention in ASP:
    # If Req = Demand (Pos) - Offer (Pos), then Net Flow (In - Out) should equal Req.
    check_df['balance_check'] = check_df['amount'] - check_df['net_flow_calc']

    is_flow_consistent = np.all(np.isclose(check_df['balance_check'], 0))

    if not is_flow_consistent:
        print(f"{YELLOW}========== Model {modelIndex}: Load Aggregation Mismatch =========={RESET}")
        inconsistent = check_df[~np.isclose(check_df['balance_check'], 0)]
        print("Inconsistent Location/product (Load vs Req):")
        # Displaying relevant columns for debugging
        print(inconsistent[['product', 'location', 'amount', 'net_flow_calc', 'balance_check']])
        #sys.exit('exit')
    else:
        print(f"Model {modelIndex}: Loads seem consistent.")

def check_routeFreq(modelIndex):
    """
    Checks coherence between route capacities and actual loads.
    """

    load_df = all_dataframes["load"]
    routeFreq_df = all_dataframes["routeFreq"]
    routeFreq_df['freq'] = routeFreq_df['freq'].apply(to_int)
    productSize_df = all_dataframes["productSize"]
    productSize_df['size'] = productSize_df['size'].apply(to_int)
    transportCapacity_df = all_dataframes["transportCapacity"]
    transportCapacity_df['capacity'] = transportCapacity_df['capacity'].apply(to_int)
    
    load_enriched = pd.merge(
        load_df,
        productSize_df,
        on='product',
        how='left'
    )
    
    load_enriched['calculated_load'] = load_enriched['amount'] * load_enriched['size']

    # 1. Aggregate the load_df to get total weight per route
    # We group by the route keys and sum the 'amount'
    load_agg = load_enriched.groupby(['from', 'to', 'transportresource'])['calculated_load'].sum().reset_index()
    load_agg.rename(columns={'calculated_load': 'total_load'}, inplace=True)
    
    # 2. Merge the aggregated load with the route frequency dataframe
    # We use 'left' merge to ensure we keep all routes defined in route_freq_df
    merged_df = pd.merge(
        routeFreq_df, 
        load_agg, 
        on=['from', 'to', 'transportresource'], 
        how='left'
    )
    
    # Fill NaN values in 'total_load' with 0 (for routes that have no products assigned)
    merged_df['total_load'] = merged_df['total_load'].fillna(0)
    
    route_full = pd.merge(
        routeFreq_df,
        transportCapacity_df,
        on='transportresource',
        how='left'
    )
    
    # Clean types
    route_full['freq'] = route_full['freq'].apply(to_int)
    route_full['capacity'] = route_full['capacity'].fillna(0)
    
    # Calculate Total Capacity for the route
    route_full['totalcapacity'] = route_full['freq'] * route_full['capacity']

    merged_df = pd.merge(
        route_full, 
        load_agg, 
        on=['from', 'to', 'transportresource'], 
        how='left'
    )
    # 3. Check Upper Bound (Constraint: Load <= Total Capacity)
    merged_df['upper_bound_ok'] = merged_df['total_load'] <= merged_df['totalcapacity']
    
    # 4. Check Lower Bound (Optimality: Is freq the lowest needed?)
    # Logic: Calculate Unit Capacity (Capacity per trip) -> Calculate Required Freq -> Compare
    
    # Avoid division by zero if freq is 0
    # We create a mask where freq > 0 to perform calculations
    mask_active = merged_df['freq'] > 0
    
    # Initialize columns
    merged_df['unit_capacity'] = 0.0
    merged_df['optimal_freq'] = 0.0
    
    # Calculate Unit Capacity (Total Capacity / Frequency)
    merged_df.loc[mask_active, 'unit_capacity'] = (
        merged_df.loc[mask_active, 'totalcapacity'] / merged_df.loc[mask_active, 'freq']
    )
    
    # Calculate Optimal Frequency: ceil(Total Load / Unit Capacity)
    # We use np.ceil to round up to the nearest whole trip required
    # Note: If unit_capacity is 0 (which shouldn't happen if freq > 0 and totalcapacity > 0), handle carefully.
    # Assuming totalcapacity > 0 if freq > 0.
    merged_df.loc[mask_active, 'optimal_freq'] = np.ceil(
        merged_df.loc[mask_active, 'total_load'] / merged_df.loc[mask_active, 'unit_capacity']
    )
    
    merged_df["optimal_freq"] = merged_df["optimal_freq"].astype(int)

    # If freq was 0, optimal freq is 0 (assuming load is 0, otherwise it's impossible)
    
    # The Check: Does the actual frequency match the calculated optimal frequency?
    merged_df['lower_bound_ok'] = merged_df['freq'] == merged_df['optimal_freq']
    merged_df.loc[merged_df['freq'] == 0, 'lower_bound_ok'] = merged_df.loc[merged_df['freq'] == 0, 'total_load'] == 0


    # 5. Filter for Errors
    # We keep rows where EITHER bound is violated
    error_mask = (~merged_df['upper_bound_ok']) | (~merged_df['lower_bound_ok'])
    error_report = merged_df[error_mask].copy()
    
    # 6. Label the Error Type
    # If Upper Bound failed -> 'Overloaded' (Critical)
    # If Upper Bound OK but Lower Bound failed -> 'Inefficient' (Wasted Capacity)
    conditions = [
        ~error_report['upper_bound_ok'], 
        (error_report['upper_bound_ok']) & (~error_report['lower_bound_ok'])
    ]
    choices = ['Overloaded (Critical)', 'Inefficient (Excess Freq)']
    
    error_report['issue_type'] = np.select(conditions, choices, default='Unknown')
    
    # Select clean columns for output
    cols_to_show = [
        'from', 'to', 'transportresource', 
        'issue_type', 
        'freq', 'optimal_freq', 
        'total_load', 'totalcapacity'
    ]
    
    if error_report[cols_to_show].empty:
        print(f"Model {modelIndex}: Upper bound and lower bound satisfied: No overload, no inefficiency. ")
    else:
        print(f"Model {modelIndex}: {error_report[cols_to_show]}.")

def to_int(val):
    # If it's a Clingo Symbol (has 'number' attribute), extract it
    if hasattr(val, 'number'): 
        return val.number
    # If it's already an int or float, return as is
    return val

def compute_costs(modelIndex):
    global all_dataframes
        
    active_flows_df = all_dataframes["activeFlow"]
    print(f"Model {modelIndex}: Total active flows: {active_flows_df.shape[0]}")
    route_df = all_dataframes["route"]
    route_df['cost'] = route_df['cost'].values
    route_df['cost'] = route_df['cost'].apply(to_int)
    route_freq_df = all_dataframes["routeFreq"]
    route_freq_df['freq'] = route_freq_df['freq'].values
    route_freq_df['freq'] = route_freq_df['freq'].apply(to_int)
    # only keep the routes in route_freq
    merged_df = pd.merge(
        route_freq_df, 
        route_df, 
        on=['from', 'to', 'transportresource'], 
        how='inner'
    )

    merged_df['total_cost'] = merged_df['freq'] * merged_df['cost']
    print(f"Model {modelIndex}: Total costs are {merged_df['total_cost'].sum()}")



def processModel(modelIndex):
    global check_clingcon_multibatching, compute_costs_var
    if check_clingcon_multibatching == True:
        cprint(f"Model {modelIndex}: Check clingcon flow, load and product/TR consistency ...")
        check_clingcon_flow(modelIndex)
        check_clingcon_load(modelIndex)
        check_productTR(modelIndex)
        check_routeFreq(modelIndex)
    if compute_costs_var == True:   
        compute_costs(modelIndex)
        cprint(f"Model {modelIndex}: Done. ")

def showModelAtoms(s, dataName, symbolFilter, columns, index):
    global iD_conversion

    filteredSymbols = list(filter(symbolFilter, s))
    arguments = list(map(lambda s: s.arguments, filteredSymbols))
    df = pd.DataFrame(data=arguments, columns=columns)
    all_dataframes[dataName] = df

    df.to_csv(f"{outputfolder}/model_{index}/{dataName}.csv", index=False)
    if showDataframes:
        print(f"========================== {dataName} ==========================")
        print(df.to_string())


def on_model(m: clingo.Model):
    global start_solving, output_hundredth_model, foundModelIndex, analyzedModelIndex, outputfolder, exit_after_optimal_found, draw, compute_costs_var

    if not output_hundredth_model or foundModelIndex % 100 == 0:
        cprint(f"=== New Model [{foundModelIndex}] found after {round((time.process_time() - start_solving) / 60, 2)} minutes of solving (Optimality proven: {m.optimality_proven}) === ")
        
        hf.emptyFolder(f"./{outputfolder}/model_{analyzedModelIndex}")

        with open(f"./{outputfolder}/model_{analyzedModelIndex}/model.txt", "w") as model_file:
            model_file.write(str(m))
        
        path_symbols = []
        for name in [next(iter(d.values())) for d in modelAtomTemplates]:
            path_symbols += list(filter(lambda s: s.name == name, m.symbols(atoms=True)))
        cprint(f"Generated {len(path_symbols)} path symbols")

        cprint(f"Start processing Model {foundModelIndex} ... ")
        cprint("creating template atoms ...")
        for modelAtomTemplate in modelAtomTemplates:
            showModelAtoms(path_symbols, modelAtomTemplate['name'], modelAtomTemplate['filter'], modelAtomTemplate['columns'],
                        foundModelIndex)
    
        if m.optimality_proven:
            print(
                f"=== Optimal model [{foundModelIndex}] found after {str(round((time.process_time() - start_solving) / 60, 2))} minutes of solving (Optimality proven: {m.optimality_proven}) === ")
            
        analyzedModelIndex += 1 

    foundModelIndex += 1

def run_asp(): 
    global output_hundredth_model, outputfolder, start_time_solving, ctl, \
    foundModelIndex, analyzedModelIndex, showModel, print_details, \
    exit_after_optimal_found, draw, check_clingcon_multibatching, timeout, compute_costs_var

    parser = argparse.ArgumentParser(description="Runs logic-programs in sub-folders 'facts' and 'rules'.")
    parser.add_argument("-rc", "--run_clingcon", required=False, default=100, type=int, help="runs clingcon.")
    parser.add_argument("-t", "--timeout", required=False, default=100, type=int, help="time out.")
    parser.add_argument("-d", "--details", required=False, default=False, help="show details on configuration.")
    parser.add_argument("-c", "--check_clingcon_multibatching", required=False, default=False, help="checks flow, load and TR/product correctness in multi batching.")
    parser.add_argument("-co", "--compute_costs", required=False, default=False, help="compute costs on multi batching.")
    parser.add_argument("-f", "--draw", required=False, default=False, help="draw figures while running.")
    parser.add_argument("-eo", "--exit_after_optimal_found", required=False, default=True, help="Exits after first optimal model is found")
    parser.add_argument("-p", "--print_details", required=False, default=True, help="Prints details to each model found")
    parser.add_argument("-n", "--models", required=False, default=1,
                        help="Maximum number of models returned by the tool.")
    parser.add_argument("-o", "--outputfolder", required=False, default=f'./output-{datetime.now()}', help="The folder to put results into.")
    parser.add_argument("-m", "--showmodel", required=False, action="store_true", default=False,
                        help="Print each model entirely.")
    parser.add_argument("-df", "--showdataframes", required=False, action="store_true", default=False, help="Print each dataframe.")
    parser.add_argument("-e", "--example", required=False, action="store_true", default=False,
                        help="Considers only the logic programs in the example.")
    parser.add_argument("-s", "--output_hundredth_model", required=False, action="store_true", default=False, help="will output each 100th model")
    parser.add_argument("-hr", "--heuristics", required=False, default=False, help="Will run different heuristics combinations from heuristics folder")
    parser.add_argument("-j", "--json_file", required=False,default=0, help="reads json file with all required paramters")
    parser.add_argument("-ts", "--timestamp_on_results_folder", required=False,default=True, help="timestamp_on_results_folder")
    parser.add_argument("-pr", "--program_folder", required=False,default=True, help="folder in which the programs are")
    args = parser.parse_args()

    json_file = args.json_file

    if json_file:

        with open(json_file, 'r') as file_open:
            json_data = json.load(file_open)[0]
        run_clingcon = json_data["run_clingcon"]
        timeout = json_data["timeout"]
        max_nr_models = json_data["max_nr_models"]
        showModel = json_data["show_model"]
        showDataframes = json_data["show_dataframes"]
        example = json_data["example"]
        output_hundredth_model = json_data["output_hundredth_model"]
        heuristics_runs = json_data["runs"]
        all_results_folder = json_data["all_results_folder"]
        print_details = json_data["print_details"]
        exit_after_optimal_found = json_data["exit_after_optimal_found"]
        draw = json_data["draw"]
        check_clingcon_multibatching = json_data["check_clingcon_multibatching"]
        compute_costs_var = json_data["compute_costs"]
        program_folder = json_data["program_folder"]

        if json_data["timestamp_on_results_folder"]:
            all_results_folder = f'{all_results_folder}_{datetime.now()}'
    else:
        run_clingcon = args.run_clingcon
        timeout = args.timeout
        max_nr_models = args.models
        showModel = args.showmodel
        showDataframes = args.showdataframes
        example = args.example
        output_hundredth_model = args.output_hundredth_model
        heuristics_runs = args.heuristics
        json_file = args.json_file
        print_details = args.print_details
        exit_after_optimal_found = args.exit_after_optimal_found
        draw = args.draw
        all_results_folder = F'{args.outputfolder}'
        check_clingcon_multibatching = args.check_clingcon_multibatching
        compute_costs_var = args.compute_costs
        program_folder = args.program_folder
        if args.timestamp_on_results_folder:
            all_results_folder = f'{all_results_folder}_{datetime.now()}'

    categories = [f"lps/{program_folder}"]

    hf.emptyFolder(all_results_folder)
    gc.collect()  
    outputfolder = f'{all_results_folder}'
    if run_clingcon == True:
        solve_clingcon(hf.get_programs_by_categories(categories), max_nr_models, False)
    else:
        hf.solve_clingo(hf.get_programs_by_categories(categories), max_nr_models, False)
    gc.collect()  

if __name__ == "__main__":
    myobs = hf.MyObserver()
    run_asp()




## externals
#     ctl.ground([('base', [])])
#   current_best = 1000000
#    optimum_model = None
#    ctl.configuration.solve.solve_limit = timeout
#
#    while True:
#        # Assign the current best cost to the external bound
#        ctl.assign_external(clingo.Function("bound", [clingo.Number(current_best)]), True)
#        
#        found_better = False
#        with ctl.solve(yield_=True) as handle:
#            for model in handle:
#                # Get theory variables
#                for name, val in thy.assignment(model.thread_id):
#                    if str(name) == "total_costs":
#                        if val < current_best:
#                            optimum_model = model.symbols(shown=True)
#                            print(f"Improved Model Found! New Bound: {current_best}")
#                            found_better = True
#                            current_best = val
#                            # We found a better one, so break this solve handle 
#                        # and restart with a tighter bound
#                        break 
#        
#        if not found_better:
#            print("Optimization Complete.")
#            break
#
#    print(f"Final Optimum Cost: {current_best}")


def check_clingcon_load_freq(modelIndex):

    """
    Verifies that:
    1. Aggregated loads (from/to) match the net requirements at each location.
    2. Loads are consistent with estimated frequencies (e.g., no load without frequency).
    """
    demand_supply_df = all_dataframes["demandOffer"]
    demand_supply_df["amount"] = demand_supply_df["amount"].apply(to_int) * -1

    estimated_freq_df = all_dataframes["estimated_freq"]
    load_df = all_dataframes["load"]
    product_size_df = all_dataframes["productSize"]
    transport_capacity_df = all_dataframes["transportCapacity"]

    load_enriched['size'] = load_enriched['size'].fillna(0)

    # Calculate the total physical load for each row
    load_enriched['calculated_load'] = load_enriched['amount'] * load_enriched['size']

    # Group by route keys and sum the NEW 'calculated_load' column
    load_agg = load_enriched.groupby(['from', 'to', 'transportresource'])['calculated_load'].sum().reset_index()
    load_agg.rename(columns={'calculated_load': 'total_load'}, inplace=True)
    

    # 1. Aggregate loads entering locations (Group by product, To)
    load_in_agg = load_df.groupby(['product', 'to'])['amount'].sum().reset_index()
    load_in_agg.rename(columns={'to': 'location', 'amount': 'amount_in_calc'}, inplace=True)

    # 2. Aggregate loads leaving locations (Group by product, From)
    load_out_agg = load_df.groupby(['product', 'from'])['amount'].sum().reset_index()
    load_out_agg.rename(columns={'from': 'location', 'amount': 'amount_out_calc'}, inplace=True)

    # 3. Merge In and Out aggregates
    flow_calc = pd.merge(
        load_in_agg, 
        load_out_agg, 
        on=['product', 'location'], 
        how='outer'
    ).fillna(0)

    # 4. Calculate Net Flow (In - Out) derived purely from loads
    flow_calc['net_flow_calc'] = flow_calc['amount_in_calc'] - flow_calc['amount_out_calc']

    # 5. Merge with Requirements (Demand/Supply)
    # We use suffixes to distinguish the req amount from our calculated flow
    check_df = pd.merge(
        flow_calc,
        demand_supply_df,
        on=['product', 'location'],
        how='outer',
        suffixes=('_calc', '_req')
    ).fillna(0)
    
    # 6. Verify: Remaining Req should be 0 (Req - NetFlow == 0)
    # Note: Depending on your sign convention in ASP:
    # If Req = Demand (Pos) - Offer (Pos), then Net Flow (In - Out) should equal Req.
    check_df['balance_check'] = check_df['amount'] - check_df['net_flow_calc']

    is_flow_consistent = np.all(np.isclose(check_df['balance_check'], 0))

    if not is_flow_consistent:
        print(f"{YELLOW}========== Model {modelIndex}: Load Aggregation Mismatch =========={RESET}")
        inconsistent = check_df[~np.isclose(check_df['balance_check'], 0)]
        print("Inconsistent Location/product (Load vs Req):")
        # Displaying relevant columns for debugging
        print(inconsistent[['product', 'location', 'amount', 'net_flow_calc', 'balance_check']])
        #sys.exit('exit')
    
    # 1. Prepare Load Data
    # Sum total load amount per edge. 
    # Result has columns: ['from', 'to', 'transportresource', 'amount']
    total_load_per_edge = load_df.groupby(['from', 'to', 'transportresource'])['amount'].sum().reset_index()
    
    # Explicitly rename 'amount' to 'amount_load' to avoid ambiguity
    total_load_per_edge.rename(columns={'amount': 'amount_load'}, inplace=True)
    
    freq_for_merge = estimated_freq_df.copy()
    freq_for_merge.rename(columns={'freq': 'amount_freq'}, inplace=True)

    # 3. Merge
    # Now we are merging explicit 'amount_load' and 'amount_freq' columns.
    edge_consistency = pd.merge(
        total_load_per_edge,
        freq_for_merge,
        on=['from', 'to', 'transportresource'],
        how='outer'
    ).fillna(0)

    # Debug print to confirm structure
    # print("Columns after merge:", edge_consistency.columns)

    # Check 1: Has Load but No Freq
    orphan_loads = edge_consistency[
        (edge_consistency['amount_load'] > 0) & 
        (edge_consistency['amount_freq'] == 0)
    ]
    
    if not orphan_loads.empty:
        print(f"{YELLOW}========== Model {modelIndex}: Load without Frequency =========={RESET}")
        print(orphan_loads)
        sys.exit('exit')

    if product_size_df is not None and transport_capacity_df is not None:
        
        
        product_size_df["size"] = product_size_df["size"].apply(to_int)
        load_df["amount"] = load_df["amount"].apply(to_int)
        estimated_freq_df["freq"] =  estimated_freq_df["freq"].apply(to_int)
        transport_capacity_df["capacity"] = transport_capacity_df["capacity"].apply(to_int)

        # 1. Calculate Required Capacity (Load * productSize)
        # Merge load with productSize. Assumes productSize columns are ['product', 'size'] (or similar value column)
        load_with_size = pd.merge(load_df, product_size_df, on='product', how='left')
        
        # Calculate total volume for this specific load entry
        load_with_size['required_vol'] = load_with_size['amount'] * load_with_size["size"]
        
        # Aggregate required volume by Edge (From, To, TR)
        req_vol_per_edge = load_with_size.groupby(['from', 'to', 'transportresource'])['required_vol'].sum().reset_index()

        # 2. Calculate Available Capacity (Freq * TR_Capacity)
        freq_with_cap = pd.merge(estimated_freq_df, transport_capacity_df, on='transportresource', how='left')
        freq_with_cap['available_vol'] = freq_with_cap['freq'] * freq_with_cap["capacity"]

        # 3. Compare Required vs Available
        cap_check = pd.merge(
            req_vol_per_edge,
            freq_with_cap[['from', 'to', 'transportresource', 'available_vol', 'freq']], 
            on=['from', 'to', 'transportresource'],
            how='outer'
        ).fillna(0)

        # Check violation: Required > Available
        violations = cap_check[cap_check['required_vol'] > cap_check['available_vol']]

        if not violations.empty:
            print(f"{YELLOW}========== Model {modelIndex}: Capacity Violation =========={RESET}")
            print(f"Violation: Total product Size > (Frequency * TR Capacity)")
            # Renaming for clear print output
            violations = violations.rename(columns={'amount': 'freq'})
            print(violations[['from', 'to', 'transportresource', 'freq', 'available_vol', 'required_vol']])
            sys.exit('exit')
    print(f"Model {modelIndex}: Frequencies, and Capacities seem consistent.")
