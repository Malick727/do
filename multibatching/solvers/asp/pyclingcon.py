
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
all_data_frames = {}
costs = []

modelAggregateTemplates = [ {'name':'flowOut', 'filter':lambda s: s.name=='flowOut', 'columns':['part' ,'location', 'amount']},
                           {'name':'flowIn', 'filter':lambda s: s.name=='flowIn', 'columns':['part' ,'location', 'amount']},
                           {'name':'req', 'filter':lambda s: s.name=='req', 'columns':['part' ,'location', 'amount']},
                          {'name':'routeCostTotal', 'filter':lambda s: s.name=='routeCostTotal', 'columns':['from' ,'to', 'transport resource', 'costs']}
                           ]

modelAtomTemplates = [
      {'name':'demandOffer','filter':lambda s: s.name=='demandOffer', 'columns':['part', 'location', 'amount']},
      {'name':'requiredNet','filter':lambda s: s.name=='requiredNet', 'columns':['part', 'location', 'amount']}
]

def cprint(Text):
    global print_details
    if print_details:
        print(Text)


def solve_clingcon(programs, max_models=1, ngo=False):
    global start_solving, produced_at_atoms, ctl
    print('Configuring Clingcon ...')
    print(f'--models={max_models}')

    prg = hf.read_files_to_string(programs)

    thy = ClingconTheory()
    ctl = Control(['0'])
    print(f'Configuration Keys: {ctl.configuration.solve.keys}')
    print("programs", programs)
    ctl.register_observer(myobs)
    myobs.start()
    thy.register(ctl)
    with ProgramBuilder(ctl) as bld:
        parse_string(prg, lambda ast: thy.rewrite_ast(ast, bld.add))

    start_time_grounding = time.process_time()
    print(f"=== Grounding programs: {programs} ... ===")
    ctl.ground([('base', [])])
    
    end_time_grounding = time.process_time()
    print(f"=== Grounding done ({round((end_time_grounding - start_time_grounding) / 60, 2)} minutes, {myobs.number_rules} rules, {myobs.number_choice_rules} choice rules) ===")
    
    thy.prepare(ctl)
    start_solving = time.process_time()
    with ctl.solve(yield_=True, on_model=on_model) as hnd:
        for idx, mdl in enumerate(hnd):
            print(f"Model {idx}")
            assignment_data = list(thy.assignment(mdl.thread_id))
            processAggregates(assignment_data, idx)            
            check_clingcon_flow(idx)
            compute_costs(idx)
    ctl.cleanup()

def compute_costs(idx):
    routeCostTotal_df = all_data_frames["routeCostTotal"]
    print(f"Total costs for model {idx}: {routeCostTotal_df["costs"].sum()}")
    
def processAggregates(assignment_data, idx):
    for modelAggregateTemplate in modelAggregateTemplates:
        showModelAggregates(
            assignment_data=assignment_data,
            dataName=modelAggregateTemplate["name"], 
            columns=modelAggregateTemplate["columns"],
            index=idx
        )
    
def showModelAggregates(assignment_data, dataName, columns, index):
    filtered_data = [[*key.arguments, val] for key, val in assignment_data if key.name==dataName]
    df = pd.DataFrame(data=filtered_data, columns=columns)
    all_data_frames[dataName] = df
    output_dir = f"{outputfolder}/model_{index}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{dataName}.csv", index=False)


def check_clingcon_flow(modelIndex):
    demand_supply_df = all_data_frames["req"]
    flow_in = all_data_frames["flowIn"]
    flow_out = all_data_frames["flowOut"]

    in_and_out = pd.merge(
    flow_in, 
    flow_out, 
    on=['part', 'location'], 
    how='outer', 
    suffixes=('_in', '_out')
    )

    final_merged_df = pd.merge(
    in_and_out, 
    demand_supply_df, 
    on=['part', 'location'], 
    how='outer', 
    suffixes=('_merged', '_net_req')
    )   
    final_merged_df['net_flow'] = final_merged_df['amount_in'] - final_merged_df['amount_out']
    final_merged_df['remaining_req'] = final_merged_df['amount'] - final_merged_df['net_flow']

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
        print(f"Model {modelIndex} seems correct")



def processModel(m, modelIndex):
    global check_route_correctness_var, check_flow_correctness_var, compute_costs_var, check_clingcon_flow_var

    #hf.cprint("evaluate flow correctness ...")
    #evaluate_transportLinks(modelIndex, all_data_frames)
    if check_clingcon_flow_var == True:
        cprint("check clingcon flow correctness ...")
        check_clingcon_flow(modelIndex, all_data_frames)

    if check_flow_correctness_var == True:        
        hf.cprint("check flow correctness ...")
        check_flow_correctness(modelIndex, all_data_frames)
    if check_route_correctness_var == True:        
        hf.cprint("check route correctness ...")
        check_route_correctness(modelIndex, all_data_frames)
    if compute_costs_var == True:   
        hf.cprint("compute costs ...")
        compute_costs(modelIndex, all_data_frames)
        hf.cprint(f"Done processing Model {modelIndex} ... ")

def showModelAtoms(s, dataName, symbolFilter, columns, index):
    global iD_conversion

    filteredSymbols = list(filter(symbolFilter, s))
    arguments = list(map(lambda s: s.arguments, filteredSymbols))
    df = pd.DataFrame(data=arguments, columns=columns)
    all_data_frames[dataName] = df

    df.to_csv(f"{outputfolder}/model_{index}/{dataName}.csv")
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
            
            if exit_after_optimal_found:
                print(str(m))
                if compute_costs_var == True:
                    costs_df = pd.read_csv(f"{outputfolder}/costs.csv")
                    if draw:
                        draw_figures(costs_df)
                sys.exit()

        analyzedModelIndex += 1 

    foundModelIndex += 1

def run_asp(): 
    global output_hundredth_model, outputfolder, start_time_solving, ctl, \
    foundModelIndex, analyzedModelIndex, showModel, print_details, \
    exit_after_optimal_found, draw, check_route_correctness_var, compute_costs_var, check_flow_correctness_var, check_clingcon_flow_var
  
    parser = argparse.ArgumentParser(description="Runs logic-programs in sub-folders 'facts' and 'rules'.")
    parser.add_argument("-rc", "--run_clingcon", required=False, default=100, type=int, help="runs clingcon.")
    parser.add_argument("-t", "--timeout", required=False, default=100, type=int, help="time out.")
    parser.add_argument("-d", "--details", required=False, default=False, help="show details on configuration.")
    parser.add_argument("-cr", "--check_route_correctness", required=False, default=False, help="checks route correctness on multi batching.")
    parser.add_argument("-cf", "--check_flow_correctness", required=False, default=False, help="checks flow correctness on multi batching.")
    parser.add_argument("-ccf", "--check_clingcon_flow", required=False, default=False, help="check_clingcon_flow")

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
        check_clingcon_flow_var = json_data["check_clingcon_flow"]
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
        check_route_correctness_var = json_data["check_route_correctness"]
        check_flow_correctness_var = json_data["check_flow_correctness"]
        compute_costs_var = json_data["compute_costs"]
        program_folder = json_data["program_folder"]

        if json_data["timestamp_on_results_folder"]:
            all_results_folder = f'{all_results_folder}_{datetime.now()}'
    else:
        run_clingcon = args.run_clingcon
        check_clingcon_flow_var = args.check_clingcon_flow
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
        check_route_correctness_var = args.check_route_correctness
        check_flow_correctness_var = args.check_flow_correctness
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