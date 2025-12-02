
import os
import re
from clingo import Number, Function, String
import clingo
import shutil
from clingo.control import Control
from clingo.backend import Observer
import pandas as pd

all_data_frames = {}

class MyObserver(Observer):
    def __init__(self):
        self.number_rules = 0
        self.number_choice_rules = 0
        self.list_of_rules = []

    def start(self):
        self.number_rules = 0

    def rule(self, choice, head, body):
        self.list_of_rules.append([choice, head, body])
        if choice:
            self.number_choice_rules += 1
        self.number_rules += 1

def emptyFolder(folder):
    #print(f"Creating emtpy folder: '{folder}' ")
    try:
        shutil.rmtree(folder)
    except:
        pass
    finally:
        os.mkdir(folder)

def printModel(m: clingo.Model):
    print(m)

def get_artifacts(type: str):
    hiddenFiles = re.compile('/\.')
    return filter(lambda x: (hiddenFiles.search(x) is None),
                  list(map(lambda x: './' + type + '/' + x, os.listdir('./' + type))))

def get_artifacts_no_hidden_files(type: str):
    return list(map(lambda x: './' + type + '/' + x, os.listdir('./' + type)))


def get_programs_by_categories_no_hidden(categories: list[str]):
    programs = []
    for category in categories:
        if os.path.isdir(category):
            programs += get_artifacts_no_hidden_files(category)
        elif os.path.isfile(category):
            programs += [category]
        else:
            print(f"Folder or file \'{category}\' does not exist.")
    return programs

def read_files_to_string(file_list, separator="\n"):
    accumulated_content = []

    for filename in file_list:
        # Check if the file name (ignoring parent folders) starts with a dot
        if os.path.basename(filename).startswith('.'):
            print(f"Skipping hidden file: {filename}")
            continue

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                accumulated_content.append(f.read())
        except FileNotFoundError:
            print(f"Warning: File '{filename}' not found. Skipping.")
        except Exception as e:
            print(f"Error reading '{filename}': {e}")

    return separator.join(accumulated_content)

def cprint(Text):
    global print_details
    if print_details:
        print(Text)

def get_programs_by_categories(categories: list[str]):
    programs = []
    for category in categories:
        if os.path.isdir(category):
            programs += get_artifacts(category)
        elif os.path.isfile(category):
            programs += [category]
        else:
            print(f"Folder or file \'{category}\' does not exist.")
    return programs



def solve_clingo(programs, max_models=1, ngo=False):
    global start_solving, produced_at_atoms, ctl
    print('Configuring Clingo ...')
    print(f'--models={max_models}')

    ctl = clingo.Control(
        ['--warn=no-atom-undefined', f'--models={max_models}', "--opt-mode=optN", "--parallel-mode=12", "--heuristic=Domain"])
    print(f'Configuration Keys: {ctl.configuration.solve.keys}')
    ctl.register_observer(myobs)
    myobs.start()#
    
    print("programs", programs)
    if ngo==True:
        print(f"=== NGO: Encoding Optimization ===")
        all_programs = " ".join(programs)
        command1 = f"cat {all_programs} > encoding.lp" 
        command2 = " cat encoding.lp | ngo > new_encoding.lp"
        
        programs = ['new_encoding.lp']
        # 5. Execute the combined pipeline
        try:
        
            result1 = subprocess.run(
                command1,
                shell=True,
                check=True,  # Raise CalledProcessError on non-zero exit code
                capture_output=True,
                text=True
            )
            
            print("\n✅ Merge completed successfully!")

            result2 = subprocess.run(
                command2,
                shell=True,
                check=True,  # Raise CalledProcessError on non-zero exit code
                capture_output=True,
                text=True
            )
            print(f"The optimized logic program is in: **new_encoding.lp**")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error during command execution (Exit Code {e.returncode}):")
            print(f"Stderr from ngo: {e.stderr}")
        except FileNotFoundError:
            print("\n❌ Error: The 'cat' or 'ngo' command was not found.")

    print(f"=== Loading programs: {programs} ===")
    for p in programs:
        ctl.load(p)
    start_time_grounding = time.process_time()

    print(f"=== Grounding programs: {programs} ... ===")
    ctl.ground([("base", [])])
    end_time_grounding = time.process_time()
    print(f"=== Grounding done ({round((time.process_time() - start_time_grounding) / 60, 2)} minutes, {myobs.number_rules} rules, {myobs.number_choice_rules} choice rules) ===")
    
    start_solving = time.process_time()
    res = ctl.solve(on_model=on_model)
    ctl.cleanup()
    return res
