
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