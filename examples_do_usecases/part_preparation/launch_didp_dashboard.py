
import streamlit as st
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

data_file = Path(r"/home/ehsow/Documents/do/data/part_preparation/solutions/xlarge_didp_multi_cabs_solution.pkl")
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])
st.session_state.solver = data.get('solver', 'solveur inconnu')

from part_preparation.unified_dashboard import main
main()
