
import streamlit as st
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load problem and solution
data_file = Path(__file__).parent / "solution_data" / "medium_solution.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])

# Import and run unified dashboard
from part_preparation.unified_dashboard import main
main()
