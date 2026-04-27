
import streamlit as st
import pickle
from pathlib import Path
import sys

# Add project root to path BEFORE any imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load data
data_file = Path(__file__).parent / "solution_data.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']

# Import and run dashboard
from part_preparation.dashboard_streamlit import main
main()
