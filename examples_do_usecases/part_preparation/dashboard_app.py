"""
Streamlit Dashboard Launcher

Run with: uv run streamlit run examples_do_usecases/part_preparation/dashboard_app.py
"""
import streamlit as st
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from part_preparation.dashboard_streamlit import main as dashboard_main

# Load solution data if available
data_file = Path(__file__).parent / "solution_data.pkl"

if data_file.exists():
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    st.session_state.problem = data['problem']
    st.session_state.solution = data['solution']

# Run dashboard
dashboard_main()
