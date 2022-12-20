import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import numpy as np
import math
import plost # pip install plost

from file_uploader import FileUploader
from utils import Utils

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Dashboard')

# st.sidebar.subheader('Heat map parameter')
# time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max'))

# st.sidebar.subheader('Donut chart parameter')
# donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

# st.sidebar.subheader('Line chart parameters')
# plot_data = st.sidebar.multiselect(
#     'Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created by Shreejit Verma.
''')
# --------------------------- Uploading Files Starts -------------------------------------

cma = FileUploader('CMA')
cmaDf = cma.upload_file()
ccscen = FileUploader('CC Scen')
ccscenDf = ccscen.upload_file()
# if not cmaDf.empty:
#     st.dataframe(data=cmaDf, width=200, height=200)

# if not ccscenDf.empty:
#     st.dataframe(data=ccscenDf, width=200, height=200)
# --------------------------Uploading Files Ends --------------------------------------

UT = Utils()
st.checkbox("Use container width", value=False, key="result_risk_allocation_width")
result_risk_allocation_calculation = UT.get_result_risk_allocation_calculation()
st.dataframe(data=result_risk_allocation_calculation, use_container_width=st.session_state.result_risk_allocation_width)
st.checkbox("Use container width", value=False, key="result_climate_change_stress_tests_width")
result_climate_change_stress_tests = UT.get_result_climate_change_stress_tests()
st.dataframe(data=result_climate_change_stress_tests, use_container_width=st.session_state.result_climate_change_stress_tests_width)