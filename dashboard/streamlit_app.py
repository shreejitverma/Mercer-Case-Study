import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import numpy as np
import math
import plost # pip install plost
import os

from file_uploader import FileUploader
from utils import Utils

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Dashboard')



st.sidebar.markdown('''
---
Created by Shreejit Verma.
''')
# --------------------------- Uploading Files Starts -------------------------------------
# cmaDf = pd.DataFrame()


cma = FileUploader('CMA')
cmaDf = cma.upload_file()

if not cmaDf.empty:
    st.checkbox("Use container width", value=False, key="cma_width")
    st.dataframe(data=cmaDf,use_container_width=st.session_state.cma_width)

ccscen = FileUploader('CC Scen')
ccscenDf = ccscen.upload_file()

if not ccscenDf.empty:
    st.checkbox("Use container width", value=False, key="ccscen_width")
    st.dataframe(data=ccscenDf, use_container_width=st.session_state.ccscen_width)

portfolio_allocation =  FileUploader('Portfolio Allocation')
portfolio_allocationDf = portfolio_allocation.upload_file()

if not portfolio_allocationDf.empty:
    st.checkbox("Use container width", value=False, key="portfolio_allocation_width")
    st.dataframe(data=ccscenDf, use_container_width=st.session_state.portfolio_allocation_width)
# --------------------------Uploading Files Ends --------------------------------------

UT = Utils()
st.markdown(""" Risk Return Calculaion""")
st.checkbox("Use container width", value=False, key="result_risk_allocation_width")
result_risk_allocation_calculation = UT.get_result_risk_allocation_calculation()
st.dataframe(data=result_risk_allocation_calculation, use_container_width=st.session_state.result_risk_allocation_width)


st.markdown(""" Climate Change Stress Test """)
st.checkbox("Use container width", value=False, key="result_climate_change_stress_tests_width")
result_climate_change_stress_tests = UT.get_result_climate_change_stress_tests()
st.dataframe(data=result_climate_change_stress_tests, use_container_width=st.session_state.result_climate_change_stress_tests_width)