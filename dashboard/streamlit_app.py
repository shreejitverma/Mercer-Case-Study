import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import numpy as np
import math
import plost  # pip install plost
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
st.markdown("""  """)
st.markdown("""  """)

st.markdown(""" Upload Required Files Below """)
cma = FileUploader('CMA')
cmaDf = cma.upload_file()

if not cmaDf.empty:
    st.checkbox("Use container width", value=False, key="cma_width")
    st.dataframe(data=cmaDf, use_container_width=st.session_state.cma_width)

ccscen = FileUploader('CC Scen')
ccscenDf = ccscen.upload_file()

if not ccscenDf.empty:
    st.checkbox("Use container width", value=False, key="ccscen_width")
    st.dataframe(
        data=ccscenDf, use_container_width=st.session_state.ccscen_width)

portfolio_allocation = FileUploader('Portfolio Allocation')
portfolio_allocationDf = portfolio_allocation.upload_file()

if not portfolio_allocationDf.empty:
    st.checkbox("Use container width", value=False,
                key="portfolio_allocation_width")
    st.dataframe(
        data=ccscenDf, use_container_width=st.session_state.portfolio_allocation_width)
# --------------------------Uploading Files Ends --------------------------------------

UT = Utils()


if st.button(
        'Risk Return Calculaion', key='risk_return_calculaion'):

    st.markdown(""" Risk Return Calculaion""")
    st.checkbox("Use container width", value=False,
                key="result_risk_allocation_width")
    risk_return_calculationDf = UT.get_risk_return_calculation().style.format(
        "{:.3%}")
    st.dataframe(data=risk_return_calculationDf,
                 use_container_width=st.session_state.result_risk_allocation_width)


if st.button(
        'Climate Change Stress Test Calculaion', key="change_stress_test_calculaion"):
    st.markdown(""" Climate Change Stress Test """)
    st.checkbox("Use container width", value=False,
                key="result_climate_change_stress_tests_width")
    # pd.options.display.float_format = '${:,.2f}'.format
    result_climate_change_stress_tests = UT.get_result_climate_change_stress_tests(
    )
    # result_climate_change_stress_tests.loc[['Transition (2°C) 1yr', 'Transition (2°C) 3yr'], :] =
    df = result_climate_change_stress_tests.iloc[[
        1, 2, 4, 5, 7, 8, 10, 11]].style.format("{:.3%}")
    
    st.dataframe(data=df,
                 use_container_width=st.session_state.result_climate_change_stress_tests_width)

if st.button('Generate Final Report'):

    UT.write_final_report()
    st.write('Final Report has been genrated at path ' +
             UT.path_Final_Portfolio)
    with open(UT.path_Final_Portfolio, 'rb') as my_file:
        st.download_button(label='Download', data=my_file, file_name='Portfolio Final Result.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
