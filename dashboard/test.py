import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import math
from utils import Utils

UT = Utils()
UT.write_final_report()
portfolio_allocation_Df = UT.get_portfolio()
cma_correlation_Df = UT.get_cma_correlation_table()
SETUP = UT.get_Setup()
passive_correlation_table = UT.get_passive_correlation_table()
passive_covariencenp = UT.get_passive_covariance_np_matrix()
active_correlation_table = UT.get_active_correlation_table()
passive_covarianceDf = UT.get_passive_covariance_table()
active_covarianceDf = UT.get_active_covariance_table()
total_covarianceDf = UT.get_total_covariance_table()
result_risk_allocation_calculation = UT.get_result_risk_allocation_calculation()
ClimateChangeStressTestsDf = UT.get_result_climate_change_stress_tests()

