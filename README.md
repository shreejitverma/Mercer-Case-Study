# Mercer-Case-Study

Streamlit is an open-source Python library that helps us create and share custom web apps in pure python.

The astreamlit_app.py file runs continuously in background.
(Functions written here run whenever we hit refresh)

Install the following dependencies:

pip install pandas
pip install numpy
pip install openpyxl
pip install streamlit


Open Terminal in the downloaded folder:

1. cd dashboard
2. streamlit run streamlit_app.py
Two files need to be uploaded: 
(Please see the existing file and their structure in the Folder; make sure the uploaded file is consistent with their respective existing files in the folder)
CMA.xlsx - (Please see the existing file in the Folder)
CC Scen.xlsx - (Without Cumulative Returns)
Portfolio Allocation.xlsx - (All the allocation changes to be made here only)
3. Click on 'Risk Return Caculation' button to see the respective result.
4. Click on 'Climate Change Stress Test Calculaion' button to see the respective result.
5. Click on 'Generate Final Report' button to generate the final report with All the Sheets(i.e. Allocation and Result, Setup, CMA, and CCScen) in it
6. Click 'Download' to download the Final Report.

Files:
Code is inside dashboard folder
1. streamlit_app.py - Code to run the streamlit application
2. utils.py - All the functions Utilities are present here
3. file_uploader.py - code to upload files on the application
   

functions and their usage:

utils.py

'diagonalise_matrix' - Gets diagonal of a matrix
 'get_CCScen' - Gets processed CC Scen Table from the uploaded CC Scen file
 'get_CMA'- Gets processed CMA Table from the uploaded CMA file
 'get_Setup' - Gets processed Setup Table (First Table in Setup Tab)
 'get_active_correlation_table' - Calculates Active Correlation Table from provided CMA
 'get_active_covariance_table' - Calculates Active Covariance Table from provided CMA
 'get_climate_change_stress_tests' - Calculates Climate Change Stress Tests Table that will be written in last of Setup Tab
 'get_cma_correlation_table' - Calculates CMA Correlation Table
 'get_passive_correlation_matrix' - Calculates Passive Correlation Table
 'get_passive_correlation_table' - Calculates Passive Correlation Table
 'get_passive_covariance_np_matrix' - Calculates Passive Covariance Numpy Matrix
 'get_passive_covariance_table' - Calculates Passive Covariance Table from provided CMA
 'get_portfolio' - Get Processed Portfolio from uploaded Portfolio Allocation Table
 'get_result_climate_change_stress_tests' - Calculates required result for Climate Change Stress Tests
 'get_risk_return_calculation' - Calculates Risk Return (the first required result)
 'get_total_covariance_table' - Calculates Total Covariance
 'write_active_correlation_table' - Writes Active Correlation Table in excel
 'write_active_covariance_table' - Writes Active Covariance Table in Excel
 'write_passive_correlation_matrix' - Writes Passive Correlation Table in excel 
 'write_passive_covariance_table' - Writes Passive Covariance Table in excel
 'write_total_covariance_table'
 'write_final_report' - Writes Final Report in excel with All the Sheets(i.e. Allocation and Result, Setup, CMA, and CCScen) in it
 

