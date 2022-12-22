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

1. 'diagonalise_matrix' - Gets diagonal of a matrix
2. 'get_CCScen' - Gets processed CC Scen Table from the uploaded CC Scen file
3. 'get_CMA'- Gets processed CMA Table from the uploaded CMA file
4. 'get_Setup' - Gets processed Setup Table (First Table in Setup Tab)
5. 'get_active_correlation_table' - Calculates Active Correlation Table from provided CMA
6. 'get_active_covariance_table' - Calculates Active Covariance Table from provided CMA
7. 'get_climate_change_stress_tests' - Calculates Climate Change Stress Tests Table that will be written in last of Setup Tab
8. 'get_cma_correlation_table' - Calculates CMA Correlation Table
9. 'get_passive_correlation_matrix' - Calculates Passive Correlation Table
10. 'get_passive_correlation_table' - Calculates Passive Correlation Table
11. 'get_passive_covariance_np_matrix' - Calculates Passive Covariance Numpy Matrix
12. 'get_passive_covariance_table' - Calculates Passive Covariance Table from provided CMA
13. 'get_portfolio' - Get Processed Portfolio from uploaded Portfolio Allocation Table
14. 'get_result_climate_change_stress_tests' - Calculates required result for Climate Change Stress Tests
15. 'get_risk_return_calculation' - Calculates Risk Return (the first required result)
16. 'get_total_covariance_table' - Calculates Total Covariance
17. 'write_active_correlation_table' - Writes Active Correlation Table in excel
18. 'write_active_covariance_table' - Writes Active Covariance Table in Excel
19. 'write_passive_correlation_matrix' - Writes Passive Correlation Table in excel 
20. 'write_passive_covariance_table' - Writes Passive Covariance Table in excel
21. 'write_total_covariance_table'
22. 'write_final_report' - Writes Final Report in excel with All the Sheets(i.e. Allocation and Result, Setup, CMA, and CCScen) in it
 

