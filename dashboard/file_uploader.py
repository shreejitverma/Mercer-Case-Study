import streamlit as st
import pandas as pd


class FileUploader:
    def __init__(self, name='CMA'):
        self.name = name

    def upload_file(self):
        message = "Choose" + str(self.name) + "file"
        uploaded_file = st.file_uploader(message)
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                # read csv
                df1 = pd.read_csv(uploaded_file)
                st.write("Filename: ", uploaded_file.name)

            elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
                # read xls or xlsx
                df1 = pd.read_excel(uploaded_file)
                st.write("Filename: ", uploaded_file.name)
            return df1
        else:
            st.warning("You need to upload a csv or excel file.")
