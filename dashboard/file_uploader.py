import streamlit as st
import pandas as pd
import os


class FileUploader:
    def __init__(self, name='CMA'):
        self.name = name

    def upload_file(self):
        message = "Choose " + str(self.name) + " file"
        uploaded_file = st.file_uploader(message)
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                # read csv
                df1 = pd.read_csv(uploaded_file)
                st.write("Filename: ", uploaded_file.name)
                self.save_uploadedfile(uploaded_file)
            elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
                # read xls or xlsx
                df1 = pd.read_excel(uploaded_file)
                st.write("Filename: ", uploaded_file.name)
                self.save_uploadedfile(uploaded_file)

            return df1
        else:
            st.warning("You need to upload a csv or excel file.")
            return pd.DataFrame()

    def save_uploadedfile(self,uploadedfile):
        with open(os.path.join("../", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
