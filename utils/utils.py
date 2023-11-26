import streamlit as st
import os

def file_selector(folder_path='.',select_object="file"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a '+select_object, filenames)
    return os.path.join(folder_path, selected_filename)