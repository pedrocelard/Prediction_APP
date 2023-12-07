import streamlit as st
import os

def file_selector(folder_path='.',select_object="file"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a '+select_object, filenames)
    return os.path.join(folder_path, selected_filename)

def get_stage_position(stages):
    positions = []
    for stage in stages:
        if stage in ['tPB2', 'tPNa', 'tPNf']: pos = 0
        if stage == "t2": pos = 1
        if stage == "t3": pos = 2
        if stage == "t4": pos = 3
        if stage == "t5": pos = 4
        if stage == "t6": pos = 5
        if stage == "t7": pos = 6
        if stage == "t8": pos = 7
        if stage == "t9+": pos = 8
        if stage == "tM": pos = 9
        if stage == "tSB": pos = 10
        if stage == "tB": pos = 11
        if stage == "tEB": pos = 12
        if stage == "tHB": pos = 13
        if stage == "empty": pos = 14

        positions.append(str(pos))
    position_str = ",".join(positions)

    return position_str