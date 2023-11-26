import streamlit as st
import os
import cv2
import numpy as np

from natsort import natsorted
# from utils.utils import file_selector
from streamlit_image_select import image_select
from PIL import Image

st.set_page_config(
    page_title="Inspect sequence",
    page_icon="🔍",
    layout="wide"
) 


st.markdown(
    """
    #### Show an specific sequence of a case
    """
)

# TODO: Read score, MSE and MAG from json and show it

casename = "./data/case_0"
filename = "./data/case_0/overview_1"

col1, col2, col3 = st.columns(3)

with col1:
    # Select a case path
    folder_path = "./data/"

    filenames = [dir for dir in os.listdir(folder_path) if (os.path.isdir(os.path.join(folder_path, dir)))]
    selected_filename = st.selectbox('Select a case', filenames, key="IS_case_sel")
    casename = os.path.join(folder_path, selected_filename)
    st.write('You selected `%s`' % casename)

    # Select an overview path
    filenames = [dir for dir in os.listdir(casename) if (os.path.isdir(os.path.join(casename, dir)))]
    selected_filename = st.selectbox('Select a sequence', filenames, key="IS_over_sel")
    filename = os.path.join(casename, selected_filename).replace("\\","/")
    st.write('You selected `%s`' % filename)



img_list = [os.path.join(filename, img) for img in natsorted(os.listdir(filename))]

if (len(img_list) != 0):

    captions_list = ["Frame "+str(i+1) for i in range(len(img_list))]
    #print(img_list)

    IS_img = image_select(
        label="",
        images=img_list,
        captions=captions_list,
        use_container_width = False
    )

    st.image(IS_img, width=256)
else:

    st.error('Empty folder', icon="🚨")

    img_list = [os.path.join(filename, img) for img in natsorted(os.listdir("./data/case_0/overview_0"))]
    IS_img = image_select(
        label="",
        images=img_list,
        captions=None,
        use_container_width = False
    )

    st.image(IS_img, width=256)