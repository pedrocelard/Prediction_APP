import streamlit as st
import os
import cv2
import numpy as np
import json

from natsort import natsorted
from streamlit_image_select import image_select
from PIL import Image

st.set_page_config(
    page_title="Sequences ranking",
    page_icon="🏅",
    layout="wide"
) 


st.markdown(
    """
    #### Show an specific sequence of a case
    """
)

# TODO: Manual ranking of a sequence (up and down buttons)
# TODO: Delete overview sequence (delete button)
# TODO: Inspect sequence -> Go to its page with curren selection

# Select a case path
folder_path = "./data/"

col1, col2, col3 = st.columns(3)

with col1:
    # We get the  case dirs checking if the object returned by listdir is a directory
    filenames = [dir for dir in os.listdir(folder_path) if (os.path.isdir(os.path.join(folder_path, dir)))]
    selected_filename = st.selectbox('Select a case', filenames, key="SR_case_sel")
    casename = os.path.join(folder_path, selected_filename)
    st.write('You selected `%s`' % casename)

    # Get the overview folders checking if they are directories
    overview_folder = [os.path.join(casename, dir) for dir in natsorted(os.listdir(casename))
                    if (os.path.isdir(os.path.join(casename, dir)))]


if(selected_filename != "case_0"):

    # Specify the path to your JSON file
    json_file_path = os.path.join(casename,"case_info.json")
    
    # Read the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    if (data["overview_maual_ranking"]):
        st.success(f'This ranking was modified by the user')

    # Loop over every overview for each case
    for overview in data["overview_ranking"]:
        
        overview = os.path.join(casename, overview)

        # Get the overview identifier name
        overview_id_value = os.path.basename(overview)
        
        # Filter overview_info based on overview_id
        filtered_info = [info for info in data["overview_info"] if info["overview_id"] == overview_id_value]

        # Output overview images
        img_list = [os.path.join(overview, img) for img in natsorted(os.listdir(overview))]
        captions_list = ["Frame "+str(i+1) for i in range(len(img_list))]

        # Get the overview id and score
        overview_id = filtered_info[0]["overview_id"].split("_")[-1]
        score = filtered_info[0]["score"]

        # Show the images
        img = image_select(
            label=f"Overview {overview_id} Score: {score}",
            images=img_list,
            captions=None,
            use_container_width = False
        )
else:

    img_list = [os.path.join(overview_folder[0], img) for img in natsorted(os.listdir(overview_folder[0]))]
    captions_list = ["Frame "+str(i+1) for i in range(len(img_list))]

    img = image_select(
        label="Score: XX.XX",
        images=img_list,
        captions=None,
        use_container_width = False
    )