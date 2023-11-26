import streamlit as st
import os
import cv2
import numpy as np
import json
import time

from natsort import natsorted
from streamlit_image_select import image_select
from PIL import Image

st.set_page_config(
    page_title="Timeline",
    page_icon="⏳",
    layout="wide"
) 


st.markdown(
    """
    #### Show the timeline of a case
    """
)

#TODO: Build a timeline example with the svg images

# Select a case path
folder_path = "./data/"

col1, col2, col3 = st.columns(3)

with col1:
    # We get the  case dirs checking if the object returned by listdir is a directory
    filenames = [dir for dir in os.listdir(folder_path) if (os.path.isdir(os.path.join(folder_path, dir)))]
    selected_filename = st.selectbox('Select a case', filenames, key="TL_case_sel")
    casename = os.path.join(folder_path, selected_filename)
    st.write('You selected `%s`' % casename)

    # Get the overview folders checking if they are directories
    overview_folder = [os.path.join(casename, dir) for dir in natsorted(os.listdir(casename))
                    if (os.path.isdir(os.path.join(casename, dir)))]


# Specify the path to your JSON file
json_file_path = os.path.join(casename,"case_info.json")

# Read the JSON file
with open(json_file_path, "r") as file:
    data = json.load(file)


if(selected_filename != "case_0"):

    # Get the timeline info and loop over timelines
    for time_line in data["timeline_info"]:
        
        # Specify the path to your JSON file
        json_file_path = os.path.join(casename,"case_info.json")
        
        # Read the JSON file
        with open(json_file_path, "r") as file:
            data = json.load(file)

        # Get the overview identifier name
        overview_id_value = os.path.basename(time_line["timeline_id"])
        
        # Complete img paths
        img_list = []
        for idx,img in enumerate(time_line["timeline_imgs"]):
            if(img == "-"):
                img_list.append(os.path.join("./media/overview_0_light",f"{idx+1}.jpg"))
            else:
                img_list.append(os.path.join(casename,img))


        captions_list = ["Frame "+str(i+1) for i in range(len(img_list))]

        # Show the images
        img = image_select(
            label=f"Timeline {overview_id_value}",
            images=img_list,
            captions=None,
            use_container_width = False
        )

    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose a image file", type="jpg")
        
        select_stage = False    
        gen_button = False

        if uploaded_file is not None:

            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Get image stage 
            if 'stage' not in st.session_state:
                st.session_state.stage = 'tPB2'
            

            # Display image:
            st.image(opencv_image, channels="BGR")
            gen_button = True
            select_stage = True
            st.markdown(
                f"""
                ##### Assigned stage: **{st.session_state.stage}**.
                """
            )
            

        option = st.selectbox(
            'Change assigned stage:',
            ('tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 
            't7', 't8', 't9+', 'tM', 'tSB', 'tB', 'tEB', 'tHB', 'empty'),
            disabled=not(select_stage))
        
        if st.button("Change stage", disabled=not(gen_button)):
            st.session_state.stage = option
            st.success(f'Done! Stage changed to {st.session_state.stage}')
            st.experimental_rerun()
            
        if st.button("Generate prediction", disabled=not(gen_button)):
            with st.spinner('Generating prediction...'):
                time.sleep(3)
            st.success('Done! Prediction saved in...')
else:

    img_list = [os.path.join(overview_folder[0], img) for img in natsorted(os.listdir(overview_folder[0]))]
    captions_list = ["Frame "+str(i+1) for i in range(len(img_list))]

    img = image_select(
        label="Score: XX.XX",
        images=img_list,
        captions=None,
        use_container_width = False
    )


