import streamlit as st
import os
import cv2
import numpy as np
import json
import time
import io

from natsort import natsorted
from streamlit_image_select import image_select
from PIL import Image

from classification.classifier import ImageClassifier
from diffusion.predict import Prediction
from utils.utils import get_stage_position

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

# delete all session states with this keys
def del_session_states(keys):
    for key in keys:
        if (key in st.session_state):
            del st.session_state[key]

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

# Mockup for case 0 
if(selected_filename != "case_0"):
    # Check if the case has a time-line, if not show initial options
    if(data["timeline_info"] is None):

        ov_col1, ov_col2, ov_col3 = st.columns([1.5,1,0.5])
        with ov_col1:
            st.error('''This case does not have a timeline. 
                    Please use controls below to create one.''')

            st.markdown('''
                        **In order to create a timeline, first you have to select an overview of this case.**

                        **Then, you will be able to upload new images to create alternative timelines.**
                        ''')
                     
            # Select an overview path
            filenames = [dir for dir in os.listdir(casename) if (os.path.isdir(os.path.join(casename, dir)))]
            selected_filename = st.selectbox('Select an overview', filenames, key="IS_over_sel")
            filename = os.path.join(casename, selected_filename).replace("\\","/")
            st.write('You selected `%s`' % filename)

            img_list = [os.path.join(filename, img) for img in natsorted(os.listdir(filename))]

        if (len(img_list) != 0):
            image1 = Image.open(img_list[0])
            # Get the size of the first image
            width1, height1 = image1.size

            # Calculate the width of the concatenated image
            concatenated_width = width1*len(img_list)

            # Create a new image with the calculated width and the shared height
            concatenated_image = Image.new("RGB", (concatenated_width, height1))

            for idx, conc_img in enumerate(img_list):

                # Paste the first image at the left side of the new image
                concatenated_image.paste(Image.open(conc_img), (width1*idx, 0))


            st.image(concatenated_image)
        
            if st.button(f'Generate timeline using {selected_filename}'):
                with st.spinner("Generating timeline, please wait..."):
                    time.sleep(3)
                    print(f"GENERATE TIMELINE USING {selected_filename}")   
    else:

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
            uploaded_file = st.file_uploader("Choose a image file", type="jpg",
                                        on_change=del_session_states(["assigned_stage"]),
                                        key="new_timeline_uploader")
            
            select_stage = False    
            gen_button = False

            if uploaded_file is not None:

                # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                image = Image.open(io.BytesIO(file_bytes))

                # Get image stage 
                if 'assigned_stage' not in st.session_state:
                    classifier = ImageClassifier()
                    _, st.session_state.assigned_stage = classifier.classify_image(image)
                

                # Display image:
                st.image(opencv_image, channels="BGR")
                gen_button = True
                select_stage = True
                st.markdown(
                    f"""
                    ##### Assigned stage: **{st.session_state.assigned_stage}**.
                    """
                )
                
            # Change stage and generate prediction form
            with st.form("my_form"):
                # Use selected stage to generate content instead of automatic classification
                use_manual_stage = st.checkbox('Use manual stage', 
                                            key = "check_manual_stage",
                                            disabled=not(gen_button))

                # Manual stage selection
                option = st.selectbox(
                    'Change assigned stage to:',
                    ('tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 
                    't7', 't8', 't9+', 'tM', 'tSB', 'tB', 'tEB', 'tHB', 'empty'),
                    disabled=not(gen_button),
                    )
                
                num_samples = 1

                # Submit form an generate content
                gen = st.form_submit_button('Generate prediction', disabled=not(gen_button))


            if gen:
                with st.spinner("Generating prediction, please wait..."):
                    if(use_manual_stage):
                        cond_frames = get_stage_position([option])
                    else:
                        cond_frames = get_stage_position([st.session_state.assigned_stage])

                    pred = Prediction(cond_frames=cond_frames, num_samples=num_samples)
                    output_path = pred.generate_prediction(cond_images=[image])
                    del pred
                st.success(f'Done! Prediction saved to {output_path}'.replace("\\","/"))
                st.cache_data.clear()
else:

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


