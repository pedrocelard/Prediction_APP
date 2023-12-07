import streamlit as st
import os
import cv2
import numpy as np
import io
import json

from natsort import natsorted

from streamlit_image_select import image_select
from PIL import Image

from classification.classifier import ImageClassifier
from diffusion.predict import Prediction
from scoring.scorer import Scorer
from utils.utils import get_stage_position

# Take whole screen with
st.set_page_config(layout="centered")
st.session_state.lang = 'en'

# Page first sentence
st.markdown(
    """
    #### Embryo development frame visualization
    """
)

# delete all session states with this keys
def del_session_states(keys):
    for key in keys:
        if (key in st.session_state):
            del st.session_state[key]


# Content control
showSequence = False
uploaded_file = None
gen_button = False
select_stage = False
st.session_state.check_manual_stage = False

placeholder = st.empty()

# Upload image
uploaded_file = st.file_uploader("Choose a image file", 
                                    type="jpg",
                                    on_change=del_session_states(["assigned_stage"]),
                                    key="new_case_uploader")


col1, col2 = st.columns(2)
       
with col1:
    if uploaded_file is not None:
   
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image = Image.open(io.BytesIO(file_bytes))
        # Get image stage 
        if 'assigned_stage' not in st.session_state:
            classifier = ImageClassifier()
            _, st.session_state.assigned_stage = classifier.classify_image(image)

        # Enable form if the user uploads an image
        gen_button = True
        select_stage = True

    else:
        # Show a mockup image
        opencv_image = Image.open("./media/overview_0_light/1.jpg")

    # Display image:
    st.image(opencv_image, channels="BGR", width=256)

with col2:
    # Show upload image classification result
    if uploaded_file is not None:
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
        
        num_samples = st.number_input("Number of overviews", 
                                      value=1, 
                                      min_value=1,
                                      max_value=5,
                                      step = 1,
                                      disabled=not(gen_button))

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

            scorer = Scorer()
            scoring = scorer.generate_scoring(output_path)
            del scorer 

            # Read the existing JSON case data from the file
            json_file_path = os.path.join(output_path,"case_info.json")
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

            overview_ranking = []
            for score in scoring:
                overview_ranking.append(score["case"])

                # Find the element with overview_id equal to "overview_1"
                for element in data["overview_info"]:
                    if (element["overview_id"] == score["case"]):
                        # Modify the score value
                        element["score"] = score["sequence_score"]
                        element["mse"] = score["mse"]
                        element["mag"] = score["mag"]
            
            data["overview_ranking"] = overview_ranking

            # Write the modified data back to the file
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)

                st.success(f'Done! Prediction saved to {output_path}'.replace("\\","/"))
                st.cache_data.clear()


