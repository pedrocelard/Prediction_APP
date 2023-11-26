import streamlit as st
import os
import cv2
import numpy as np
import time
import io

from natsort import natsorted

from streamlit_image_select import image_select
from PIL import Image

from classification.classifier import ImageClassifier

# Take whole screen with
st.set_page_config(layout="centered")

# Page first sentence
st.markdown(
    """
    #### Embryo development frame visualization
    """
)

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
        
        # Get image stage 
        if 'assigned_stage' not in st.session_state:
            classifier = ImageClassifier()
            _, st.session_state.assigned_stage = classifier.classify_image(Image.open(io.BytesIO(file_bytes)))
            
    else:
        opencv_image = Image.open("./media/overview_0_light/1.jpg")

    # Display image:
    st.image(opencv_image, channels="BGR", width=256)
    gen_button = True
    select_stage = True

with col2:
    if uploaded_file is not None:
        st.markdown(
                f"""
                ##### Assigned stage: **{st.session_state.assigned_stage}**.
                """
            )

    with st.form("my_form"):
        use_manual_stage = st.checkbox('Use manual stage', 
                                       key = "check_manual_stage",
                                       disabled=not(gen_button))

        option = st.selectbox(
            'Change assigned stage to:',
            ('tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 
            't7', 't8', 't9+', 'tM', 'tSB', 'tB', 'tEB', 'tHB', 'empty'),
            disabled=not(gen_button),
            )
        
        # if st.button("Change stage", disabled=not(gen_button)):
        #     st.session_state.manual_stage = option
        #     st.experimental_rerun()
        
        gen = st.form_submit_button('Generate prediction', disabled=not(gen_button))
    
    if gen:
        with st.spinner('Generating prediction...'):
            print("GENERANDO!")
            time.sleep(3)
        st.success('Done! Prediction saved to...')