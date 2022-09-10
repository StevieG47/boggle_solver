import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from boggleSolver import *
import readline
from Trie import Trie
from boggle_helper import *
import pandas as pd

# Page title/headers
st.set_page_config(page_title = "Boggle Solver")
st.title("Boggle Solver")
st.subheader("Take a top down image of the board")
st.subheader("'Browse Files' -> Select Image")
st.subheader("'Get Words'")
button_example = st.button("Run Example")
st.markdown("---")

# Load keras model
global model
model_file = './augmented_letters_10epoch.h5' #all_binary_letters_75epoch.h5'
if not os.path.exists(model_file):
    with st.spinner("Downloading model"):
      model_url = 'https://drive.google.com/file/d/16SdU9YfFez9IxsOLEwLB295NOOUsL156/view?usp=sharingv'
      os.system('gdown --id 16SdU9YfFez9IxsOLEwLB295NOOUsL156')
model = load_model(model_file)

# Load dictionary
dictionary = Trie()
with open('allScrabbleWords.txt', 'r') as file:
    for i in file.read().split():
        dictionary.insert(i)

# Setup Input Fields
button_pressed = st.button("Get Words")
calculating_display = st.empty()
image_uploaded = False
imageLocation = st.empty()
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    imageLocation.image(img)
    image_uploaded = True
if button_example:
    img_path = 'images/example.jpg'
    img = Image.open(img_path).convert('RGB')
    image_uploaded = True
    button_pressed = True

# Process image on button press
if button_pressed:
    if image_uploaded:
        calculating_display.text("Getting Words...")

        #try:
            # Run image through boggleSolver to get board letters
        bs = boggleSolver(model,showImages=False,verbose=False)
        im = np.array(img)
        im = im[:, :, ::-1].copy() # rgb to bgr
        output_im,letters_out = bs.process_image(im)

        # Set output image display
        cv2.imwrite('output_im.png',output_im)
        out_im = Image.open('output_im.png')
        imageLocation.image(out_im)

        # Run boggle board solver and sort words longest to shortest
        found_words = solve_board(letters_out, dictionary)
        found_words = sorted(found_words,key=len,reverse=True)

        # Display found words in table
        df = pd.DataFrame(found_words)
        df.columns = ['WORDS']
        df.index += 1
        st.table(df)

        # Update display message
        calculating_display.text("Done!")
        #except:
        #    calculating_display.text("Error Getting Words")
        #    imageLocation = st.empty()
    else:
        st.write('No Image Uploaded!')	