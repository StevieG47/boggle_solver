import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from boggleSolver import *
#from keras.models import load_model
from tensorflow.keras.models import load_model
import cv2
import time
import readline
from Trie import Trie
from boggle_helper import *

# Load test image
im_file = 'images/example.jpg'
im = cv2.imread(im_file)

# Load keras model
global model
model_file = 'models/augmented_letters_10epoch.h5' #all_binary_letters_75epoch.h5'
model = load_model(model_file)

# Run boggle solver
bs = boggleSolver(model,showImages=False,verbose=False)
output_im, letters_out = bs.process_image(im)

# Run boggle board solver
dictionary = Trie()
with open('allScrabbleWords.txt', 'r') as file:
    for i in file.read().split():
        dictionary.insert(i)
solution = solve_board(letters_out, dictionary)
print(solution)

# Show image
imshow(output_im,'Output')