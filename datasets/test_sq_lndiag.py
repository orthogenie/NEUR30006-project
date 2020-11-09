# This script was written by Mabrur Rahman (2020). Last modified November 2020.
# 
# This script creates a csv file of 1-dimension test stimuli. These stimuli are 
# simple squares or diagonal lines that cut through them at 2 angles: 
# up and down diagonal
# 
# The size of the dataset can be modified 


import random
import numpy as np
import cv2
from skimage.draw import line_aa
import csv


# Ask for the name of the dataset and how many datapoints 
csv_file = input("Enter a name for a data file (.csv): ")
data_size = int(input("How many data points would you like? "))

# Create an empty .csv file
with open(csv_file, 'wb') as data_file:
    pass


# Values for how big the square will be and the length of the line
SM_DIM = 6
LRG_DIM = 12
LN_LEN = 8

# This loop will run for however many datapoints was requested
for stim in range(data_size):

    # Randomly decide whether it will be a square or line
    dim = random.randint(0, 1)

    # If it's a square...
    if not dim:
        
        # Blank array for square
        sq_blank = np.zeros(shape=[28, 28])

        # Randomly decide whether the square will be small or large
        isbig = random.randint(0, 1)
        thick = 1
        
        # Assign the values depending on the square size
        if isbig:
            sq_size = LRG_DIM
            x_pt1 = random.randint(1, 26 - LRG_DIM)
            y_pt1 = random.randint(1, 26 - LRG_DIM)

            label = np.array([5])
        else:
            sq_size = SM_DIM
            x_pt1 = random.randint(1, 26 - SM_DIM)
            y_pt1 = random.randint(1, 26 - SM_DIM) 

            label = np.array([1])
        x_pt2 = x_pt1 + sq_size
        y_pt2 = y_pt1 + sq_size

        # Create the square
        image = cv2.rectangle(sq_blank, pt1=(x_pt1, y_pt1), pt2=(x_pt2, y_pt2), color=(255, 255, 255), thickness=thick)

    # If it's a line...
    else: 
        
        # Blank array for line
        ln_blank = np.zeros([28, 28])

        # Randomly decide whether the line will be up or down diagonal
        orientation = random.randint(0, 1)
        
        # Assign values depending on the line orientation
        if orientation:
            # up diagonal (/)
            
            ln_col_start = random.randint(1, 18)
            ln_row_start = random.randint(9, 26)
            ln_col_end = ln_col_start + LN_LEN
            ln_row_end = ln_row_start - LN_LEN

            label = np.array([4])
        else:
            # down diagonal (\)

            ln_col_start = random.randint(1, 18)
            ln_row_start = random.randint(1, 18)
            ln_col_end = ln_col_start + LN_LEN
            ln_row_end = ln_row_start + LN_LEN

            label = np.array([2])

        # Create the line
        rr, cc, val = line_aa(ln_row_start, ln_col_start, ln_row_end, ln_col_end)
        ln_blank[rr, cc] = val * 255
        image = ln_blank

    # Resize the image into 1D
    stimulus = np.reshape(image, (1,784))
    data = np.append(label, stimulus)

    # Append the stimulus to the .csv file
    with open(csv_file, 'a', newline='') as data_file:
        df_writer = csv.writer(data_file)
        df_writer.writerow(data)

print("All done (:")