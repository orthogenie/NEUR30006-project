# This script was written by Mabrur Rahman (2020). Last modified November 2020.
# 
# This script creates a csv file of 1-dimension train stimuli. These stimuli 
# are simple squares with straight lines that cut through them at 2 angles: 
# horizontal and vertical
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


# Values for how big the square will be
SM_DIM = 6
LRG_DIM = 12

# This loop will run for however many datapoints was requested
for stim in range(data_size):
    
    # Blank arrays where the square and line will be drawn
    sq_blank = np.zeros(shape=[28, 28])
    ln_blank = np.zeros([28, 28])

    # Randomly decide whether it will be a small or large square
    isbig = random.randint(0, 1)
    thick = 1
    
    # Assign points in relation to whether the square will be small or large
    if isbig:    
        sq_size = LRG_DIM
        x_pt1 = random.randint(1, 26 - LRG_DIM)
        y_pt1 = random.randint(1, 26 - LRG_DIM)
    else:      
        sq_size = SM_DIM
        x_pt1 = random.randint(1, 26 - SM_DIM)
        y_pt1 = random.randint(1, 26 - SM_DIM) 
    x_pt2 = x_pt1 + sq_size
    y_pt2 = y_pt1 + sq_size

    # Create the square
    cv2.rectangle(sq_blank, pt1=(x_pt1, y_pt1), pt2=(x_pt2, y_pt2), color=(255, 255, 255), thickness=thick)


    # This constant keeps track of the thickness of the square, and makes sure 
    # that there are no overlapping pixels
    LN_OFFSET = thick

    # Randomly assign whether the line will be horizontal or vertical
    orientation = random.randint(0, 1)
    
    # Each line point is reliant on the size of the square and the orientation 
    # of the line. We can also use this point to assign the labels
    if orientation and isbig:
        # large and vertical 
        
        ln_col_start = x_pt1 + (LRG_DIM//2)
        ln_row_start = y_pt1 + LN_OFFSET
        ln_col_end = ln_col_start
        ln_row_end = y_pt2 - LN_OFFSET

        label = np.array([3])

    elif orientation and not isbig:
        # small and vertical 
        
        ln_col_start = x_pt1 + (SM_DIM//2)
        ln_row_start = y_pt1 + LN_OFFSET
        ln_col_end = ln_col_start
        ln_row_end = y_pt2 - LN_OFFSET

        label = np.array([1])

    elif not orientation and isbig:
        # large and horizontal
        
        ln_col_start = x_pt1 + LN_OFFSET
        ln_row_start = y_pt1 + (LRG_DIM//2)
        ln_col_end = x_pt2 - LN_OFFSET
        ln_row_end = ln_row_start

        label = np.array([2])
    
    else:
        # small and horizontal
        
        ln_col_start = x_pt1 + LN_OFFSET
        ln_row_start = y_pt1 + (SM_DIM//2)
        ln_col_end = x_pt2 - LN_OFFSET
        ln_row_end = ln_row_start

        label = np.array([0])

    # Create the line
    rr, cc, val = line_aa(ln_row_start, ln_col_start, ln_row_end, ln_col_end)
    ln_blank[rr, cc] = val * 255

    # Combine the line and the square, and compress it into a 1-D "image"
    image = sq_blank + ln_blank
    stimulus = np.reshape(image, (1,784))
    data = np.append(label, stimulus)

    # Append it into the .csv file
    with open(csv_file, 'a', newline='') as data_file:
        df_writer = csv.writer(data_file)
        df_writer.writerow(data)

print("All done (:")