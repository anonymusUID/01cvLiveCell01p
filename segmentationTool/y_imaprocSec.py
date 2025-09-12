#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 04:26:41 2025

@author: surajit
"""

import sys
import os
from scipy.stats import skew, kurtosis
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
import math
from scipy.spatial import distance_matrix
from numba import jit, njit, prange
import pickle

# Numba-optimized functions
@njit
def ske_kur_numba(patch):
    variance = np.var(patch.flatten())
    if variance < 1e-10:
        return 0.0, 0.0
    else:
        data = patch.flatten()
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0, 0.0
        
        skew_val = np.sum(((data - mean) / std) ** 3) / n
        kurt_val = np.sum(((data - mean) / std) ** 4) / n - 3
        return skew_val, kurt_val

@njit
def check_majority_greater_than_127_numba(items):
    if len(items) == 0:
        return 0
    
    count_greater = 0
    max_item = items[0]
    min_item = items[0]
    
    for item in items:
        if item > 80:
            count_greater += 1
        if item > max_item:
            max_item = item
        if item < min_item:
            min_item = item
    
    if count_greater >= len(items) / 2:
        return max_item
    else:
        return min_item

@njit
def get_all_modes_numba(data):
    if len(data) == 0:
        return 0
    
    unique_values = []
    counts = []
    
    for val in data:
        found = False
        for i in range(len(unique_values)):
            if unique_values[i] == val:
                counts[i] += 1
                found = True
                break
        if not found:
            unique_values.append(val)
            counts.append(1)
    
    max_count = 0
    for count in counts:
        if count > max_count:
            max_count = count
    
    modes = []
    for i in range(len(unique_values)):
        if counts[i] == max_count:
            modes.append(unique_values[i])
    
    return check_majority_greater_than_127_numba(modes)

@njit(parallel=True)
def adjacency_shift_numba(values, flag):
    rows, cols = values.shape
    neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
    numerator = 0.0
    S0 = 0
    
    for i in prange(rows):
        for j in prange(cols):
            for offset in neighbors_offsets:
                ni, nj = i + offset[0], j + offset[1]
                if 0 <= ni < rows and 0 <= nj < cols:
                    numerator += (values[i, j] - values[ni, nj]) ** 2
                    S0 += 1
    
    return numerator / 2

@njit
def calculate_standard_deviation_numba(patch):
    return np.std(patch)

@njit
def calculate_morans_i_numba(patch):
    patch_mean = np.mean(patch)
    deviations = patch - patch_mean
    numerator = 0.0
    denominator = np.sum(deviations**2)

    height, width = patch.shape
    for y in range(height):
        for x in range(width):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    numerator += deviations[y, x] * deviations[ny, nx]

    morans_i = (numerator / denominator) * (height * width / (4 * height * width - 2 * (height + width))) if denominator != 0 else 0
    return morans_i

@njit
def calculate_variogram_numba(patch):
    pixels = patch.flatten()
    n = len(pixels)
    total = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = i // patch.shape[1], i % patch.shape[1]
            xj, yj = j // patch.shape[1], j % patch.shape[1]
            distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if distance > 0:
                total += (pixels[i] - pixels[j])**2 / (2 * distance)
                count += 1
    
    return total / count if count > 0 else 0

@njit
def calculate_general_stat_numba(patch):
    mn = np.mean(patch)
    mdn = np.median(patch)
    md = get_all_modes_numba(patch.flatten())
    return mn, mdn, md

# Replace original functions with Numba versions
def ske_kur(patch):
    return ske_kur_numba(patch)

def get_all_modes(data):
    return get_all_modes_numba(np.array(data))

def adjacency_shift(values, flag):
    return adjacency_shift_numba(np.array(values, dtype=np.float64), flag)

def calculate_morans_i(patch):
    return calculate_morans_i_numba(np.array(patch, dtype=np.float64))

def calculate_variogram(patch):
    return calculate_variogram_numba(np.array(patch, dtype=np.float64))

def calculate_standard_deviation(patch):
    return calculate_standard_deviation_numba(np.array(patch, dtype=np.float64))

def calculate_general_stat(patch):
    return calculate_general_stat_numba(np.array(patch, dtype=np.float64))

# Rest of your original code remains unchanged...



##################################################################
###################################################################

import sys
import os
from scipy.stats import skew, kurtosis
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
import math
from scipy.spatial import distance_matrix

from z_stat import * # calculate_morans_i, calculate_variogram, calculate_standard_deviation, calculate_general_stat

from tqdm import tqdm
import cProfile

print("Process Started. Please wait till end...")

upperbound=255
lowerbound=5
RC_thresh=0.1
Dep_thresh=80
sv1=110
sv2=0.273 #80
sv3=0
sv4=sv5=0.5
color_spec="HIP Perceived Grayscale Analysis"
max_std_dev = max_moran_I=max_dnsi=0
Dep_thresh_flag=0
max_varigram = 1
flg_hough=0

LL_vario = 1.0          # Maximum value of the sigmoid function
KK_vario = 10.0         # Steepness of the sigmoid curve
x0_vario = 0.8 

LL_dnsi = 1.0          # Maximum value of the sigmoid function
KK_dnsi = 10.0         # Steepness of the sigmoid curve
x0_dnsi = 0.8 






from scipy.ndimage import generic_filter

def homogeneous_masking(image, output_path, flg, neighborhood_size=3, background_color='black'):
    global lowerbound, Dep_thresh, RC_thresh, Dep_thresh_flag, max_varigram, L, K, x0

    # Ensure the image is loaded properly
    if image is None or image.size == 0:
        raise ValueError("Error: Image is empty or not loaded correctly.")

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure uint8 type
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    height, width = image.shape
    half_size = neighborhood_size // 2

    # Initialize new image for filtering
    new_img = image.copy()

    # Apply a local standard deviation filter
    def local_std(patch):
        return np.std(patch)

    std_dev_map = generic_filter(image, local_std, size=(neighborhood_size, neighborhood_size))

    # Suppress pixels with low standard deviation
    new_img[std_dev_map <= lowerbound] = 0 

    # Show processed image
    """
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')
    plt.show()
    """

    gray = new_img
    '''
    def hough_transform_remove_circles(gray):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Dynamically determine circle radius limits
        minRadius = max(5, int(min(height, width) * 0.02))  
        maxRadius = max(6, int(min(height, width) * 0.1))  

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # Create a copy of the original image
        result = gray.copy()

        # If circles are detected, replace them with black color
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                # Replace the circle region with black (0 intensity)
                cv2.circle(result, center, radius, 0, -1)

        else:
            print("Warning: No circles detected.")
        return result


    '''
    
    def hough_transform_remove_circles(gray):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Get image dimensions
        height, width = gray.shape[:2]

        # Dynamically determine circle radius limits
        minRadius = max(5, int(min(height, width) * 0.02))  
        maxRadius = max(6, int(min(height, width) * 0.1))  

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # Create a copy of the original image
        result = gray.copy()

        # If circles are detected, process them
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, radius = circle  # Center coordinates and radius

                # Define a circular mask for the central region (inner 50% of radius)
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), max(1, radius // 2), 255, -1)  # Create circular mask
                
                # Extract pixel values inside the mask
                circle_pixels = gray[mask == 255]

                # Compute the average intensity of the central region
                avg_intensity = np.mean(circle_pixels) if circle_pixels.size > 0 else 0

                # Remove circle only if the average intensity > 200
                print("avg int****", avg_intensity)
                if avg_intensity > a:
                    cv2.circle(result, (cx, cy), radius, 0, -1)  # Fill circle with black

        else:
            print("Warning: No circles detected.")

        return result
    
    
    if(flg==1):
        result = hough_transform_remove_circles(gray)
    else:
        result=new_img
    # Save the modified image
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to {output_path}")

    # Display the output
    """
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()
    """
    return result





    




def gap_tunning(original_image):
    canny = cv2.Canny(original_image, threshold1=100, threshold2=200)
    canny_copy = canny.copy()
    #output_image = np.ones_like(original_image) * 255
    #output_image[edges_copy == 255] = 255
    original_image[canny_copy == 255] = 255

def remaining_min(left_seg):
    return min(left_seg, 255-left_seg)
    




import numpy as np

def adjacency_shift(values, flag):
    # Ensure values is a NumPy array
    values = np.array(values, dtype=np.float64)
    
    # Check if it's 1D and reshape it to 2D (e.g., (1, n) or (n, 1) depending on your patch)
    if values.ndim == 1:
        values = values.reshape(1, -1)  # Reshapes the 1D array to 2D (1, n) array
        #print(f"Input reshaped to 2D: {values.shape}")
    
    # Ensure it's a 2D array
    if values.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    
    rows, cols = values.shape
    neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Horizontal & Vertical
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
    numerator = 0
    S0 = 0
    
    # Iterate over all pixels
    for i in range(rows):
        for j in range(cols):
            for offset in neighbors_offsets:
                ni, nj = i + offset[0], j + offset[1]

                # Check if neighbor is within bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    numerator = numerator + (values[i, j] - values[ni, nj]) ** 2
                    S0 += 1
    
    valley_well = numerator / 2
    return valley_well



# Degree of centrality function
def calculate_degree_centrality(patch):
    n = patch.shape[0]  # n x n patch
    central_pixel = patch[n//2, n//2]  # Central pixel intensity value
    
    # Define directions (including diagonals) and corresponding offset positions
    directions = {
        "up": patch[:n//2, n//2],              # Top half
        "down": patch[n//2+1:, n//2],         # Bottom half
        "left": patch[n//2, :n//2],           # Left half
        "right": patch[n//2, n//2+1:],        # Right half
        "top-left": patch[:n//2, :n//2],      # Top-left diagonal
        "top-right": patch[:n//2, n//2+1:],   # Top-right diagonal
        "bottom-left": patch[n//2+1:, :n//2], # Bottom-left diagonal
        "bottom-right": patch[n//2+1:, n//2+1:] # Bottom-right diagonal
    }
    
    # Thresholds
    intensity_threshold = 50
    adjacency_threshold = 500
    
    dissimilar_pixels = 0  # Counter for dissimilar pixels
    
    for direction, values in directions.items():
        # Calculate the median intensity in each direction
        median_intensity = np.median(values)
        
        # Calculate adjacency shift for the current direction (using 2D array of intensity values)
        adj_shift_value = adjacency_shift(values, flag=0)  # flag=0 for normal operation
        
        # Check for intensity difference
        intensity_diff = abs(central_pixel - median_intensity)
        
        # Check for adjacency shift threshold
        if intensity_diff > intensity_threshold or adj_shift_value > adjacency_threshold:
            dissimilar_pixels += 1
    
    # Degree of centrality (higher is more central)
    degree_centrality = (8 - dissimilar_pixels)  # 8 directions (4 main directions + 4 diagonals)
    
    return degree_centrality































def read_sample(file_path, image, neighborhood_size):
    global upperbound, lowerbound, RC_thresh, Dep_thresh, color_spec, Dep_thresh_flag
  
    # Load the Excel file
    file_path = file_path  # Replace with your actual file path
    df = pd.read_excel(file_path)
    #print(df)
    # Ensure the columns 'R', 'G', 'B' exist in the DataFrame
    for col in ['GS1S','GS2S','GS1RC','GS2RC','GS1Dep','GS2Dep']:
        # Calculate mean, standard deviation, and skewness
        mean_std = df[['GS1S','GS2S','GS1RC','GS2RC','GS1Dep','GS2Dep']].describe()
        print("Other Statistical Data:" "\n========================\n", mean_std)
        



        
        RC_thresh=round(mean_std.loc["min"]["GS2RC"],2)+(round(max_moran_I,2)-round(mean_std.loc["min"]["GS2RC"],2))*sv4
        if (Dep_thresh_flag==1):
            Dep_thresh_controll=10*math.log(round(max_varigram,2)-round(mean_std.loc["min"]["GS2Dep"],2))*sv5
            Dep_thresh=round(mean_std.loc["min"]["GS2Dep"],2)+ round(3.5 * round(mean_std.loc["std"]["GS1S"],2),2) +   Dep_thresh_controll     
        else:  
            Dep_thresh=round(max_varigram/2) *sv5 
            print("////////////",Dep_thresh)
        
         
        # Calculate the threshold for white pixels
        def calculate_upperbound():
                global upperbound, lowerbound
                upperbound=20

        # Calculate the threshold for black pixels
        def calculate_lowerbound():
            global upperbound, lowerbound 
            if color_spec=="HIP Perceived Grayscale Analysis":
                lowerbound=round(mean_std.loc["mean"]["GS1S"],2) + round(3.5 * round(mean_std.loc["std"]["GS1S"],2),2)
                
            else:
                lowerbound=round(mean_std.loc["mean"]["GS2S"],2) + round(3.5 * round(mean_std.loc["std"]["GS2S"],2),2) 
                
        
        
        calculate_upperbound()     
        calculate_lowerbound()
        
        lowerbound=lowerbound+lowerbound*sv3*2
        
        
        print("*************","\nRelational Threshold Set:", round(RC_thresh,2), "\nDependancy Threshold Set:", round(Dep_thresh,2)) 
        print("*************","\nlowerbound:", lowerbound, "\nupperbound:", upperbound) 
        print("\n*****************\nsv3:",sv3,"sv4:",sv4,"sv5:",sv5)
        
        return mean_std






def display_matplt(image, new_img):
    # Plot original and classified image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title("new_img (Texture vs. Background)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("xxx.png", bbox_inches='tight', dpi=300)
    plt.show()






def Eor_dia(binary_mask,output_file):
    '''
    mask = cv2.imread(output_file, cv2.IMREAD_GRAYSCALE)


    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    '''
    
    # Step 1: Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Create a filled mask
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)



    # Compute original area
    original_area = np.sum(binary_mask > 0)
    target_area = original_area * 0.40  # Retain 90% (erode 10%)

    # Define a structuring element (kernel)
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for shape preservation

    # Iteratively erode while maintaining shape
    eroded_mask = filled_mask.copy()

    while True:
        temp_mask = cv2.erode(eroded_mask, kernel, iterations=1)
        new_area = np.sum(temp_mask > 0)

        # Stop when the remaining area reaches the target
        if new_area <= target_area:
            break  

        eroded_mask = temp_mask  # Update the eroded object

    # Save or visualize the result
    cv2.imwrite('Final_'+output_file, eroded_mask)
    cv2.imshow('Eroded Object (10%)', eroded_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







countcount = 0

'''
def fl_math(intensity_value,mean, median, modd, sv1,sv2):
    global countcount
    
    input_px=intensity_value
    #if(dc>3):
    input_px = max(intensity_value, mean, median, modd)

    a = int(sv1) - int(sv2 * remaining_min(sv1))  # DEFAULT 80
    #print("a is=======================", a)
    alpha = int(sv1)  # c in case of u_d, DEFAULT 110
    b = alpha  # DEFAULT 110
    beta = alpha  # a in case of u_b, DEFAULT 110
    c = int(sv1) + int(sv2 * remaining_min(sv1))  # DEFAULT 140

    if countcount == 0:
        print("a:", a, "alpha:", alpha, "b:", b, "beta:", beta, "c:", c)
        countcount += 1
    
    def u_d(input_px):
        if input_px <= a:
            output = 1
        else:
            output = max((alpha - input_px) / (alpha - a), 0)
        return output

    def u_g(input_px):
        output = max(min((input_px - a) / (b - a), (c - input_px) / (c - b)), 0)
        return output

    def u_b(input_px):
        if input_px >= c:
            output = 1
        else:
            output = max((input_px - beta) / (c - beta), 0)
        return output

    def process_pixel(intensity_value):
        v_d = 0
        v_g = 127
        v_b = 255

        num = v_d * u_d(input_px) + v_g * u_g(input_px) + v_b * u_b(input_px)
        den = u_d(input_px) + u_g(input_px) + u_b(input_px)
        output = int(num / den)

        return int(output)

    # Compute degree centrality for the current pixel (row, col)
    #degree_centrality = calculate_degree_centrality(row, col, patch)
    #print(f"Degree Centrality of pixel ({row},{col}): {degree_centrality}")

    # Adjust the input based on degree centrality
    #adjusted_input = input_px * (1 + degree_centrality * 0.05)  # Adjust by degree centrality
    
    x = process_pixel(int(input_px))
    return x

'''

@njit
def fl_math(intensity_value, mean, median, modd, sv1, sv2):
    a = int(sv1) - int(sv2 * (255 - sv1 if sv1 > 127 else sv1))
    alpha = int(sv1)
    b = alpha
    beta = alpha
    c = int(sv1) + int(sv2 * (255 - sv1 if sv1 > 127 else sv1))
    
    input_px = max(intensity_value, mean, median, modd)
    
    # u_d calculation
    if input_px <= a:
        u_d_val = 1.0
    else:
        u_d_val = max((alpha - input_px) / (alpha - a), 0.0)
    
    # u_g calculation
    u_g_val = max(min((input_px - a) / (b - a), (c - input_px) / (c - b)), 0.0)
    
    # u_b calculation
    if input_px >= c:
        u_b_val = 1.0
    else:
        u_b_val = max((input_px - beta) / (c - beta), 0.0)
    
    # Final calculation
    v_d = 0 * u_d_val
    v_g = 127 * u_g_val
    v_b = 255 * u_b_val
    
    denom = u_d_val + u_g_val + u_b_val
    if denom > 1e-10:
        return int((v_d + v_g + v_b) / denom)
    else:
        return 0






    






@njit
def vote_numba(intensity_value, fl_pix_val, vario_std, 
              R_dnsi_by_std, G_dnsi_by_std, B_dnsi_by_std,
              RR, GG, BB):
    if ((GG < 100 and GG < RR and GG < BB) and 
        (R_dnsi_by_std > G_dnsi_by_std and G_dnsi_by_std > B_dnsi_by_std)):
        return 0
    else:
        return 255

    





 



@njit(parallel=True)
def process_pixels_numba(
    image, new_img, x_coords, y_coords, 
    padded_image, R_padded, G_padded, B_padded,
    patch_radius, sv1, sv2, sv4, sv5
):
    height, width = image.shape
    count = 0
    a = int(sv1) - int(sv2 * (255 - sv1 if sv1 > 127 else sv1))
    
    for idx in prange(len(x_coords)):
        x = x_coords[idx]
        y = y_coords[idx]
        
        if image[y, x] > a:
            patch = padded_image[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
            R_patch = R_padded[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
            G_patch = G_padded[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
            B_patch = B_padded[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
            
            intensity = image[y, x]
            mean = np.mean(patch)
            median = np.median(patch)
            modd = get_all_modes_numba(patch.flatten())
            
            fl_pix_val = fl_math(intensity, mean, median, modd, sv1, sv2)
            
            if fl_pix_val < a:
                new_img[y, x] = 0
            elif a <= fl_pix_val <= 140:
                std_dev = calculate_standard_deviation_numba(patch)
                if std_dev < 1e-10:
                    new_img[y, x] = 0
                    continue
                    
                vario = calculate_variogram_numba(patch)
                vario_std = vario / std_dev
                
                if vario_std < sv4:
                    new_img[y, x] = 0
                else:
                    R_std = calculate_standard_deviation_numba(R_patch)
                    G_std = calculate_standard_deviation_numba(G_patch)
                    B_std = calculate_standard_deviation_numba(B_patch)
                    
                    R_dnsi = (adjacency_shift_numba(R_patch, 0) / R_std 
                             if R_std > 1e-10 else 0)
                    G_dnsi = (adjacency_shift_numba(G_patch, 0) / G_std 
                             if G_std > 1e-10 else 0)
                    B_dnsi = (adjacency_shift_numba(B_patch, 0) / B_std 
                             if B_std > 1e-10 else 0)
                    
                    new_img[y, x] = vote_numba(
                        intensity, fl_pix_val, vario_std,
                        R_dnsi, G_dnsi, B_dnsi,
                        R_patch.mean(), G_patch.mean(), B_patch.mean()
                    )
                    count += 1
            else:
                new_img[y, x] = 255
        else:
            new_img[y, x] = 0
    
    return count

def extract_and_classify_image_gray(img_colr, image, output_file, neighborhood_size, sv1, sv2,output_dir,loc_specifier):
    ccc = 0
    a = int(sv1) - int(sv2 * (255 - sv1 if sv1 > 127 else sv1))
    
    img_colr_R = img_colr[:,:,0].astype(float)
    img_colr_G = img_colr[:,:,1].astype(float)
    img_colr_B = img_colr[:,:,2].astype(float)
    
    image = homogeneous_masking(image, "xxx_Houg.jpg", flg_hough, neighborhood_size, background_color='black')
    zero_count = np.sum(image == 0)
    
    new_img = image.copy()
    y_coords, x_coords = np.where(image > 0)
    height, width = image.shape

    patch_radius = neighborhood_size // 2
    padded_image = np.pad(image, pad_width=patch_radius, mode='reflect')
    R_padded_image = np.pad(img_colr_R, pad_width=patch_radius, mode='reflect')
    G_padded_image = np.pad(img_colr_G, pad_width=patch_radius, mode='reflect')    
    B_padded_image = np.pad(img_colr_B, pad_width=patch_radius, mode='reflect')
    
    # Process pixels using Numba-optimized function
    ccc = process_pixels_numba(
        image, new_img, x_coords, y_coords,
        padded_image, R_padded_image, G_padded_image, B_padded_image,
        patch_radius, sv1, sv2, sv4, sv5
    )
    
    print(ccc)
    # This now saves the image to the correct full path passed into the function
    cv2.imwrite(output_file, new_img)
    print(f"Processed image saved to {output_file}")

    # Save pickle version of the results
    pickle_data = {
        "image": new_img,
        "classification_count": ccc,
        "params": {
            "neighborhood_size": neighborhood_size,
            "sv1": sv1,
            "sv2": sv2
        }
    }
    
    # This part is already correct!
    pickle_path = os.path.join(output_dir, f"{loc_specifier}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_data, f)
        
    print(f"[SUCCESS] Pickle file saved at {pickle_path}")

   
    #display_matplt(image, new_img)



import sys
import traceback


def main():
    try:
        # 1. Your argument parsing is correct.
        excel_file = sys.argv[1]
        image_path = sys.argv[2]
        color_spec = sys.argv[3]
        neighborhood_size = int(sys.argv[4])
        sv1 = float(sys.argv[5])
        sv2 = float(sys.argv[6])
        sv3 = float(sys.argv[7])
        sv4 = float(sys.argv[8])
        sv5 = float(sys.argv[9])
        loc_specifier = sys.argv[10]
        output_dir = sys.argv[11]

        # 2. (Good Practice) Create the full output directory path once, right here.
        os.makedirs(output_dir, exist_ok=True)
        print(f"All outputs will be saved to: {output_dir}")

        # 3. --- FIX 1: Construct ALL output paths using the 'output_dir' variable ---
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        
        # This now creates the correct, full path for the image file.
        output_image_file = os.path.join(output_dir, f"{loc_specifier}_processed_image{file_extension}")


        # 4. Load and process the image as before
        image_original = cv2.imread(image_path)
        if image_original is None:
            print("Error: Image could not be loaded.")
            return # Exit if image fails to load

        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        height, width, _ = image_original.shape
        R = image_original[:,:,0].astype(float)
        G = image_original[:,:,1].astype(float)
        B = image_original[:,:,2].astype(float)
        
        if color_spec == "HIP Perceived Grayscale Analysis":
            image = np.round(0.299 * R + 0.587 * G + 0.114 * B)
        else:
            image = np.round(R/3 + G/3 + B/3)

        tet = read_sample(excel_file, image, neighborhood_size)

        # 5. --- FIX 2: Call the function with the CORRECT variable names and paths ---
        extract_and_classify_image_gray(
            image_original,
            image,
            output_image_file,  # Pass the corrected image path
            neighborhood_size,
            sv1,
            sv2,
            output_dir,         # Pass the correct directory variable
            loc_specifier
        )

    except IndexError:
        print("Error: Incorrect number of command-line arguments provided.")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print("An error occurred during processing:")
        traceback.print_exc()
        sys.exit(1)    
    

if __name__ == "__main__":
    main()
    #cProfile.run('main()') #main()



