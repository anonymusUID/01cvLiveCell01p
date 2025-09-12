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



'''
def manual_skewness(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation (n-1)
    skew = (np.sum((data - mean)**3) / n) / (std**3)
    return skew

def manual_kurtosis(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    kurt = (np.sum((data - mean)**4) / n )/ (std**4 - 3)
    return kurt

print(f"Manual Skewness: {manual_skewness(data):.2f}")         # Output: -0.27
print(f"Manual Excess Kurtosis: {manual_kurtosis(data):.2f}")  # Output: -0.88
'''

def ske_kur(patch):
    variance = np.var(patch.flatten())
    if variance < 1e-10:  # Threshold for "nearly constant"
        skew_val = 0  # Skewness is 0 for symmetric data
        kurt_val = 0  # Excess kurtosis is 0 for normal distribution
        return skew_val, kurt_val
    else:
        skew_val = skew(patch.flatten(), bias=False)
        kurt_val = kurtosis(patch.flatten(), bias=False)
        return skew_val, kurt_val

    
def check_majority_greater_than_127(items):
    if not items:  # Handle empty list
        return None
    
    # Count how many items are greater than 127
    count_greater = sum(1 for item in items if item > 80)
    
    # Check if majority of items are greater than 127
    if count_greater >= len(items) / 2:
        return max(items)  # Return the maximum item
    else:
        return min(items)  # Return the maximum item

    
    
def get_all_modes(data):
    data = np.array(data)
    if data.size == 0:
        return []  # Handle empty input
    
    values, counts = np.unique(data, return_counts=True)
    max_count = np.max(counts)
    modes = values[counts == max_count].tolist()  # Convert to list
    
    return check_majority_greater_than_127(modes)






'''


def calculate_SD(image,neighborhood_size):
    neighborhood_size=n
    h,w,_=image.shape
    R = image[:,:,0].astype(float)
    G = image[:,:,1].astype(float)
    B = image[:,:,2].astype(float)
    Gscale1=np.round(0.299 * R + 0.587 * G + 0.114 * B)
    Gscale2=np.round(R/3+G/3+B/3)
   
    half_size = neighborhood_size // 2

    print("******************", half_size)

    # Pad the channels with zeros to handle border pixels
    padded_R = np.pad(R, half_size, mode='constant', constant_values=0)
    padded_G = np.pad(G, half_size, mode='constant', constant_values=0)
    padded_B = np.pad(B, half_size, mode='constant', constant_values=0)
    padded_GS1 = np.pad(Gscale1, half_size, mode='constant', constant_values=0)
    padded_GS = np.pad(Gscale2, half_size, mode='constant', constant_values=0)
    

    # Get nxn neighborhood for each channel
    R_neighborhood = padded_R[y:y + neighborhood_size, x:x + neighborhood_size]
    G_neighborhood = padded_G[y:y + neighborhood_size, x:x + neighborhood_size]
    B_neighborhood = padded_B[y:y + neighborhood_size, x:x + neighborhood_size]
    GS1_neighborhood = padded_GS1[y:y + neighborhood_size, x:x + neighborhood_size]    
    GS_neighborhood = padded_GS[y:y + neighborhood_size, x:x + neighborhood_size]

    print("***********\n",R_neighborhood,"\n\n",G_neighborhood,"\n\n",B_neighborhood,"\n\n",GS_neighborhood)
    
    R_std, G_std, B_std, Gscale1_std, Gscale2_std =np.std(R_neighborhood), np.std(G_neighborhood), np.std(B_neighborhood), np.std(GS1_neighborhood),np.std(GS_neighborhood)
   
    return R_std, G_std, B_std, Gscale1_std, Gscale2_std

'''









########################### MORAN'S I #################################

'''
def calculate_morans_i(patch): ### slow
    """
    Calculate Moran's I for a given 2D patch (neighborhood).

    Parameters:
        patch (numpy.ndarray): A 2D array representing the patch.

    Returns:
        float: Moran's I value.
        float: p-value for Moran's I.
    """
    # Flatten the patch to a 1D array
    values = patch.flatten()

    # Create a spatial weights matrix based on adjacency (queen contiguity)
    n_rows, n_cols = patch.shape
    w = lat2W(n_rows, n_cols, rook=False)  # rook=False for queen contiguity

    # Calculate Moran's I
    moran = Moran(values, w)
    return moran.I, moran.p_norm

'''


def calculate_morans_i(patch): ### WRONG RESULTS
    """Calculate Moran's I for a given patch."""
    patch_mean = np.mean(patch)
    deviations = patch - patch_mean
    numerator = 0
    denominator = np.sum(deviations**2)

    height, width = patch.shape
    for y in range(height):
        for x in range(width):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    numerator += deviations[y, x] * deviations[ny, nx]

    morans_i = numerator / denominator if denominator != 0 else 0
    return morans_i
'''



# Define a function to calculate Moran's I
def calculate_morans_i(matrix):
    n, m = matrix.shape
    W = np.zeros((n*m, n*m))
    
    # Create spatial weight matrix for 5x5 neighborhood
    coords = np.array([(i, j) for i in range(n) for j in range(m)])
    dist_mat = distance_matrix(coords, coords)
    
    for i in range(n*m):
        for j in range(n*m):
            if dist_mat[i, j] > 0 and dist_mat[i, j] <= np.sqrt(2):  # 8-connected neighborhood
                W[i, j] = 1
    
    W_sum = np.sum(W)
    X = matrix.flatten()
    X_mean = np.mean(X)
    
    numerator = 0
    denominator = np.sum((X - X_mean) ** 2)
    
    for i in range(n*m):
        for j in range(n*m):
            numerator += W[i, j] * (X[i] - X_mean) * (X[j] - X_mean)

    
    if W_sum == 0 or denominator == 0:
        print("Warning: Division by zero in Moran's I calculation")
        morans_I = 0  # or some other fallback value
    else:
        morans_I = (n * m / W_sum) * (numerator / denominator)
    

    
    #morans_I = (n*m / W_sum) * (numerator / denominator)
    #morans_I = np.nan_to_num((n * m / W_sum) * (numerator / denominator))
    return morans_I







import numpy as np
from scipy.ndimage import convolve

def calculate_morans_i_improved(matrix):
    n, m = matrix.shape
    N = n * m
    X_mean = np.mean(matrix)
    deviation = matrix - X_mean

    # Define kernel for 8-connected neighbors (excluding the center)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Use convolution to compute the sum of deviations of neighbors for each cell
    neighbor_sum = convolve(deviation, kernel, mode='constant', cval=0)
    
    # Numerator: sum of products of each cell's deviation and the sum of its neighbors' deviations
    numerator = np.sum(deviation * neighbor_sum)
    
    # Compute the sum of weights using the same kernel over a matrix of ones
    weight_matrix = convolve(np.ones_like(matrix), kernel, mode='constant', cval=0)
    W_sum = np.sum(weight_matrix)
    
    # Denominator: total sum of squared deviations
    denominator = np.sum(deviation ** 2)
    
    # Calculate Moran's I using the formula
    morans_I = (N / W_sum) * (numerator / denominator)
    return morans_I
'''


'''
def calculate_morans_i(patch):
    """
    Calculate Moran's I for a given patch.
    """
    patch_mean = np.mean(patch)
    deviations = patch - patch_mean
    numerator = 0
    denominator = np.sum(deviations**2)

    # Use a simple weight matrix for 4-neighbor spatial relationships
    height, width = patch.shape
    weights = np.zeros((height, width, height, width))
    for y in range(height):
        for x in range(width):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    weights[y, x, ny, nx] = 1

    for y in range(height):
        for x in range(width):
            for ny in range(height):
                for nx in range(width):
                    if weights[y, x, ny, nx] > 0:
                        numerator += deviations[y, x] * deviations[ny, nx]


    morans_i = (numerator / denominator) * (height * width / np.sum(weights))
    return morans_i
'''

#######################################




#######################################

def calculate_gearys_c(patch):
    """Calculate Geary's C for a given patch."""
    patch_mean = np.mean(patch)
    deviations = patch - patch_mean
    numerator = 0
    denominator = np.sum(deviations**2)

    height, width = patch.shape
    count = 0
    for y in range(height):
        for x in range(width):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    numerator += (patch[y, x] - patch[ny, nx])**2
                    count += 1

    gearys_c = (numerator / count) / denominator if denominator != 0 else 0
    return gearys_c




##################################################################################################






######################################## VARIOGRAM #########################

'''
def calculate_variogram(patch):
    """
    Calculate the variogram for the entire patch (all pairs of pixels).

    Parameters:
        patch (numpy.ndarray): A 2D array representing the patch.

    Returns:
        distances (numpy.ndarray): Array of unique distances between pixel pairs.
        variogram (numpy.ndarray): Variogram values for each distance.
    """
    # Get the shape of the patch
    rows, cols = patch.shape

    # Initialize lists to store distances and squared differences
    distances = []
    squared_diffs = []

    # Iterate over all pairs of pixels in the patch
    for i1 in range(rows):
        for j1 in range(cols):
            for i2 in range(rows):
                for j2 in range(cols):
                    if i1 == i2 and j1 == j2:
                        continue  # Skip the same pixel pair

                    # Calculate the Euclidean distance between the two pixels
                    distance = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)

                    # Calculate the squared difference in values
                    squared_diff = (patch[i1, j1] - patch[i2, j2])**2

                    # Append to lists
                    distances.append(distance)
                    squared_diffs.append(squared_diff)

    # Convert lists to numpy arrays
    distances = np.array(distances)
    squared_diffs = np.array(squared_diffs)

    # Sort distances and squared differences by distance
    sorted_indices = np.argsort(distances)
    distances = distances[sorted_indices]
    squared_diffs = squared_diffs[sorted_indices]

    # Calculate the variogram (half the mean squared difference for each unique distance)
    unique_distances, indices = np.unique(distances, return_inverse=True)
    variogram = np.zeros_like(unique_distances)

    for k, dist in enumerate(unique_distances):
        mask = (distances == dist)
        variogram[k] = 0.5 * np.mean(squared_diffs[mask])
    #print(variogram)
    return np.sum(variogram)

'''

def calculate_variogram(patch):
    """Calculate the semivariogram for a given patch."""
    pixels = patch.flatten()
    distances = pdist(np.arange(len(pixels)).reshape(-1, 1), metric='euclidean')
    differences = pdist(pixels.reshape(-1, 1), metric='sqeuclidean')
    variogram = np.mean(differences / (2 * distances))
    return variogram



def semivariogram_vectorized(patch, lags=[(0, 1), (1, 0), (1, 1), (-1, 1)]):
    patch = np.array(patch, dtype=np.float64)
    semivariograms = {}
    rows, cols = patch.shape
    
    for lag in lags:
        # Shift the patch using np.roll
        shifted_patch = np.roll(patch, shift=-np.array(lag), axis=(0, 1))
        
        # Create a mask to exclude wrapped-around values due to np.roll
        mask = np.ones_like(patch, dtype=bool)
        if lag[0] > 0:
            mask[:lag[0], :] = False
        elif lag[0] < 0:
            mask[lag[0]:, :] = False
        if lag[1] > 0:
            mask[:, :lag[1]] = False
        elif lag[1] < 0:
            mask[:, lag[1]:] = False

        # Compute differences on valid pixels
        diff = patch - shifted_patch
        valid_diff = diff[mask]
        
        # Compute semivariogram value
        semivariograms[lag] = np.mean(valid_diff**2) / 2.0
        
    return semivariograms




def adjacency_shift(values, flag):
  
    values = np.array(values, dtype=np.float64)
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    rows, cols = values.shape
    N = rows * cols  # Number of elements
    mean_value = np.mean(values)
    neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    numerator = 0
    S0 = 0
    # Iterate over all pixels
    for i in range(rows):
        for j in range(cols):
            for offset in neighbors_offsets:
                ni, nj = i + offset[0], j + offset[1]

                # Check if neighbor is within bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    numerator = numerator+(values[i, j] - values[ni, nj]) ** 2
                    #if flag==1:
                        #print((values[i, j], values[ni, nj]), (values[i, j] - values[ni, nj]))
                    S0 += 1
    vally_well = numerator /2
    #if flag==1:
        #print(values.shape,S0, (N-1), numerator, denominator, "\n", values)
    
    return vally_well

def calculate_standard_deviation(patch):
    """Calculate the standard deviation for a given patch."""
    return np.std(patch)

def mode(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    modes = unique[counts == max_count]
    return np.mean(modes)




def calculate_general_stat(patch):
    mn=np.mean(patch)
    mdn=np.median(patch)
    md=mode(patch)

    return mn,mdn,md
