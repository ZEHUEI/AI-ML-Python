import os
import numpy as np
import pandas as pd
from scipy.ndimage import label, generate_binary_structure

images_folder = "./images"

feature_data = []

feature_names = [
    "image_label", "index", "nr_pix", "rows_with_1", "cols_with_1", "rows_with_3p", "cols_with_3p",
    "aspect_ratio", "neigh_1", "no_neigh_above", "no_neigh_below", "no_neigh_left",
    "no_neigh_right", "no_neigh_horiz", "no_neigh_vert", "connected_areas", "eyes", "custom"
]
# Function to calculate the number of black pixels
def calculate_nr_pix(matrix):
    return np.sum(matrix)


# Function to calculate the number of rows with exactly 1 black pixel
def calculate_rows_with_1(matrix):
    return np.sum(np.sum(matrix, axis=1) == 1)


# Function to calculate the number of columns with exactly 1 black pixel
def calculate_cols_with_1(matrix):
    return np.sum(np.sum(matrix, axis=0) == 1)


# Function to calculate the number of rows with 3 or more black pixels
def calculate_rows_with_3p(matrix):
    return np.sum(np.sum(matrix, axis=1) >= 3)


# Function to calculate the number of columns with 3 or more black pixels
def calculate_cols_with_3p(matrix):
    return np.sum(np.sum(matrix, axis=0) >= 3)


# Function to calculate the aspect ratio
def calculate_aspect_ratio(matrix):
    rows, cols = np.where(matrix == 1)
    if len(rows) == 0:
        return 0
    height = np.max(rows) - np.min(rows) + 1
    width = np.max(cols) - np.min(cols) + 1
    return width / height if height != 0 else 0


# Function to calculate the number of black pixels with only 1 black pixel neighbour
def calculate_neigh_1(matrix):
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)
    count = 0
    for i in range(1, padded_matrix.shape[0] - 1):
        for j in range(1, padded_matrix.shape[1] - 1):
            if padded_matrix[i, j] == 1:
                neighbours = padded_matrix[i - 1:i + 2, j - 1:j + 2].sum() - 1
                if neighbours == 1:
                    count += 1
    return count


# Function to calculate the number of black pixels with no black pixel neighbours in specific positions
def calculate_no_neigh(matrix, positions):
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)
    count = 0
    for i in range(1, padded_matrix.shape[0] - 1):
        for j in range(1, padded_matrix.shape[1] - 1):
            if padded_matrix[i, j] == 1:
                has_neighbour = False
                for pos in positions:
                    if padded_matrix[i + pos[0], j + pos[1]] == 1:
                        has_neighbour = True
                        break
                if not has_neighbour:
                    count += 1
    return count


# Function to calculate the number of connected areas
def calculate_connected_areas(matrix):
    structure = generate_binary_structure(2, 2)
    labeled_matrix, num_features = label(matrix, structure)
    return num_features


# Function to calculate the number of eyes (whitespace regions completely surrounded by black pixels)
def calculate_eyes(matrix):
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=1)
    inverted_matrix = 1 - padded_matrix
    structure = generate_binary_structure(2, 2)
    labeled_matrix, num_features = label(inverted_matrix, structure)
    return num_features - 1  # Subtract 1 for the background


# Function to calculate a custom feature
def calculate_custom(matrix):
    # Count the number of corners in the image
    corners = 0
    for i in range(matrix.shape[0] - 1):
        for j in range(matrix.shape[1] - 1):
            if matrix[i, j] == 1 and matrix[i + 1, j] == 0 and matrix[i, j + 1] == 0:
                corners += 1
    return corners


for filename in os.listdir(images_folder):
    if filename.endswith(".csv"):

        image_label = filename.split("_")[1]
        index = filename.split("_")[2].split(".")[0]

        matrix = np.loadtxt(os.path.join(images_folder, filename), delimiter=",")

        nr_pix = calculate_nr_pix(matrix)
        rows_with_1 = calculate_rows_with_1(matrix)
        cols_with_1 = calculate_cols_with_1(matrix)
        rows_with_3p = calculate_rows_with_3p(matrix)
        cols_with_3p = calculate_cols_with_3p(matrix)
        aspect_ratio = calculate_aspect_ratio(matrix)
        neigh_1 = calculate_neigh_1(matrix)
        no_neigh_above = calculate_no_neigh(matrix, [(-1, -1), (-1, 0), (-1, 1)])
        no_neigh_below = calculate_no_neigh(matrix, [(1, -1), (1, 0), (1, 1)])
        no_neigh_left = calculate_no_neigh(matrix, [(-1, -1), (0, -1), (1, -1)])
        no_neigh_right = calculate_no_neigh(matrix, [(-1, 1), (0, 1), (1, 1)])
        no_neigh_horiz = calculate_no_neigh(matrix, [(0, -1), (0, 1)])
        no_neigh_vert = calculate_no_neigh(matrix, [(-1, 0), (1, 0)])
        connected_areas = calculate_connected_areas(matrix)
        eyes = calculate_eyes(matrix)
        custom = calculate_custom(matrix)

        feature_data.append([
            image_label, index, nr_pix, rows_with_1, cols_with_1, rows_with_3p, cols_with_3p,
            aspect_ratio, neigh_1, no_neigh_above, no_neigh_below, no_neigh_left,
            no_neigh_right, no_neigh_horiz, no_neigh_vert, connected_areas, eyes, custom
        ])

features_df = pd.DataFrame(feature_data, columns=feature_names)

features_df = features_df.sort_values(by=["image_label", "index"])

features_df.to_csv("40443486_features.csv", index=False)

print("Feature extraction complete. Features saved to 40443486_features.csv")