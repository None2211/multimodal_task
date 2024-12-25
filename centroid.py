import cv2
import numpy as np
import os

def compute_mask_centroid(img_dir):

    y, x = np.where(mask > 0)

    return round(np.mean(x), 0), round(np.mean(y), 0)

preprocessed_img_dir = r'...'
all_data = []


for file in os.listdir(preprocessed_img_dir):
    img_path = os.path.join(preprocessed_img_dir, file)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path, 0)
        centroid_x, centroid_y = compute_mask_centroid(img)
        all_data.append([file, centroid_x, centroid_y])


with open("centroids_breast.csv", "w") as file:
    file.write("FileName,X_Coordinate,Y_Coordinate\n")
    for data in all_data:
        file.write(f"{data[0]},{data[1]},{data[2]}\n")
