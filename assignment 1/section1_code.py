import os
import numpy as np
import pandas as pd
from PIL import Image

input_folder = "xclaim"
output_folder = "images"

os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith(".pgm"):

        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)

        img_array = np.array(img)

        inverted_array = np.where(img_array == 0, 1, 0)

        df = pd.DataFrame(inverted_array)

        csv_filename = "40443486_" + os.path.splitext(filename)[0] + ".csv"
        df.to_csv(os.path.join(output_folder, csv_filename), index=False, header=False)

print("Conversion completed. CSV files are saved in", output_folder)
