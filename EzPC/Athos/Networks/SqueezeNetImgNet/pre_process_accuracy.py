import os
from PIL import Image
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from datasets import load_dataset

# Fixed-point scale factor
SCALE = 12

# Mean pixel values for BGR
MEAN_PIXEL = np.array([104.006, 116.669, 122.679], dtype=np.float32)


# Function to preprocess images as per SqueezeNet requirements
def preprocess(image):
    # Resize to 227x227
    image = image.resize((227, 227))

    # Convert grayscale images to RGB by duplicating the channels
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert RGB to BGR
    image = np.array(image)
    image = image[:, :, ::-1]  # Flip R and B channels

    # Subtract mean pixel values
    image = image - MEAN_PIXEL

    return image


# Function to process images and save them as .inp files with labels in the filename
def process_save_convert_images(dataset, num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i, image_data in enumerate(dataset):
        if i >= num_images:
            break

        image = image_data["image"]  # The image is a PIL.Image object
        label = image_data["label"]

        # Preprocess the image
        preprocessed_image = preprocess(image)

        # Add a batch dimension (needed for the model input)
        input_batch = np.expand_dims(
            preprocessed_image, axis=0
        )  # Shape: [1, 227, 227, 3]

        # Convert the image to a NumPy array and save as .inp file with the label in the filename
        inp_filename = os.path.join(
            save_dir, f"image_{i}_label_{label}_fixedpt_scale_{SCALE}.inp"
        )
        save_to_inp(input_batch, inp_filename, SCALE)
        print(f"Processed and saved as '{inp_filename}'")


# Function to save numpy array to fixed-point .inp file
def save_to_inp(input_array, output_path, scaling_factor):
    with open(output_path, "w") as ff:
        for xx in np.nditer(input_array, order="C"):
            ff.write(str(int(xx * (1 << scaling_factor))) + " ")
        ff.write("\n")


# Main function to execute the processing
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--num_images",
        required=False,
        type=int,
        default=100,
        help="Number of images to process",
    )
    parser.add_argument(
        "--save_dir",
        required=False,
        type=str,
        default="imagenet_inp_files",
        help="Directory to save .inp files",
    )

    args = parser.parse_args()

    # Load the ImageNet validation dataset
    print("Loading dataset...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    # Process and save as .inp files
    process_save_convert_images(dataset, args.num_images, args.save_dir)


if __name__ == "__main__":
    main()
