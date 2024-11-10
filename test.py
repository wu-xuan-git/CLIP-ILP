import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Ensure that the model uses GPU for prediction, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.cuda.is_available())
print(device)

# Load the trained model and processor from the specified directory
model_save_dir = "./run"  # Replace with the path where your model is saved
model = CLIPModel.from_pretrained(model_save_dir).to(device)  # Load the model and move it to the device (GPU or CPU)
processor = CLIPProcessor.from_pretrained(model_save_dir)  # Load the processor for image-text encoding

# Define the prediction function
def predict(image_path):
    # Load the image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image and text inputs for the model
    inputs = processor(images=image, text=[
        "have person in four-wheeled truck container",
        "no person in four-wheeled truck container",
        "have person in three-wheeled truck container",
        "no person in three-wheeled truck container"
    ], return_tensors="pt", padding=True)  # Convert inputs into tensors and pad the text as needed

    with torch.no_grad():  # Disable gradient calculation for inference
        # Forward pass through the model
        outputs = model(input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        pixel_values=inputs['pixel_values'].to(device))

    # Get the logits (raw outputs) for the image-text similarity
    logits = outputs.logits_per_image
    probabilities = logits.softmax(dim=1)  # Convert logits to probabilities using softmax

    # Get the predicted class based on the highest probability
    predicted_class = torch.argmax(probabilities, dim=1)

    # Return the predicted class label and the probabilities for each class
    return (
        [
            "have person in four-wheeled truck container",
            "no person in four-wheeled truck container",
            "have person in three-wheeled truck container",
            "no person in three-wheeled truck container"
        ][predicted_class.item()],  # Convert the index to the corresponding class label
        probabilities[0].cpu().numpy()  # Return the probabilities as a numpy array (move to CPU)
    )

# Main function for processing a folder of images
if __name__ == "__main__":
    # Specify the folder containing the images for prediction
    folder_path = "./ex_tricycle_legal_test"  # Path to the folder containing images for prediction

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")  # Print error message if folder is missing
    else:
        # Initialize the category counters to track predictions for each class
        category_counts = {
            "have person in four-wheeled truck container": 0,
            "no person in four-wheeled truck container": 0,
            "have person in three-wheeled truck container": 0,
            "no person in three-wheeled truck container": 0
        }

        # Iterate over each file in the folder
        for filename in os.listdir(folder_path):
            # Check if the file has an image extension (case-insensitive)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)  # Get the full path of the image

                # Make the prediction for the image
                result, probabilities = predict(image_path)
                # Print the prediction result and the probabilities for each class
                print(
                    f"File: {filename}, Prediction: {result}, Probabilities: "
                    f"Have 4-wheeled truck: {probabilities[0]:.4f}, "
                    f"No person in 4-wheeled truck: {probabilities[1]:.4f}, "
                    f"Have 3-wheeled truck: {probabilities[2]:.4f}, "
                    f"No person in 3-wheeled truck: {probabilities[3]:.4f}"
                )

                # Update the category count based on the predicted class
                category_counts[result] += 1

        # Print the count of images predicted for each category
        print("\nCategory Statistics:")
        for category, count in category_counts.items():
            print(f"{category}: {count}")
