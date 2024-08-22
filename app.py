import os
from flask import Flask, render_template, request
import requests
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize the Flask application
app = Flask(__name__)

# Define the local directories where the pre-trained models and processors are saved
captioning_model_directory = "C:\\Users\\SHAIK FAYAZ\\blip-image-captioning-large"
sentiment_model_directory = "C:\\Users\\SHAIK FAYAZ\\twitter-roberta-base-sentiment"

# Load the image captioning model and processor from the local directory
captioning_model = BlipForConditionalGeneration.from_pretrained(captioning_model_directory)
captioning_processor = BlipProcessor.from_pretrained(captioning_model_directory)

# Load the sentiment analysis model and tokenizer from the local directory
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_directory)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_directory)

# Function to download and open an image from a given URL
def download_image(image_url):
    response = requests.get(image_url)  # Send a GET request to the image URL
    response.raise_for_status()  # Check for HTTP errors and raise an exception if any
    return Image.open(io.BytesIO(response.content))  # Open the image using PIL and return it

# Function to generate a caption for an image using the image captioning model
def generate_caption(image_url):
    image = download_image(image_url)  # Download and preprocess the image
    inputs = captioning_processor(images=image, return_tensors="pt")  # Process the image for the model

    # Generate a caption using the model without computing gradients (for efficiency)
    with torch.no_grad():
        outputs = captioning_model.generate(**inputs)
    
    # Decode the generated caption and return it
    caption = captioning_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to analyze the sentiment of a given text using the sentiment analysis model
def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt")  # Tokenize the input text for the model

    # Perform inference using the sentiment model without computing gradients (for efficiency)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    # Extract the predicted sentiment logits (output scores)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()  # Find the class with the highest score

    # Map the predicted class index to a sentiment label (happy, sad, or neutral)
    label_map = {0: "sad", 1: "neutral", 2: "happy"}
    predicted_label = label_map.get(predicted_class, "unknown")  # Default to "unknown" if not found
    
    return predicted_label

# Define the main route for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    description = None
    emotion = None
    image_url = None
    
    # Check if the request method is POST (form submission)
    if request.method == 'POST':
        image_url = request.form['image_url']  # Get the image URL from the form data
        description = generate_caption(image_url)  # Generate a caption for the image
        emotion = analyze_sentiment(description)  # Analyze the sentiment of the caption
    
    # Render the index.html template with the generated description, emotion, and image URL
    return render_template('index.html', description=description, emotion=emotion, image_url=image_url)

# Run the Flask application in debug mode
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
