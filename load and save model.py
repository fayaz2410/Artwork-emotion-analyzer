#downloading sentimental analysis model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the model name
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the local directory where you want to save the model
local_directory = "C:\\Users\\SHAIK FAYAZ\\twitter-roberta-base-sentiment"

# Save the model and tokenizer locally
model.save_pretrained(local_directory)
tokenizer.save_pretrained(local_directory)

print(f"Model and tokenizer saved to {local_directory}")


#loadind sentimental analysis model from local_directory
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the local directory where the model is saved
local_directory = "C:\\Users\\SHAIK FAYAZ\\twitter-roberta-base-sentiment"

# Load the model and tokenizer from the local directory
model = AutoModelForSequenceClassification.from_pretrained(local_directory)
tokenizer = AutoTokenizer.from_pretrained(local_directory)

print("Model and tokenizer loaded from local directory")

#downloding image captioning model from hugging face

from transformers import BlipProcessor, BlipForConditionalGeneration

# Define the model name
model_name = "Salesforce/blip-image-captioning-large"

# Load the model and processor
model = BlipForConditionalGeneration.from_pretrained(model_name)
processor = BlipProcessor.from_pretrained(model_name)

# Define the local directory where you want to save the model and processor
local_directory = "C:\\Users\\SHAIK FAYAZ\\blip-image-captioning-large"

# Save the model and processor locally
model.save_pretrained(local_directory)
processor.save_pretrained(local_directory)

print(f"Model and processor saved to {local_directory}")

#loading image captioning model from local_directory
from transformers import BlipProcessor, BlipForConditionalGeneration

# Define the local directory where the model is saved
local_directory = "C:\\Users\\SHAIK FAYAZ\\blip-image-captioning-large"

# Load the model and processor from the local directory
model = BlipForConditionalGeneration.from_pretrained(local_directory)
processor = BlipProcessor.from_pretrained(local_directory)

print("Model and processor loaded from local directory")
