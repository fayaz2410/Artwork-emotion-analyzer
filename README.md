# AI-Powered Artwork Description and Sentiment Analysis

This Flask application provides AI-powered descriptions for images and analyzes the emotional tone of these descriptions. It utilizes pre-trained models for image captioning and sentiment analysis, making it easy to get automated insights from visual content.

## Features

- **Image Captioning**: Generate a descriptive caption for an image from a given URL.
- **Sentiment Analysis**: Analyze the sentiment of the generated caption (happy, sad, or neutral).

## Requirements

To run this application, you need to have Python 3.x installed along with the following packages:

- Flask
- requests
- Pillow (PIL)
- transformers
- torch

You can install these dependencies using pip:

```bash
pip install Flask requests Pillow transformers torch
1.Setup
Clone the Repository:
```bash
git clone https://github.com/fayaz2410/Artwork-emotion-analyzer.git
cd Artwork-emotion-analyzer
2.Download Pre-trained Models:

The application uses pre-trained models for image captioning and sentiment analysis, which are loaded locally. Make sure you have the pre-trained models downloaded and saved in the following directories:

Image Captioning Model: blip-image-captioning-large
Sentiment Analysis Model: twitter-roberta-base-sentiment
You can download these models from the Hugging Face Model Hub or another source and place them in the appropriate directories.

3.Update Model Directories:

Modify the captioning_model_directory and sentiment_model_directory paths in app.py to point to the local directories where the models are saved.
4.Run the Application:

Start the Flask server by running:
bash
python app.py
The application will be accessible at http://localhost:5000 by default.

5.Usage
Open your web browser and go to http://localhost:5000.
Enter the URL of the image you want to analyze.
Submit the form to get a caption for the image and the sentiment of the caption.
6.Files
app.py: The main Flask application script.
templates/index.html: The HTML template for the web interface.
7.Contributing
Feel free to fork the repository, make changes, and submit pull requests. Contributions and suggestions are welcome!
Acknowledgements
The transformers library by Hugging Face for pre-trained models.