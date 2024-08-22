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
