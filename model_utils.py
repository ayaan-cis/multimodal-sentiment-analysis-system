import torch
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
from torchvision import transforms, models

# Text sentiment analysis
def load_text_model():
    """
    Load pre-trained sentiment analysis model

    Returns:
        tokenizer, model: Pretrained tokenizer and model
    """
    # Load pre-trained sentiment analysis model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

def analyze_text_sentiment(text, tokenizer=None, model=None):
    """
    Analyze sentiment of input text

    Args:
        text (str): Input text
        tokenizer: Optional tokenizer
        model: Optional model

    Returns:
        tuple: (sentiment, confidence) pair
    """
    if tokenizer is None or model is None:
        # Use pipeline for simplicity
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]
        sentiment = result["label"]
        confidence = result["score"]
    else:
        # Manual prediction
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive_prob = probabilities[0][1].item()
        sentiment = "POSITIVE" if positive_prob > 0.5 else "NEGATIVE"
        confidence = positive_prob if sentiment == "POSITIVE" else 1 - positive_prob

    return sentiment, confidence

# Image sentiment analysis
def load_image_model():
    """
    Load pre-trained image model

    Returns:
        model: Pretrained model for image sentiment
    """
    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    # Modify the final layer for sentiment classification
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # Binary classification

    # In a real implementation, you would load fine-tuned weights
    # Here we use the pretrained model as a placeholder

    return model

def analyze_image_sentiment(image_path, model=None):
    """
    Analyze sentiment of input image

    Args:
        image_path (str): Path to image file
        model: Optional model

    Returns:
        tuple: (sentiment, confidence) pair
    """
    # For demonstration purposes, we'll use a simple rule-based approach
    # In a real scenario, you would use your fine-tuned model

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        if model is None:
            # Simple color-based heuristic (for demonstration only)
            # In practice, you would use a properly trained model
            img_array = np.array(image)
            avg_pixel = np.mean(img_array, axis=(0, 1))

            # Crude heuristic: more green tends to be positive, more red tends to be negative
            if avg_pixel[1] > avg_pixel[0]:  # Green > Red
                sentiment = "POSITIVE"
                confidence = min(0.5 + (avg_pixel[1] - avg_pixel[0]) / 255, 0.9)
            else:
                sentiment = "NEGATIVE"
                confidence = min(0.5 + (avg_pixel[0] - avg_pixel[1]) / 255, 0.9)
        else:
            # Use the model for prediction
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                positive_prob = probabilities[0][1].item()
                sentiment = "POSITIVE" if positive_prob > 0.5 else "NEGATIVE"
                confidence = positive_prob if sentiment == "POSITIVE" else 1 - positive_prob

    except Exception as e:
        print(f"Error analyzing image: {e}")
        sentiment = "NEUTRAL"
        confidence = 0.5

    return sentiment, confidence

# Combining modalities
def combine_sentiments(text_sentiment, text_confidence, image_sentiment, image_confidence):
    """
    Combine text and image sentiments

    Args:
        text_sentiment (str): Sentiment from text analysis
        text_confidence (float): Confidence of text sentiment
        image_sentiment (str): Sentiment from image analysis
        image_confidence (float): Confidence of image sentiment

    Returns:
        tuple: (final_sentiment, text_weight, image_weight)
    """
    # Convert sentiments to numeric values
    text_score = 1 if text_sentiment == "POSITIVE" else -1
    image_score = 1 if image_sentiment == "POSITIVE" else -1

    # Weight by confidence
    text_weight = text_confidence
    image_weight = image_confidence

    # Normalize weights
    total_weight = text_weight + image_weight
    text_weight /= total_weight
    image_weight /= total_weight

    # Calculate weighted score
    weighted_score = (text_score * text_weight) + (image_score * image_weight)

    # Determine final sentiment
    final_sentiment = "POSITIVE" if weighted_score > 0 else "NEGATIVE"

    return final_sentiment, text_weight, image_weight