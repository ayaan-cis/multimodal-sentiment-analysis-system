import os
import torch
import numpy as np
import pandas as pd
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Import local modules correctly - fixing the error you encountered
from model_utils import load_text_model, load_image_model
from model_utils import analyze_text_sentiment, analyze_image_sentiment, combine_sentiments

# Setup visualization function
def visualize_results(text, image_path, text_sentiment, text_confidence,
                      image_sentiment, image_confidence, final_sentiment,
                      text_weight, image_weight):
    """
    Visualize the sentiment analysis results for both modalities
    and the combined result.
    """
    # Create figure with 2 rows and 2 columns
    fig = plt.figure(figsize=(12, 8))

    # 1. Original Text
    ax1 = fig.add_subplot(221)
    ax1.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12)
    ax1.set_title("Original Text")
    ax1.axis('off')

    # 2. Original Image
    ax2 = fig.add_subplot(222)
    img = Image.open(image_path)
    ax2.imshow(img)
    ax2.set_title("Original Image")
    ax2.axis('off')

    # 3. Sentiment Results
    ax3 = fig.add_subplot(223)

    # Create bar chart for text and image sentiment
    labels = ['Text', 'Image']
    sentiment_values = [
        text_confidence if text_sentiment == "POSITIVE" else -text_confidence,
        image_confidence if image_sentiment == "POSITIVE" else -image_confidence
    ]
    colors = ['green' if v > 0 else 'red' for v in sentiment_values]

    bars = ax3.bar(labels, sentiment_values, color=colors)

    # Add confidence values as text
    for bar, val, weight in zip(bars, sentiment_values, [text_weight, image_weight]):
        height = bar.get_height() if val > 0 else bar.get_height()
        y_pos = 0.5*height if val > 0 else 1.1*height
        ax3.text(bar.get_x() + bar.get_width()/2.,
                 y_pos,
                 f'conf: {abs(val):.2f}\nweight: {weight:.2f}',
                 ha='center', va='bottom', rotation=0)

    ax3.set_ylim(-1, 1)
    ax3.set_title("Individual Sentiment Scores")
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # 4. Final Result
    ax4 = fig.add_subplot(224)
    final_color = 'green' if final_sentiment == "POSITIVE" else 'red'

    # Create a pie chart showing the weights
    labels = [f'Text ({text_sentiment})', f'Image ({image_sentiment})']
    sizes = [text_weight, image_weight]
    ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f"Final Sentiment: {final_sentiment}", color=final_color)

    plt.tight_layout()
    plt.savefig("sentiment_analysis_result.png")
    plt.show()

    print(f"Results saved to sentiment_analysis_result.png")

# Main function to analyze multimodal sentiment
def analyze_multimodal_sentiment(text, image_path):
    """
    Analyze sentiment using both text and image modalities

    Args:
        text (str): Input text
        image_path (str): Path to image file

    Returns:
        str: Final sentiment classification
    """
    print(f"Analyzing text: '{text}'")
    print(f"Analyzing image: '{image_path}'")

    # 1. Analyze text sentiment
    text_sentiment, text_confidence = analyze_text_sentiment(text)
    print(f"Text sentiment: {text_sentiment} (confidence: {text_confidence:.2f})")

    # 2. Analyze image sentiment
    image_sentiment, image_confidence = analyze_image_sentiment(image_path)
    print(f"Image sentiment: {image_sentiment} (confidence: {image_confidence:.2f})")

    # 3. Combine results
    final_sentiment, text_weight, image_weight = combine_sentiments(
        text_sentiment, text_confidence,
        image_sentiment, image_confidence
    )
    print(f"Final sentiment: {final_sentiment}")
    print(f"Text weight: {text_weight:.2f}, Image weight: {image_weight:.2f}")

    # 4. Visualize results
    visualize_results(
        text, image_path,
        text_sentiment, text_confidence,
        image_sentiment, image_confidence,
        final_sentiment, text_weight, image_weight
    )

    return final_sentiment

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    sentiment = analyze_multimodal_sentiment(args.text, args.image)