import os
import json
import numpy as np
from PIL import Image
import urllib.request
import shutil
import random

# Define basic dataset structure
def create_sample_dataset(base_dir="dataset"):
    """Create a small sample dataset for multimodal sentiment analysis"""

    # Create directory structure
    os.makedirs(f"{base_dir}/images", exist_ok=True)

    # Sample positive images URLs (these are placeholders - you can replace with real URLs)
    positive_images = [
        "https://img.freepik.com/free-photo/happy-woman-enjoying-nature_23-2149543635.jpg",
        "https://img.freepik.com/free-photo/friends-having-fun-concert_23-2149642986.jpg"
    ]

    # Sample neutral images URLs
    neutral_images = [
        "https://img.freepik.com/free-photo/businesswoman-using-tablet-office_23-2148174081.jpg",
        "https://img.freepik.com/free-photo/woman-sitting-beach-enjoying-view_23-2149063424.jpg"
    ]

    # Sample negative images URLs
    negative_images = [
        "https://img.freepik.com/free-photo/sad-woman-sitting-floor-home_23-2149346846.jpg",
        "https://img.freepik.com/free-photo/medium-shot-frustrated-man_23-2149172641.jpg"
    ]

    # Create sample texts
    positive_texts = [
        "This is amazing! I'm having a great time.",
        "Feeling wonderful today, everything is perfect!",
        "So happy with how this turned out!"
    ]

    neutral_texts = [
        "This is what I expected, nothing special.",
        "Regular day at work, same as usual.",
        "The weather is mild today."
    ]

    negative_texts = [
        "This is terrible, I'm very disappointed.",
        "Feeling down today, nothing is going right.",
        "So upset with how this turned out."
    ]

    # Create dataset entries
    dataset = []
    image_id = 1

    # Helper function to download an image
    def download_image(url, filename):
        try:
            # Use a placeholder image if download fails
            urllib.request.urlretrieve(url, filename)
            return True
        except:
            # If download fails, create a colored square as placeholder
            sentiment = "positive" if "positive" in filename else "negative" if "negative" in filename else "neutral"
            color = (0, 255, 0) if sentiment == "positive" else (255, 0, 0) if sentiment == "negative" else (128, 128, 128)
            img = Image.new('RGB', (128, 128), color=color)
            img.save(filename)
            return False

    # Function to add entries
    def add_entries(texts, image_urls, sentiment, sentiment_score):
        nonlocal image_id
        for text in texts:
            img_url = random.choice(image_urls)
            img_filename = f"{base_dir}/images/{sentiment}_{image_id}.jpg"

            # Download or create image
            downloaded = download_image(img_url, img_filename)

            # Add entry to dataset
            dataset.append({
                "id": f"sample_{len(dataset) + 1}",
                "text": text,
                "image_path": img_filename,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score
            })

            image_id += 1

    # Add positive, neutral and negative entries
    add_entries(positive_texts, positive_images, "positive", 1.0)
    add_entries(neutral_texts, neutral_images, "neutral", 0.0)
    add_entries(negative_texts, negative_images, "negative", -1.0)

    # Shuffle dataset
    random.shuffle(dataset)

    # Split into train and test sets (80/20)
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # Create metadata
    metadata = {
        "description": "Sample dataset for multimodal sentiment analysis",
        "size": len(dataset),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "classes": ["positive", "neutral", "negative"]
    }

    # Save to JSON files
    with open(f"{base_dir}/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(f"{base_dir}/test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    with open(f"{base_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create a single placeholder image for quick testing
    happy_image = Image.new('RGB', (128, 128), color=(0, 255, 0))
    happy_image.save("happy_scene.jpg")

    print(f"Dataset created successfully with {len(dataset)} entries.")
    print(f"- Training set: {len(train_data)} entries")
    print(f"- Test set: {len(test_data)} entries")
    print("- Sample image for testing saved as 'happy_scene.jpg'")

if __name__ == "__main__":
    create_sample_dataset()