# Multimodal Sentiment Analysis System

This project creates a sentiment analysis system that analyzes both text and images to determine sentiment. The system takes both a text message and an associated image as input, analyzes sentiment in both modalities, combines the results for a final sentiment score, and visualizes the results.

## Project Structure

```
multimodal_sentiment/
├── main.py               # Main script that ties everything together
├── model_utils.py        # Utility functions for loading and using models
├── visualization.py      # Functions for visualizing results
├── dataset_generator.py  # Script to generate a sample dataset
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Clone the repository or create the file structure as shown above.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Generate a sample dataset:
   ```
   python dataset_generator.py
   ```
   This will create:
    - A `dataset` directory with sample images
    - JSON files with sample data
    - A `happy_scene.jpg` test image in the current directory

4. Run the analysis on your sample image:
   ```
   python main.py --text "This is amazing!" --image "happy_scene.jpg"
   ```

## Using Your Own Data

To use your own data:
1. Place your images in a directory
2. Update the paths in the JSON files or create your own dataset structure
3. Run the main script with your text and image paths

## Available Datasets

Based on our research, here are some free datasets you can use for multimodal sentiment analysis:

1. **MVSA (Multimodal Volatility Sentiment Analysis)** - Contains Twitter posts with text and images
2. **MELD** - Multimodal EmotionLines Dataset with sentiment and emotion annotations
3. **T4SA** - Twitter dataset with text and images labeled for sentiment
4. **CMU-MOSI/MOSEI** - Multimodal Opinion Sentiment and Emotion Intensity datasets

## How to Expand This Project

1. **Add More Modalities**: Include audio analysis for videos or speech
2. **Improve Models**: Fine-tune the pre-trained models on your specific dataset
3. **Add Real-time Analysis**: Implement webcam input for real-time sentiment analysis
4. **Create a Web Interface**: Build a simple web UI using Flask or Streamlit

## Troubleshooting

If you encounter the `ImportError: cannot import name 'predict_sentiment'` error:
- Make sure all functions in `model_utils.py` are correctly imported in `main.py`
- The current code has fixed this issue by removing the unused import

If you have issues with the visualization:
- Make sure matplotlib and all dependencies are installed correctly
- Check that the image paths are valid and accessible