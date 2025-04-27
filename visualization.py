import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

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

    return "sentiment_analysis_result.png"

def create_confusion_matrix(actual_labels, predicted_labels, classes=None):
    """
    Create a confusion matrix visualization

    Args:
        actual_labels: Ground truth labels
        predicted_labels: Predicted labels
        classes: List of class names
    """
    if classes is None:
        classes = sorted(list(set(actual_labels + predicted_labels)))

    # Create confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    # Fill confusion matrix
    for actual, pred in zip(actual_labels, predicted_labels):
        actual_idx = classes.index(actual)
        pred_idx = classes.index(pred)
        cm[actual_idx, pred_idx] += 1

    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    return "confusion_matrix.png"

def plot_training_history(history):
    """
    Plot training history metrics

    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []))
    plt.plot(history.get('val_accuracy', []))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []))
    plt.plot(history.get('val_loss', []))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig("training_history.png")

    return "training_history.png"