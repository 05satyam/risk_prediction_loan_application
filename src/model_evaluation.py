import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils import get_path_to_save_and_load_model
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

'''
This script is for model evalation after training is done on preprocessed data

'''
def load_model(path=None):
    """
    Load a trained LightGBM model from disk.

    Args:
    path (str): The file path to the trained model file.

    Returns:
    model: Loaded model.
    """
    return joblib.load(path)

def evaluate_model(X_test, y_test, model=None):
    """
    Evaluate the performance of the LightGBM model on the test dataset.

    Args:
    X_test (DataFrame): Features from the test dataset.
    y_test (Series): True labels from the test dataset.
    model (LightGBM model, optional): A trained model. If not provided, the model is loaded from .env path.

    Returns:
    tuple: A tuple containing the accuracy and ROC-AUC of the model.
    """
    if model is None:
        model = load_model(get_path_to_save_and_load_model())
   
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1]
   
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prob_predictions)
   
    print(f"Model evaluation complete with Accuracy: {accuracy}, ROC-AUC: {roc_auc}")
    plot_metrics(y_test, predictions, prob_predictions)
   
    return accuracy, roc_auc

def plot_metrics(y_test, predictions, prob_predictions):
    """
    Visualize the evaluation metrics for the model.

    Args:
    y_test (Series): True labels from the test dataset.
    predictions (array): Predicted labels by the model.
    prob_predictions (array): Predicted probabilities by the model.

    Displays:
    plots: Confusion matrix and ROC curve.
    """
    plot_confusion_matrix(y_test, predictions)
    plot_roc_curve(y_test, prob_predictions)
    plt.show()


def plot_confusion_matrix(y_test, predictions):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_curve(y_test, prob_predictions):
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    fpr, tpr, _ = roc_curve(y_test, prob_predictions)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    return fig