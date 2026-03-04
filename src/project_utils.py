# src/project_utils.py
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, model_name, model_dir="models"):
    """Save trained model to disk"""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(model_name, model_dir="models"):
    """Load trained model from disk"""
    model_path = Path(model_dir) / f"{model_name}.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_results(results, filename="results.json"):
    """Save evaluation results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Non-Fraud', 'Fraud'],
        y=['Non-Fraud', 'Fraud'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    return fig

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h'
        ))
        fig.update_layout(
            title=f'Top {top_n} Feature Importances - {model_name}',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600
        )
        return fig
    return None