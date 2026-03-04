# src/model_evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import VotingClassifier, StackingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from src.project_utils import load_config, plot_confusion_matrix, plot_roc_curve, plot_feature_importance  # Changed

class ModelEvaluator:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        
    def create_ensemble_model(self, models):
        """Create ensemble model using voting classifier"""
        print("Creating Ensemble Model...")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft',
            weights=[1, 1.5, 0.8]  # Give XGBoost more weight
        )
        
        return voting_clf
    
    def create_stacking_model(self, models, X_train, y_train):
        """Create stacking ensemble model"""
        print("Creating Stacking Model...")
        
        # Use XGBoost as final estimator
        from xgboost import XGBClassifier
        
        stacking_clf = StackingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            final_estimator=XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            ),
            cv=5,
            passthrough=True
        )
        
        stacking_clf.fit(X_train, y_train)
        return stacking_clf
    
    def evaluate_ensemble(self, ensemble_model, X_test, y_test, ensemble_name):
        """Evaluate ensemble model"""
        print(f"\nEvaluating {ensemble_name}...")
        
        # Make predictions
        y_pred = ensemble_model.predict(X_test)
        y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, report, cm, y_pred, y_pred_proba
    
    def plot_model_comparison(self, results_dict):
        """Create comparison plot of all models"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, metric in enumerate(metrics):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            values = [results_dict[model].get(metric, 0) for model in models]
            
            if metric == 'roc_auc':
                fig.add_trace(
                    go.Scatter(x=models, y=values, mode='lines+markers',
                             line=dict(color=colors[idx], width=3),
                             marker=dict(size=10)),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Bar(x=models, y=values, marker_color=colors[idx]),
                    row=row, col=col
                )
            
            fig.update_yaxes(range=[0, 1], row=row, col=col)
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba_dict, model_names):
        """Plot precision-recall curve for all models"""
        fig = go.Figure()
        
        colors = ['blue', 'green', 'red', 'purple']
        
        for idx, (model_name, y_proba) in enumerate(y_pred_proba_dict.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AUC = {pr_auc:.3f})',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_calibration_curve(self, y_true, y_pred_proba_dict, model_names):
        """Plot calibration curve for all models"""
        from sklearn.calibration import calibration_curve
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        colors = ['blue', 'green', 'red', 'purple']
        
        for idx, (model_name, y_proba) in enumerate(y_pred_proba_dict.items()):
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            
            fig.add_trace(go.Scatter(
                x=prob_pred, y=prob_true,
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Calibration Curve',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=700,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def generate_detailed_report(self, results_dict, feature_names=None):
        """Generate detailed evaluation report"""
        print("\n" + "=" * 50)
        print("DETAILED EVALUATION REPORT")
        print("=" * 50)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results_dict).T
        
        print("\nPerformance Summary:")
        print(summary_df.round(4))
        
        # Identify best model for each metric
        print("\nBest Models for Each Metric:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in summary_df.columns:
                best_model = summary_df[metric].idxmax()
                best_score = summary_df[metric].max()
                print(f"  {metric}: {best_model} ({best_score:.4f})")
        
        # Feature importance for tree-based models
        if feature_names is not None:
            print("\nFeature Importance Analysis:")
            # This would be implemented based on specific models
        
        return summary_df
    
    def run_complete_evaluation(self, models_dict, X_test, y_test, X_train=None, y_train=None):
        """Run complete evaluation pipeline"""
        print("=" * 50)
        print("MODEL EVALUATION PHASE")
        print("=" * 50)
        
        # Individual model evaluation
        all_results = {}
        all_predictions = {}
        all_probabilities = {}
        
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            all_results[model_name] = metrics
            all_predictions[model_name] = y_pred
            all_probabilities[model_name] = y_pred_proba
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Create ensemble models
        if X_train is not None and y_train is not None:
            voting_clf = self.create_ensemble_model(models_dict)
            voting_clf.fit(X_train, y_train)
            
            voting_metrics, _, _, voting_pred, voting_proba = self.evaluate_ensemble(
                voting_clf, X_test, y_test, 'Voting Ensemble'
            )
            
            all_results['Voting Ensemble'] = voting_metrics
            all_probabilities['Voting Ensemble'] = voting_proba
            
            # Stacking ensemble
            stacking_clf = self.create_stacking_model(models_dict, X_train, y_train)
            stacking_metrics, _, _, stacking_pred, stacking_proba = self.evaluate_ensemble(
                stacking_clf, X_test, y_test, 'Stacking Ensemble'
            )
            
            all_results['Stacking Ensemble'] = stacking_metrics
            all_probabilities['Stacking Ensemble'] = stacking_proba
        
        # Generate plots
        comparison_plot = self.plot_model_comparison(all_results)
        pr_curve = self.plot_precision_recall_curve(y_test, all_probabilities, list(all_probabilities.keys()))
        
        # Generate detailed report
        summary_df = self.generate_detailed_report(all_results)
        
        return {
            'results': all_results,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'plots': {
                'comparison': comparison_plot,
                'pr_curve': pr_curve
            },
            'summary': summary_df
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Load data
    from model_training import ModelTrainer
    trainer = ModelTrainer()
    data_path = trainer.config['paths']['output_dir']
    X_train, X_test, y_train, y_test = trainer.load_data(data_path)
    
    # Load models
    from project_utils import load_model
    models = {
        'Random Forest': load_model('random_forest'),
        'XGBoost': load_model('xgboost'),
        'Logistic Regression': load_model('logistic_regression')
    }
    
    # Run evaluation
    evaluation_results = evaluator.run_complete_evaluation(models, X_test, y_test, X_train, y_train)