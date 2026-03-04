import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')
from src.project_utils import load_config, save_model, save_results  # Changed from utils
import time

class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.models = {}
        self.results = {}
        
    def load_data(self, data_path):
        """Load processed data"""
        X_train = pd.read_csv(f"{data_path}/X_train.csv")
        X_test = pd.read_csv(f"{data_path}/X_test.csv")
        y_train = pd.read_csv(f"{data_path}/y_train.csv").squeeze()
        y_test = pd.read_csv(f"{data_path}/y_test.csv").squeeze()
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        rf_params = self.config['models']['random_forest']
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            random_state=rf_params['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )

        # Optional hyperparameter tuning
        if self.config.get('training', {}).get('hyperparameter_tuning', False):
            print("Random Forest: hyperparameter tuning...")
            rf_dist = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            tuner = RandomizedSearchCV(
                rf_model,
                rf_dist,
                n_iter=self.config['training'].get('tuning_iterations', 25),
                scoring='roc_auc',
                cv=3,
                random_state=rf_params['random_state'],
                n_jobs=-1
            )
            tuner.fit(X_train, y_train)
            rf_model = tuner.best_estimator_
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['random_forest'] = rf_model
        print(f"Random Forest trained in {training_time:.2f} seconds")
        
        return rf_model, training_time
    
    def train_xgboost(self, X_train, y_train, scale_pos_weight=1.0):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        xgb_params = self.config['models']['xgboost']
        xgb_model = XGBClassifier(
            n_estimators=xgb_params['n_estimators'],
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            random_state=xgb_params['random_state'],
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method='hist',
            scale_pos_weight=max(1.0, scale_pos_weight)
        )

        # Optional hyperparameter tuning
        if self.config.get('training', {}).get('hyperparameter_tuning', False):
            print("XGBoost: hyperparameter tuning...")
            xgb_dist = {
                'n_estimators': [200, 300, 500, 700],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_lambda': [0.5, 1.0, 1.5]
            }
            tuner = RandomizedSearchCV(
                xgb_model,
                xgb_dist,
                n_iter=self.config['training'].get('tuning_iterations', 25),
                scoring='roc_auc',
                cv=3,
                random_state=xgb_params['random_state'],
                n_jobs=-1
            )
            tuner.fit(X_train, y_train)
            xgb_model = tuner.best_estimator_

        # Early stopping using an internal validation split
        early_rounds = self.config.get('training', {}).get('early_stopping_rounds', 50)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train,
            random_state=self.config['preprocessing']['random_state']
        )
        
        start_time = time.time()
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_rounds,
            verbose=False
        )
        training_time = time.time() - start_time
        
        self.models['xgboost'] = xgb_model
        print(f"XGBoost trained in {training_time:.2f} seconds")
        
        return xgb_model, training_time
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        lr_params = self.config['models']['logistic_regression']
        lr_model = LogisticRegression(
            C=lr_params['C'],
            max_iter=lr_params['max_iter'],
            random_state=lr_params['random_state'],
            class_weight='balanced',
            solver='liblinear'
        )

        # Optional probability calibration (isotonic) for better thresholding
        if self.config.get('training', {}).get('calibrate_probabilities', False):
            print("Logistic Regression: calibrating probabilities...")
            lr_model = CalibratedClassifierCV(lr_model, method='isotonic', cv=3)
        
        start_time = time.time()
        lr_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['logistic_regression'] = lr_model
        print(f"Logistic Regression trained in {training_time:.2f} seconds")
        
        return lr_model, training_time
    
    def train_lightgbm(self, X_train, y_train, scale_pos_weight=1.0):
        """Train LightGBM model"""
        print("Training LightGBM...")
        
        lgb_model = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            reg_lambda=1.0,
            reg_alpha=0.5,
            class_weight='balanced',
            verbose=-1
        )
        
        # Optional hyperparameter tuning
        if self.config.get('training', {}).get('hyperparameter_tuning', False):
            print("LightGBM: hyperparameter tuning...")
            lgb_dist = {
                'n_estimators': [150, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 63],
                'reg_lambda': [0.5, 1.0, 1.5]
            }
            tuner = RandomizedSearchCV(
                lgb_model,
                lgb_dist,
                n_iter=self.config['training'].get('tuning_iterations', 15),
                scoring='roc_auc',
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            tuner.fit(X_train, y_train)
            lgb_model = tuner.best_estimator_
        
        start_time = time.time()
        lgb_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['lightgbm'] = lgb_model
        print(f"LightGBM trained in {training_time:.2f} seconds")
        
        return lgb_model, training_time
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Threshold optimization (default metric f1)
        metric_name = self.config.get('training', {}).get('threshold_optimization_metric', 'f1')
        best_thr, thr_metrics = self.optimize_threshold(y_test, y_pred_proba, metric_name)
        y_pred = (y_pred_proba >= best_thr).astype(int)
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': pr_auc,
            'best_threshold': best_thr
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config['evaluation']['cv_folds'], shuffle=True, 
                           random_state=self.config['preprocessing']['random_state'])
        
        cv_scores = {}
        for metric_name in self.config['evaluation']['metrics']:
            if metric_name == 'roc_auc':
                scorer = 'roc_auc'
            elif metric_name == 'accuracy':
                scorer = 'accuracy'
            elif metric_name == 'precision':
                scorer = 'precision'
            elif metric_name == 'recall':
                scorer = 'recall'
            elif metric_name == 'f1':
                scorer = 'f1'
            else:
                continue
                
            scores = cross_val_score(model, X_test, y_test, cv=cv, scoring=scorer)
            cv_scores[f'cv_{metric_name}_mean'] = scores.mean()
            cv_scores[f'cv_{metric_name}_std'] = scores.std()
        
        # Combine all results
        results = {**metrics, **cv_scores}
        
        # Print results
        print(f"{model_name} Results:")
        for key, value in results.items():
            if 'mean' in key or 'auc' in key or 'accuracy' in key or key == 'best_threshold':
                print(f"  {key}: {value:.4f}")
        
        return results, y_pred, y_pred_proba

    def optimize_threshold(self, y_true, y_proba, metric='f1'):
        """Find decision threshold that maximizes a given metric."""
        thresholds = np.linspace(0.1, 0.9, 81)
        best_thr = 0.5
        best_score = -1
        best_metrics = {}
        for thr in thresholds:
            y_hat = (y_proba >= thr).astype(int)
            acc = accuracy_score(y_true, y_hat)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1s = f1_score(y_true, y_hat, zero_division=0)
            score = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1s}[metric]
            if score > best_score:
                best_score = score
                best_thr = thr
                best_metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1s}
        return best_thr, best_metrics
    
    def run_cross_validation(self, X, y):
        """Run cross-validation for all models"""
        print("\nRunning Cross-Validation...")
        
        cv = StratifiedKFold(n_splits=self.config['evaluation']['cv_folds'], shuffle=True,
                           random_state=self.config['preprocessing']['random_state'])
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nCV for {model_name}:")
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            cv_results[model_name] = {
                'mean_roc_auc': scores.mean(),
                'std_roc_auc': scores.std(),
                'all_scores': scores.tolist()
            }
            print(f"  ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def train_all_models(self, data_path):
        """Train all models"""
        print("=" * 50)
        print("MODEL TRAINING PHASE")
        print("=" * 50)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(data_path)
        
        # Compute class weights for cost-sensitive learning
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / max(1, pos_count)
        print(f"Class weight ratio (neg/pos): {scale_pos_weight:.2f}")
        
        # Train models
        rf_model, rf_time = self.train_random_forest(X_train, y_train)
        xgb_model, xgb_time = self.train_xgboost(X_train, y_train, scale_pos_weight)
        lr_model, lr_time = self.train_logistic_regression(X_train, y_train)
        lgb_model = None
        lgb_time = 0
        
        if HAS_LIGHTGBM:
            lgb_model, lgb_time = self.train_lightgbm(X_train, y_train, scale_pos_weight)
        else:
            print("Warning: LightGBM not installed, skipping...")
        
        # Evaluate models
        self.results['random_forest'], rf_pred, rf_proba = self.evaluate_model(
            rf_model, X_test, y_test, 'Random Forest'
        )
        self.results['random_forest']['training_time'] = rf_time
        
        self.results['xgboost'], xgb_pred, xgb_proba = self.evaluate_model(
            xgb_model, X_test, y_test, 'XGBoost'
        )
        self.results['xgboost']['training_time'] = xgb_time
        
        self.results['logistic_regression'], lr_pred, lr_proba = self.evaluate_model(
            lr_model, X_test, y_test, 'Logistic Regression'
        )
        self.results['logistic_regression']['training_time'] = lr_time
        
        predictions = {
            'random_forest': {'y_pred': rf_pred, 'y_proba': rf_proba},
            'xgboost': {'y_pred': xgb_pred, 'y_proba': xgb_proba},
            'logistic_regression': {'y_pred': lr_pred, 'y_proba': lr_proba},
            'y_true': y_test.values
        }
        
        if lgb_model is not None:
            lgb_results, lgb_pred, lgb_proba = self.evaluate_model(
                lgb_model, X_test, y_test, 'LightGBM'
            )
            self.results['lightgbm'] = lgb_results
            self.results['lightgbm']['training_time'] = lgb_time
            predictions['lightgbm'] = {'y_pred': lgb_pred, 'y_proba': lgb_proba}
        
        return predictions
    
    def save_models(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            save_model(model, model_name, self.config['paths']['model_dir'])
    
    def save_all_results(self):
        """Save all results to files"""
        save_results(self.results, "model_results.json")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.results).T
        summary_df.to_csv("model_summary.csv")
        print("\nModel summary saved to model_summary.csv")

if __name__ == "__main__":
    trainer = ModelTrainer()
    data_path = trainer.config['paths']['output_dir']
    predictions = trainer.train_all_models(data_path)
    trainer.save_models()
    trainer.save_all_results()