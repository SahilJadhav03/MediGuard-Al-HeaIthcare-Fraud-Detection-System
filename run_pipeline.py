# run_pipeline.py
import os
import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    """Run complete project pipeline"""
    print("=" * 60)
    print("HEALTHCARE FRAUD DETECTION SYSTEM - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Create directories
    directories = ['data/raw', 'data/processed', 'models', 'notebooks', 'results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    steps = [
        ("Data Preprocessing", "preprocess"),
        ("Model Training", "train"),
        ("Model Evaluation", "evaluate"),
    ]
    
    for step_name, mode in steps:
        print(f"\n{'='*40}")
        print(f"STEP: {step_name}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        try:
            if mode == 'preprocess':
                from src.data_preprocessing import DataPreprocessor
                preprocessor = DataPreprocessor()
                preprocessor.run_pipeline()
                
            elif mode == 'train':
                from src.model_training import ModelTrainer
                trainer = ModelTrainer()
                trainer.train_all_models(trainer.config['paths']['output_dir'])
                trainer.save_models()
                trainer.save_all_results()
                
            elif mode == 'evaluate':
                from src.model_evaluation import ModelEvaluator
                from src.model_training import ModelTrainer
                from src.project_utils import load_model  # Changed
                
                trainer = ModelTrainer()
                evaluator = ModelEvaluator()
                
                data_path = trainer.config['paths']['output_dir']
                X_train, X_test, y_train, y_test = trainer.load_data(data_path)
                
                models = {
                    'Random Forest': load_model('random_forest'),
                    'XGBoost': load_model('xgboost'),
                    'Logistic Regression': load_model('logistic_regression')
                }
                
                evaluation_results = evaluator.run_complete_evaluation(
                    models, X_test, y_test, X_train, y_train
                )
            
            elapsed_time = time.time() - start_time
            print(f"✓ {step_name} completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED!")
    print("="*60)
    print("\nTo launch the dashboard, run:")
    print("python main.py --mode dashboard")
    print("or")
    print("streamlit run src/streamlit_app.py")

if __name__ == "__main__":
    run_pipeline()