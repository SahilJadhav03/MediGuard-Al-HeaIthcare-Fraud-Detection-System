# src/data_preprocessing.py - ADD THESE LINES AT THE TOP
import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC, SMOTE
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from project_utils import load_config  # This should work now


class DataPreprocessor:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        
    def load_data(self):
        """Load all datasets from Kaggle Healthcare Fraud Detection"""
        print("Loading datasets...")
        
        # Load datasets
        train_data = pd.read_csv(self.config['paths']['train_data'])
        beneficiary_data = pd.read_csv(self.config['paths']['beneficiary_data'])
        inpatient_data = pd.read_csv(self.config['paths']['inpatient_data'])
        outpatient_data = pd.read_csv(self.config['paths']['outpatient_data'])
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Beneficiary data shape: {beneficiary_data.shape}")
        print(f"Inpatient data shape: {inpatient_data.shape}")
        print(f"Outpatient data shape: {outpatient_data.shape}")
        
        return train_data, beneficiary_data, inpatient_data, outpatient_data
    
    def merge_datasets(self, train_data, beneficiary_data, inpatient_data, outpatient_data):
        """Merge claims with beneficiary info and attach provider labels"""
        print("Merging datasets...")

        # Combine inpatient and outpatient claims into a single claims table
        all_claims = pd.concat([inpatient_data, outpatient_data], ignore_index=True, sort=False)

        # Attach beneficiary demographics to each claim
        claims_with_beneficiary = pd.merge(all_claims, beneficiary_data, on='BeneID', how='left')

        # Attach provider-level fraud labels
        merged_data = pd.merge(claims_with_beneficiary, train_data, on='Provider', how='left')

        # Drop rows where provider label is missing (should be very few)
        before_drop = len(merged_data)
        merged_data = merged_data.dropna(subset=['PotentialFraud'])
        if before_drop != len(merged_data):
            print(f"Dropped {before_drop - len(merged_data)} claim rows without provider labels")

        print(f"Merged data shape: {merged_data.shape}")
        return merged_data
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")

        # Convert known numeric-like columns that may be read as objects
        numeric_like_cols = [
            'InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'NoOfMonths_PartACov',
            'NoOfMonths_PartBCov', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
            'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt'
        ]
        for col in numeric_like_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Separate numerical and categorical columns
        numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(include=['object']).columns)

        # Impute numerical columns
        num_imputer = SimpleImputer(strategy=self.config['preprocessing']['numerical_imputation'])
        if len(numerical_cols) > 0:
            num_imputed = num_imputer.fit_transform(df[numerical_cols])
            if num_imputed.shape[1] != len(numerical_cols):
                print(f"Warning: expected {len(numerical_cols)} numeric columns, got {num_imputed.shape[1]} after imputation")
                numerical_cols = numerical_cols[:num_imputed.shape[1]]
            df[numerical_cols] = pd.DataFrame(num_imputed, columns=numerical_cols, index=df.index)

        # Impute categorical columns
        cat_strategy = self.config['preprocessing']['categorical_imputation']
        if cat_strategy == 'mode':
            cat_strategy = 'most_frequent'
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        if len(categorical_cols) > 0:
            cat_imputed = cat_imputer.fit_transform(df[categorical_cols])
            df[categorical_cols] = pd.DataFrame(cat_imputed, columns=categorical_cols, index=df.index)

        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Create label encoders for each categorical column
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        return df, label_encoders
    
    def create_features(self, df):
        """Create new features for better fraud detection"""
        print("Creating new features...")

        # Map fraud label to binary
        if 'PotentialFraud' in df.columns:
            df['PotentialFraud'] = df['PotentialFraud'].map({'Yes': 1, 'No': 0})

        # Dates to datetime
        for date_col in ['DOB', 'DOD', 'ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Beneficiary age at death or end of 2019
        if 'DOB' in df.columns:
            ref_date = pd.Timestamp('2019-12-31')
            df['Age'] = (df['DOD'].fillna(ref_date) - df['DOB']).dt.days / 365.25

        # Gender encoding (1=Male, 2=Female in dataset)
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({1: 0, 2: 1})

        # Chronic condition flags (dataset uses 1 for True, 2 for False)
        chronic_cols = ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
                       'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                       'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
                       'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
                       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                       'ChronicCond_stroke']

        for col in chronic_cols:
            if col in df.columns:
                df[col] = df[col].map({1: 1, 2: 0}).fillna(0)

        # Create chronic condition count when columns exist
        available_chronic = [c for c in chronic_cols if c in df.columns]
        if available_chronic:
            df['ChronicCond_Count'] = df[available_chronic].sum(axis=1)

        # Claim amount ratios at claim level
        if 'InscClaimAmtReimbursed' in df.columns and 'DeductibleAmtPaid' in df.columns:
            df['Reimbursement_Ratio'] = df['InscClaimAmtReimbursed'] / (df['DeductibleAmtPaid'] + 1)

        return df

    def aggregate_provider_features(self, df):
        """Aggregate claim-level data to provider-level features"""
        print("Aggregating features at provider level...")

        # Compute provider risk scores before aggregation
        print("Computing provider risk profiles...")
        provider_fraud_rate = df.groupby('Provider')['PotentialFraud'].apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        provider_claim_variance = df.groupby('Provider')['InscClaimAmtReimbursed'].apply(lambda x: x.std() / (x.mean() + 1))
        provider_rejection_likelihood = df.groupby('Provider').apply(lambda x: (x['DeductibleAmtPaid'] > x['InscClaimAmtReimbursed']).sum() / len(x) if len(x) > 0 else 0)
        
        df['Provider_Fraud_Rate'] = df['Provider'].map(provider_fraud_rate).fillna(0)
        df['Provider_Claim_Variance'] = df['Provider'].map(provider_claim_variance).fillna(0)
        df['Provider_Rejection_Likelihood'] = df['Provider'].map(provider_rejection_likelihood).fillna(0)

        # Drop raw date columns (we keep derived Age instead)
        date_cols = ['DOB', 'DOD', 'ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']
        df = df.drop(columns=[c for c in date_cols if c in df.columns], errors='ignore')

        agg_map = {}

        if 'InscClaimAmtReimbursed' in df.columns:
            agg_map['InscClaimAmtReimbursed'] = ['sum', 'mean', 'std']
        if 'DeductibleAmtPaid' in df.columns:
            agg_map['DeductibleAmtPaid'] = ['sum', 'mean', 'std']
        if 'ClaimID' in df.columns:
            agg_map['ClaimID'] = 'count'
        if 'BeneID' in df.columns:
            agg_map['BeneID'] = pd.Series.nunique
        for phys_col in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'Physician']:
            if phys_col in df.columns:
                agg_map[phys_col] = pd.Series.nunique
        for diag_col in ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'DiagnosisGroupCode']:
            if diag_col in df.columns:
                agg_map[diag_col] = pd.Series.nunique
        for proc_col in ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
                         'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6']:
            if proc_col in df.columns:
                agg_map[proc_col] = pd.Series.nunique
        for bene_col in ['Age', 'Gender', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov']:
            if bene_col in df.columns:
                agg_map[bene_col] = ['mean']
        for chronic_col in ['ChronicCond_Count']:
            if chronic_col in df.columns:
                agg_map[chronic_col] = ['mean']
        
        # Add provider risk features
        if 'Provider_Fraud_Rate' in df.columns:
            agg_map['Provider_Fraud_Rate'] = 'first'
        if 'Provider_Claim_Variance' in df.columns:
            agg_map['Provider_Claim_Variance'] = 'first'
        if 'Provider_Rejection_Likelihood' in df.columns:
            agg_map['Provider_Rejection_Likelihood'] = 'first'

        provider_grouped = df.groupby('Provider').agg(agg_map)

        # Flatten MultiIndex columns
        provider_grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                    for col in provider_grouped.columns]
        provider_grouped = provider_grouped.reset_index()

        # Add statistical anomaly detection at provider level
        print("Detecting anomalies...")
        if 'InscClaimAmtReimbursed_mean' in provider_grouped.columns:
            overall_mean = provider_grouped['InscClaimAmtReimbursed_mean'].mean()
            overall_std = provider_grouped['InscClaimAmtReimbursed_mean'].std()
            provider_grouped['Claim_Amount_Anomaly'] = (
                (provider_grouped['InscClaimAmtReimbursed_mean'] - overall_mean).abs() > 2 * overall_std
            ).astype(int)

        if 'InscClaimAmtReimbursed_count' in provider_grouped.columns:
            overall_count_mean = provider_grouped['InscClaimAmtReimbursed_count'].mean()
            overall_count_std = provider_grouped['InscClaimAmtReimbursed_count'].std()
            provider_grouped['High_Claim_Volume_Anomaly'] = (
                provider_grouped['InscClaimAmtReimbursed_count'] > overall_count_mean + 2 * overall_count_std
            ).astype(int)

        # Re-attach fraud labels (already numeric from create_features)
        labels = df[['Provider', 'PotentialFraud']].drop_duplicates(subset=['Provider'])
        provider_grouped = provider_grouped.merge(labels, on='Provider', how='left')

        print(f"Provider-level dataset shape: {provider_grouped.shape}")
        return provider_grouped
    
    def prepare_data_for_modeling(self, df, target_col='PotentialFraud'):
        """Prepare data for modeling with SMOTENC"""
        print("Preparing data for modeling...")

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        # Identify categorical columns for SMOTENC (treat integer columns as categorical/binary)
        categorical_features = X.select_dtypes(include=['int64']).columns.tolist()
        categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['preprocessing']['test_size'],
            random_state=self.config['preprocessing']['random_state'],
            stratify=y
        )

        # Apply SMOTENC
        print(f"Before SMOTENC - Class distribution: {pd.Series(y_train).value_counts().to_dict()}")

        if categorical_indices:
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy=self.config['smote']['sampling_strategy'],
                k_neighbors=self.config['smote']['k_neighbors'],
                random_state=self.config['smote']['random_state']
            )
        else:
            sampler = SMOTE(
                sampling_strategy=self.config['smote']['sampling_strategy'],
                k_neighbors=self.config['smote']['k_neighbors'],
                random_state=self.config['smote']['random_state']
            )

        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

        print(f"After SMOTENC - Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        # Scale numerical features
        numerical_cols = X_train_resampled.select_dtypes(include=[np.float64]).columns
        scaler = StandardScaler()
        X_train_resampled[numerical_cols] = scaler.fit_transform(X_train_resampled[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        return X_train_resampled, X_test, y_train_resampled, y_test, scaler
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save processed data"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        print(f"Processed data saved to {output_dir}")
    
    def run_pipeline(self):
        """Run complete preprocessing pipeline"""
        # Load data
        train_data, beneficiary_data, inpatient_data, outpatient_data = self.load_data()
        
        # Merge datasets
        merged_data = self.merge_datasets(train_data, beneficiary_data, inpatient_data, outpatient_data)

        # Handle missing values at claim level then create features
        merged_data = self.handle_missing_values(merged_data)
        merged_data = self.create_features(merged_data)

        # Aggregate to provider level (model target is provider fraud)
        provider_level_df = self.aggregate_provider_features(merged_data)

        # Final imputation and encoding
        provider_level_df = self.handle_missing_values(provider_level_df)
        provider_level_df, label_encoders = self.encode_categorical_features(provider_level_df)

        # Prepare data for modeling
        X_train, X_test, y_train, y_test, scaler = self.prepare_data_for_modeling(provider_level_df)
        
        # Save processed data
        self.save_processed_data(
            X_train, X_test, y_train, y_test,
            self.config['paths']['output_dir']
        )
        
        return X_train, X_test, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocessor.run_pipeline()