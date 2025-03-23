import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import datetime as dt
import joblib
import os

class DataProcessor:
    def __init__(self):
        self.pipeline = None
        self.categorical_features = ['company_size', 'industry', 'region', 'lead_source']
        self.numerical_features = [
            'inquiry_frequency', 
            'time_spent_on_site', 
            'pages_visited',
            'downloads', 
            'email_opens', 
            'previous_orders',
            'quote_requests'
        ]
        
    def prepare_features(self, df):
        """Prepare features from raw data."""
        df = df.copy()
        
        # Ensure last_contact and initial_contact are datetime objects
        df['last_contact'] = pd.to_datetime(df['last_contact'])
        df['initial_contact'] = pd.to_datetime(df['initial_contact'])
        today = dt.datetime.now()
        
        # Calculate days since last contact
        df['days_since_last_contact'] = (today - df['last_contact']).dt.days
        
        # Calculate contact duration
        df['contact_duration'] = (df['last_contact'] - df['initial_contact']).dt.days
        
        # Add these new features to numerical_features if not already present
        if 'days_since_last_contact' not in self.numerical_features:
            self.numerical_features.extend(['days_since_last_contact', 'contact_duration'])
        
        # Drop date columns and other non-feature columns
        processed_df = df.drop(['last_contact', 'initial_contact', 'customer_id', 'contract_value'], axis=1, errors='ignore')
        
        return processed_df
    
    def create_preprocessing_pipeline(self):
        """Create a preprocessing pipeline for feature transformation."""
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.pipeline
    
    def process_data(self, df, target_col='converted', test_size=0.2, random_state=42):
        """Process data and split into train and test sets."""
        processed_df = self.prepare_features(df)
        
        # Extract target
        X = processed_df.drop(target_col, axis=1)
        y = processed_df[target_col]
        
        # Create and fit preprocessing pipeline
        if not self.pipeline:
            self.create_preprocessing_pipeline()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit transform pipeline on training data
        X_train_processed = self.pipeline.fit_transform(X_train)
        # Transform test data
        X_test_processed = self.pipeline.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test, X.columns
    
    def process_new_lead(self, lead_data):
        """Process a new lead for prediction."""
        if not self.pipeline:
            raise ValueError("Pipeline not fitted. Call process_data first.")
        
        # Convert lead_data to DataFrame if it's a dict or Series
        if isinstance(lead_data, dict):
            lead_data = pd.DataFrame([lead_data])
        elif isinstance(lead_data, pd.Series):
            lead_data = pd.DataFrame([lead_data])
        
        processed_lead = self.prepare_features(lead_data)
        processed_lead = processed_lead.drop('converted', axis=1, errors='ignore')
        
        return self.pipeline.transform(processed_lead)
    
    def save_pipeline(self, filepath="models/data_pipeline.joblib"):
        """Save the preprocessing pipeline to a file."""
        if not self.pipeline:
            raise ValueError("Pipeline not fitted. Call process_data first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
    
    @classmethod
    def load_pipeline(cls, filepath="models/data_pipeline.joblib"):
        """Load a preprocessing pipeline from a file."""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance