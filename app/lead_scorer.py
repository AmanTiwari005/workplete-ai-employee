import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap

class LeadScorer:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.explainer = None
        
    def create_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train, feature_names=None, preprocessor=None):
        if not self.model:
            self.create_model()
        
        self.model.fit(X_train, y_train)
        
        if preprocessor:
            self.feature_names = preprocessor.get_feature_names_out()
        else:
            self.feature_names = feature_names
        
        if self.model_type in ['random_forest', 'gradient_boosting']:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(self.model, X_train)
        
        return self
    
    def predict(self, X):
        if not self.model:
            raise ValueError("Model not trained. Call train first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.model:
            raise ValueError("Model not trained. Call train first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        if not self.model:
            raise ValueError("Model not trained. Call train first.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        return metrics
    
    def get_feature_importance(self, top_n=None):
        if not self.model or self.feature_names is None or len(self.feature_names) == 0:
            raise ValueError("Model not trained or feature names not provided.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = abs(self.model.coef_[0])
        else:
            raise ValueError(f"Model type {self.model_type} does not support feature importance.")
        
        feature_imp = dict(zip(self.feature_names, importances))
        sorted_importances = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        
        if top_n:
            sorted_importances = sorted_importances[:top_n]
            
        return dict(sorted_importances)
    
    def explain_prediction(self, X):
        if not self.explainer:
            raise ValueError("Explainer not created. Train the model first.")
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (conversion) explanations
        
        return shap_values
    
    def save_model(self, filepath="models/lead_scorer.joblib"):
        if not self.model:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath="models/lead_scorer.joblib"):
        model_data = joblib.load(filepath)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        
        if instance.model_type in ['random_forest', 'gradient_boosting']:
            instance.explainer = shap.TreeExplainer(instance.model)
        else:
            instance.explainer = None
        
        return instance