import joblib
import json
import logging
import os
import warnings
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             precision_score, recall_score, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pycaret.classification import setup as classification_setup, compare_models as classification_compare
from pycaret.regression import setup as regression_setup, compare_models as regression_compare
import groq

warnings.filterwarnings('ignore')


class MLCLIPipeline:
    def __init__(self, api_key: str = None, test_size: float = 0.2, batch_size: int = 10000, temperature: float = None):
        # Use the provided API key or environment variables
        self.api_key = api_key or os.environ.get('GROQ_API') or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Provide it as a parameter or set GROQ_API environment variable.")
        self.client = groq.Groq(api_key=self.api_key)
        self.test_size = test_size
        self.batch_size = batch_size
        # New temperature parameter for LLM calls
        self.temperature = temperature
        self.logger = self._setup_logger()
        self.preprocessor = None
        self.task_type = None
        self.numeric_cols = None
        self.categorical_cols = None

    def _setup_logger(self):
        log_path = 'mlcli_logs'
        os.makedirs(log_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_path, 'mlcli.log'),
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        return logging.getLogger(__name__)

    def _analyze_data(self, df: pd.DataFrame, target_col: str) -> dict:
        data_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
        }
        if self.task_type == 'classification':
            data_info['target_distribution'] = df[target_col].value_counts().to_dict()
        else:
            target_desc = df[target_col].describe().to_dict()
            data_info['target_distribution'] = {k: float(v) for k, v in target_desc.items()}
        return data_info

    def _get_llm_plan(self, data_info: dict, target_col: str) -> dict:
        prompt = (
            f"Analyze this dataset and provide a machine learning pipeline plan. "
            f"Dataset info: {json.dumps(data_info)}. Target column: {target_col}."
        )
        try:
            # Build chat completion parameters
            params = {
                'model': "llama-3.3-70b-versatile",
                'messages': [{"role": "user", "content": prompt}]
            }
            if self.temperature is not None:
                params['temperature'] = self.temperature
            else:
                # fallback to groq-specific managing_strength if temperature not set
                params['managing_strength'] = 0.7
            completion = self.client.chat.completions.create(**params)
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"LLM query failed: {str(e)}")
            return self._default_plan()

    def _default_plan(self) -> dict:
        return {
            'preprocessing': {
                'numeric_strategy': 'mean',
                'categorical_strategy': 'most_frequent',
                'scale_features': True,
                'feature_engineering': {
                    'text_features': True,
                    'date_features': True
                }
            },
            'model_selection': {
                'cv_folds': 5
            },
            'test_size': self.test_size
        }

    def _handle_feature_engineering(self, df: pd.DataFrame, plan: dict) -> pd.DataFrame:
        df = df.copy()
        if plan['preprocessing'].get('feature_engineering', {}).get('text_features', False):
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in text_cols:
                vectorizer = CountVectorizer(max_features=100)
                try:
                    tfidf = vectorizer.fit_transform(df[col].fillna(''))
                    tfidf_df = pd.DataFrame(
                        tfidf.toarray(),
                        columns=[f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()],
                        index=df.index
                    )
                    df = pd.concat([df, tfidf_df], axis=1)
                except Exception as e:
                    self.logger.error(f"Error processing text column {col}: {str(e)}")
        if plan['preprocessing'].get('feature_engineering', {}).get('date_features', False):
            date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
            for col in date_cols:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
        return df

    def _build_preprocessor(self, plan: dict) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=plan['preprocessing']['numeric_strategy'])),
            ('scaler', StandardScaler() if plan['preprocessing'].get('scale_features', False) else 'passthrough')
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=plan['preprocessing']['categorical_strategy'])),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])

    def run_pipeline(self, data_path: str, target_col: str, output_dir: str = 'saved_models'):
        try:
            # Data Loading
            ddf = dd.read_csv(data_path)
            df = ddf.compute()
            
            # Analyze raw data
            raw_data_info = self._analyze_data(df, target_col)
            
            # Data Splitting
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
            
            # Determine task type
            self.task_type = 'classification' if y_train.nunique() < 10 else 'regression'
            
            # Get pipeline plan
            train_df = pd.concat([X_train, y_train], axis=1)
            data_info = self._analyze_data(train_df, target_col)
            plan = self._get_llm_plan(data_info, target_col)
            plan['test_size'] = self.test_size  # Add test size to plan for reporting
            
            # Feature Engineering
            X_train = self._handle_feature_engineering(X_train, plan)
            X_test = self._handle_feature_engineering(X_test, plan)
            
            # Identify columns
            self.numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
            self.categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
            
            # Build preprocessor
            self.preprocessor = self._build_preprocessor(plan)
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)
            
            # Prepare training data
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
            X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
            train_data = pd.concat([X_train_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
            
            # Model Training
            if self.task_type == 'classification':
                classification_setup(
                    data=train_data,
                    target=target_col,
                    fold=plan['model_selection']['cv_folds'],
                    verbose=False
                )
                best_model = classification_compare(n_select=1)
            else:
                regression_setup(
                    data=train_data,
                    target=target_col,
                    fold=plan['model_selection']['cv_folds'],
                    verbose=False
                )
                best_model = regression_compare(n_select=1)
            
            if isinstance(best_model, list):
                best_model = best_model[0]
            
            # Model Evaluation
            y_pred = best_model.predict(X_test_df)
            if self.task_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
            else:
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
            
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(output_dir, f'model_{timestamp}.joblib')
            joblib.dump({'model': best_model, 'preprocessor': self.preprocessor}, model_path)
            
            return {
                'model': best_model,
                'metrics': metrics,
                'preprocessor': self.preprocessor,
                'plan': plan,
                'raw_data_info': raw_data_info,
                'task_type': self.task_type,
                'model_path': model_path
            }
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise