# --- Datatool.py (updated for conversation history) ---
"""
Datatool: Core library for dataset analysis, in-terminal visualization,
LLM-driven summaries/queries using GROQ API, with interactive context history.
"""
import os
import json
import logging
import warnings
from typing import Dict, List, Union, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import plotext as pltxt
import groq
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings('ignore')

class DataTool:
    def __init__(self, api_key: str = None):
        # Use provided API key or environment variables
        self.api_key = api_key or os.environ.get('GROQ_API') or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Provide it via --api-key or set GROQ_API/GROQ_API_KEY env var.")
        self.client = groq.Groq(api_key=self.api_key)
        self.logger = self._setup_logger()
        self.context_cache = None  # For caching dataset context in interactive mode

    def _setup_logger(self):
        log_path = 'datatool_logs'
        os.makedirs(log_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_path, 'datatool.log'),
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        return logging.getLogger(__name__)

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load CSV data via Dask and return a pandas DataFrame."""
        ddf = dd.read_csv(data_path)
        return ddf.compute()

    def analyze_data(self, df: pd.DataFrame, target_col: str) -> dict:
        """Compute comprehensive dataset statistics and distributions."""
        # Basic dataset info
        info = {
            'shape': df.shape,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': {col: str(mem) for col, mem in zip(df.columns, df.memory_usage(deep=True, index=True))},
        }
        
        # Missing values analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        info['missing_values'] = {
            'total_missing_cells': int(missing.sum()),
            'total_missing_pct': float(missing.sum() / (df.shape[0] * df.shape[1]) * 100),
            'by_column': {col: int(val) for col, val in missing.items()},
            'by_column_pct': {col: float(val) for col, val in missing_pct.items()},
            'complete_rows': int(df.dropna().shape[0]),
            'complete_rows_pct': float(df.dropna().shape[0] / df.shape[0] * 100)
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        info['numeric_stats'] = {}
        for col in numeric_cols:
            info['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                '25%': float(df[col].quantile(0.25)),
                '75%': float(df[col].quantile(0.75)),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'zero_count': int((df[col] == 0).sum()),
                'zero_pct': float((df[col] == 0).sum() / len(df) * 100)
            }
        
        # Categorical columns statistics
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        info['categorical_stats'] = {}
        for col in cat_cols:
            value_counts = df[col].value_counts()
            info['categorical_stats'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_5_values': {str(k): int(v) for k, v in value_counts.head().items()},
                'top_value': str(value_counts.index[0]) if not value_counts.empty else None,
                'top_value_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'top_value_pct': float(value_counts.iloc[0] / len(df) * 100) if not value_counts.empty else 0
            }
        
        # Date columns detection and stats
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue
        
        info['date_stats'] = {}
        for col in date_cols:
            dates = pd.to_datetime(df[col], errors='coerce')
            info['date_stats'][col] = {
                'min_date': str(dates.min()),
                'max_date': str(dates.max()),
                'range_days': float((dates.max() - dates.min()).days),
                'null_dates': int(dates.isnull().sum())
            }
        
        # Target column analysis
        if target_col in df.columns:
            info['target_analysis'] = {
                'name': target_col,
                'dtype': str(df[target_col].dtype)
            }
            
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                value_counts = df[target_col].value_counts()
                info['target_analysis']['type'] = 'categorical'
                info['target_analysis']['unique_values'] = int(df[target_col].nunique())
                info['target_analysis']['distribution'] = {str(k): int(v) for k, v in value_counts.items()}
                info['target_analysis']['distribution_pct'] = {str(k): float(v/len(df)*100) for k, v in value_counts.items()}
            else:
                info['target_analysis']['type'] = 'numeric'
                desc = df[target_col].describe()
                info['target_analysis']['distribution'] = {k: float(v) for k, v in desc.items()}
        
        # Duplicate analysis
        info['duplicates'] = {
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_rows_pct': float(df.duplicated().sum() / len(df) * 100)
        }
        
        return info

    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for numeric features."""
        num_df = df.select_dtypes(include=[np.number])
        return num_df.corr()

    def print_cli_heatmap(self, corr_matrix: pd.DataFrame):
        """Render a heatmap of the correlation matrix in-terminal using plotext."""
        pltxt.clear_figure()
        pltxt.heatmap(corr_matrix)
        pltxt.title('Feature Correlation Heatmap')
        pltxt.show()

    def print_cli_scatter(self, df: pd.DataFrame, corr_matrix: pd.DataFrame, target_col: str, top_n: int = 5):
        """Render scatter plots for top N features most correlated with the target using plotext."""
        if target_col not in corr_matrix.columns:
            print(f"Target column '{target_col}' not in dataset for correlation.")
            return
        corrs = corr_matrix[target_col].abs().drop(labels=[target_col]).sort_values(ascending=False)
        top_feats = corrs.head(top_n).index.tolist()
        for feat in top_feats:
            pltxt.clear_figure()
            x = df[feat].dropna().tolist()
            y = df[target_col].loc[df[feat].notnull()].tolist()
            pltxt.scatter(x, y, marker='x')
            pltxt.title(f"{feat} vs {target_col} (corr={corr_matrix.loc[feat, target_col]:.2f})")
            pltxt.xlabel(feat)
            pltxt.ylabel(target_col)
            pltxt.show()

    def print_cli_histogram(self, df: pd.DataFrame, column: str):
        """Render a histogram for a numeric column using plotext."""
        if column not in df.columns or not is_numeric_dtype(df[column]):
            print(f"Column '{column}' not found or not numeric.")
            return
        
        pltxt.clear_figure()
        values = df[column].dropna().tolist()
        pltxt.hist(values, bins=20)
        pltxt.title(f'Histogram of {column}')
        pltxt.show()

    def print_distribution_bars(self, df: pd.DataFrame, categorical_col: str, top_n: int = 10):
        """Render bar chart for categorical column distribution using plotext."""
        if categorical_col not in df.columns:
            print(f"Column '{categorical_col}' not found in dataset.")
            return
        
        value_counts = df[categorical_col].value_counts().head(top_n)
        labels = [str(x) for x in value_counts.index.tolist()]
        values = value_counts.values.tolist()
        
        pltxt.clear_figure()
        pltxt.bar(labels, values)
        pltxt.title(f'Distribution of {categorical_col} (Top {len(labels)})')
        pltxt.show()

    def print_missing_data_chart(self, df: pd.DataFrame):
        """Render a bar chart showing missing data by column using plotext."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        cols_with_missing = [col for col in df.columns if missing[col] > 0]
        if not cols_with_missing:
            print("No missing data in the dataset.")
            return
        
        values = [missing_pct[col] for col in cols_with_missing]
        
        pltxt.clear_figure()
        pltxt.bar(cols_with_missing, values)
        pltxt.title('Missing Data by Column (%)')
        pltxt.show()
    
    def print_basic_stats(self, info: dict):
        """Print basic dataset statistics to the console."""
        print(f"\n=== Dataset Overview ===")
        print(f"Rows: {info['rows']}, Columns: {info['columns']}")
        print(f"Memory Usage: {sum(int(str(m).replace(' bytes', '')) for m in info['memory_usage'].values()) / 1_000_000:.2f} MB")
        print(f"Missing Data: {info['missing_values']['total_missing_pct']:.2f}% of all cells")
        print(f"Complete Rows: {info['missing_values']['complete_rows_pct']:.2f}% ({info['missing_values']['complete_rows']} rows)")
        print(f"Duplicate Rows: {info['duplicates']['duplicate_rows_pct']:.2f}% ({info['duplicates']['duplicate_rows']} rows)")
        
        print("\n=== Column Types ===")
        dtype_counts = {}
        for dtype in info['dtypes'].values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} columns")
        
        if 'target_analysis' in info:
            print(f"\n=== Target Column: {info['target_analysis']['name']} ===")
            print(f"Type: {info['target_analysis']['type']}")
            if info['target_analysis']['type'] == 'categorical':
                print(f"Unique Values: {info['target_analysis']['unique_values']}")
                print("Top 3 categories:")
                for i, (val, pct) in enumerate(list(info['target_analysis']['distribution_pct'].items())[:3]):
                    print(f"  {val}: {pct:.2f}%")
            else:
                print(f"Mean: {info['target_analysis']['distribution']['mean']:.2f}")
                print(f"Min: {info['target_analysis']['distribution']['min']:.2f}")
                print(f"Max: {info['target_analysis']['distribution']['max']:.2f}")
        
        # Display columns with highest missing percentages
        if info['missing_values']['total_missing_cells'] > 0:
            print("\n=== Top 5 Columns with Missing Data ===")
            missing_cols = sorted(info['missing_values']['by_column_pct'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for col, pct in missing_cols:
                if pct > 0:
                    print(f"{col}: {pct:.2f}% missing")

    def get_llm_summary(self, data_info: dict, corr_matrix: pd.DataFrame, target_col: str) -> str:
        """Send dataset summary and correlations to GROQ LLM and return human-readable summary."""
        if target_col in corr_matrix.columns:
            top = corr_matrix[target_col].abs().drop(labels=[target_col]).sort_values(ascending=False).head(5).to_dict()
        else:
            top = {}
        
        # Create and cache the context for future queries
        self._cache_context(data_info, corr_matrix, target_col)
        
        prompt = (
            f"Provide a concise, human-readable summary of the dataset.\n"
            f"Dataset info: {json.dumps(data_info)}\n"
            f"Top correlations with target ({target_col}): {json.dumps(top)}"
        )
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"LLM summary failed: {e}")
            return "Failed to generate summary via LLM."

    def _cache_context(self, data_info: dict, corr_matrix: pd.DataFrame, target_col: str):
        """Cache dataset info and correlations for reuse in interactive queries."""
        if target_col in corr_matrix.columns:
            top_corrs = corr_matrix[target_col].abs().drop(labels=[target_col]).sort_values(ascending=False).head(5).to_dict()
        else:
            top_corrs = {}
        self.context_cache = {
            'data_info': data_info,
            'top_correlations': top_corrs,
            'target_col': target_col,
            'initial_prompt': (
                f"You are an expert data scientist analyzing a dataset."
                f" Key dataset characteristics: {len(data_info['column_names'])} columns, {data_info['rows']} rows."
                f" Target column: {target_col}."
            )
        }
        self.logger.info("Dataset context cached for interactive query mode")

    def ask_query(
        self,
        query: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
        data_info: dict = None,
        corr_matrix: pd.DataFrame = None,
        target_col: str = None
    ) -> str:
        """Ask a custom query to the GROQ LLM with dataset and conversation context."""
        # Ensure context is available
        if self.context_cache is None and (data_info is None or target_col is None):
            return "Error: Dataset context not available. Run analysis first."

        # Determine which context to use
        if self.context_cache:
            context = self.context_cache
        else:
            # Fallback: build temporary context
            if target_col in corr_matrix.columns:
                top = corr_matrix[target_col].abs().drop(labels=[target_col]).sort_values(ascending=False).head(5).to_dict()
            else:
                top = {}
            context = {
                'data_info': data_info,
                'top_correlations': top,
                'target_col': target_col,
                'initial_prompt': (
                    f"You are analyzing a dataset with {len(data_info['column_names'])} columns "
                    f"and {data_info['rows']} rows."
                )
            }

        # Build the prompt with conversation history
        prompt = context['initial_prompt']
        if conversation_history:
            for idx, (prev_q, prev_a) in enumerate(conversation_history, start=1):
                prompt += f"\n\nPrevious Q{idx}: {prev_q}\nPrevious A{idx}: {prev_a}"

        # Append dataset info and current question
        prompt += (
            f"\n\nDataset info: {json.dumps(context['data_info'])}\n"
            f"Top correlations with target ({context['target_col']}): {json.dumps(context['top_correlations'])}\n\n"
            f"Current Question: {query}\n\n"
            f"Provide a clear, concise answer based on the dataset information and prior conversation."
        )

        # Send to LLM
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            return f"Failed to answer query via LLM: {str(e)}"
