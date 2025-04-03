import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import json
from typing import Dict, List, Union, Any
from collections import defaultdict
from datetime import datetime
import seaborn as sns

class UniversalJSONAnalyzer:
    def __init__(self):
        self.supported_operations = {
            'describe': self._describe,
            'value_counts': self._value_counts,
            'time_series': self._time_series,
            'correlation': self._correlation,
            'pattern_detect': self._pattern_detect,
            'nested_aggregate': self._nested_aggregate
        }
    
    def analyze(self, json_data: Union[str, Dict], operation: str, params: Dict = None) -> Dict:
        """Universal analysis entry point"""
        # print('jsondir', dir(json_data))
        data = self._load_data(json_data["tables"][0]["data"])
        if operation not in self.supported_operations:
            raise ValueError(f"Unsupported operation. Choose from: {list(self.supported_operations.keys())}")
        
        return self.supported_operations[operation](data, params or {})
    
    def _load_data(self, json_data: Union[str, Dict]) -> Any:
        """Load and normalize JSON data"""
        if isinstance(json_data, str):
            try:
                return json.loads(json_data)
            except json.JSONDecodeError:
                try:
                    with open(json_data, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    raise ValueError(f"Invalid JSON input: {str(e)}")
        return json_data
    
    def _flatten_data(self, data: Any, prefix: str = '') -> Dict:
        """Recursively flatten nested JSON structures"""
        flat = {}
        if isinstance(data, dict):
            for k, v in data.items():
                flat.update(self._flatten_data(v, f"{prefix}{k}."))
        elif isinstance(data, (list, tuple)) and len(data) > 0 and all(isinstance(x, dict) for x in data):
            # Handle lists of dictionaries (common in API responses)
            for i, item in enumerate(data):
                flat.update(self._flatten_data(item, f"{prefix}{i}."))
        else:
            flat[prefix[:-1]] = data
        return flat
    
    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert arbitrary JSON to DataFrame"""
        if isinstance(data, dict):
            if all(isinstance(v, (list, dict)) for v in data.values()):
                return pd.DataFrame.from_dict(data)
            return pd.DataFrame([self._flatten_data(data)])
        elif isinstance(data, list):
            return pd.DataFrame([self._flatten_data(x) for x in data])
        return pd.DataFrame(data)
                
    def _describe(self, data: Any, params: Dict) -> Dict:
        """Statistical description of all fields with NaN handling"""
        df = self._to_dataframe(data)
        
        # Convert NaN values to None (which becomes null in JSON)
        def clean_value(v):
            if pd.isna(v):
                return None
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            return v
        
        stats = {}
        for col in df.columns:
            col_stats = {}
            dtype = str(df[col].dtype)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'type': 'numeric',
                    'min': clean_value(df[col].min()),
                    'max': clean_value(df[col].max()),
                    'mean': clean_value(df[col].mean()),
                    'std': clean_value(df[col].std()),
                    'null_count': int(df[col].isna().sum())
                })
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_stats.update({
                    'type': 'datetime',
                    'min': clean_value(df[col].min()),
                    'max': clean_value(df[col].max()),
                    'null_count': int(df[col].isna().sum())
                })
            else:
                col_stats.update({
                    'type': 'categorical',
                    'unique_values': int(df[col].nunique()),
                    'top_value': clean_value(df[col].mode().iloc[0]) if not df[col].empty else None,
                    'null_count': int(df[col].isna().sum())
                })
            
            stats[col] = {'dtype': dtype, **col_stats}
        
        return {'statistics': stats}
    
    def _value_counts(self, data: Any, params: Dict) -> Dict:
        """Frequency analysis with visualization"""
        df = self._to_dataframe(data)
        column = params.get('column', df.columns[0])
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        counts = df[column].value_counts().to_dict()
        
        # Generate plot (with non-GUI backend)
        plt.switch_backend('Agg')  # Ensure no GUI backend is used
        fig, ax = plt.subplots(figsize=(10, 6))
        df[column].value_counts().head(20).plot(kind='barh', ax=ax)
        ax.set_title(f'Value Distribution: {column}')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Explicitly close the figure
        
        return {
            'column': column,
            'counts': counts,
            'plot': plot_base64  # Base64-encoded PNG
        }
    
    def _time_series(self, data: Any, params: Dict) -> Dict:
        """Temporal analysis with robust datetime handling and visualization"""
        # Configure matplotlib to use non-interactive backend
        
        df = self._to_dataframe(data)
        
        # Common datetime formats to try (order by likelihood)
        DATETIME_FORMATS = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',   # Dates
            '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', # Date-times
            '%Y%m%d', '%d%m%Y',                    # Compact dates
            '%Y-%m-%dT%H:%M:%S',                   # ISO format
            '%Y-%m-%d %H:%M:%S.%f'                 # With microseconds
        ]
        
        # Auto-detect time column if not specified
        time_col = params.get('date_column')
        if not time_col:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    time_col = col
                    break
                elif pd.api.types.is_string_dtype(df[col]):
                    for fmt in DATETIME_FORMATS:
                        try:
                            converted = pd.to_datetime(df[col], format=fmt, errors='raise')
                            if converted.notna().any():
                                time_col = col
                                df[col] = converted
                                break
                        except (ValueError, TypeError):
                            continue
                    if time_col:
                        break
        
        if not time_col:
            raise ValueError("No valid datetime column found or specifed")
        
        # Ensure proper datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Auto-detect value columns if not specified
        value_cols = params.get('value_columns')
        if not value_cols:
            value_cols = [
                col for col in df.columns 
                if pd.api.types.is_numeric_dtype(df[col]) 
                and col != time_col
            ]
        
        if not value_cols:
            raise ValueError("No numeric columns found for analysis")
        
        # Set default aggregation if not specified
        agg_func = params.get('agg_function', 'mean')
        
        # Resample time series with forward-fill for missing periods
        freq = params.get('frequency', 'D')  # Daily by default
        resampled = (
            df.set_index(time_col)[value_cols]
            .resample(freq)
            .agg(agg_func)
            .ffill()  # Forward fill missing values
        )
        
        # Generate plot with improved styling
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in value_cols:
            ax.plot(
                resampled.index,
                resampled[col],
                label=col,
                marker='o',
                markersize=4,
                linewidth=1.5,
                alpha=0.8
            )
        
        ax.set_title(f"Time Series Analysis ({freq} frequency)", pad=20)
        ax.set_xlabel(time_col, labelpad=10)
        ax.set_ylabel("Value", labelpad=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare JSON-safe output
        def clean_value(v):
            if pd.isna(v):
                return None
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                return v.isoformat()
            return str(v)
        
        sample_data = {
            time_col: [clean_value(x) for x in resampled.index[:10]],
            **{
                col: [clean_value(x) for x in resampled[col][:10]]
                for col in value_cols
            }
        }
        
        return {
            'time_column': time_col,
            'value_columns': value_cols,
            'frequency': freq,
            'aggregation': agg_func,
            'data_points': len(resampled),
            'start_date': clean_value(resampled.index.min()),
            'end_date': clean_value(resampled.index.max()),
            'plot': plot_base64,
            'sample_data': sample_data
        }
    
    def _correlation(self, data: Any, params: Dict) -> Dict:
        """Enhanced correlation analysis with visualization"""
        
        df = self._to_dataframe(data)
        
        # Auto-select numeric columns with NaN handling
        numeric_cols = [
            col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col]) 
            and not df[col].isna().all()  # Exclude all-NA columns
        ]
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
        
        # Calculate correlation with pairwise complete observations
        corr_matrix = df[numeric_cols].corr(method='pearson', min_periods=5)
        
        # Generate enhanced heatmap
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use seaborn for better heatmap visualization
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        
        ax.set_title("Correlation Matrix Heatmap", pad=20)
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare JSON-safe output
        def clean_value(v):
            if pd.isna(v):
                return None
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            return str(v)
        
        # Get extreme correlations with proper handling
        # def format_correlation_pairs(pairs):
        #     formatted = []
        #     for pair in pairs:
        #         if isinstance(pair, dict):
        #             formatted.append({k: clean_value(v) for k, v in pair.items()})
        #         elif isinstance(pair, (tuple, list)) and len(pair) == 2:
        #             formatted.append({
        #                 'pair': list(pair),
        #                 'correlation': clean_value(corr_matrix.loc[pair[0], pair[1]])
        #             })
        #         elif isinstance(pair, str):
        #             formatted.append({'column': pair})
        #     return formatted
        
        # Get extreme correlations - now properly formatted
        extreme_pos = self._get_extreme_correlation(corr_matrix, 'positive')
        extreme_neg = self._get_extreme_correlation(corr_matrix, 'negative')
        

        return {
            'correlation_matrix': {
                col: {k: clean_value(v) for k, v in row.items()}
                for col, row in corr_matrix.to_dict().items()
            },
            'plot': plot_base64,
            'strongest_positive': extreme_pos,
            'strongest_negative': extreme_neg,
            'numeric_columns_used': numeric_cols,
            'method': 'pearson',
            'min_valid_observations': 5,
            'warnings': [
                f"{col} has {df[col].isna().sum()} missing values" 
                for col in numeric_cols 
                if df[col].isna().any()
            ] if any(df[col].isna().any() for col in numeric_cols) else None
        }
    
    def _get_extreme_correlation(self, corr_matrix: pd.DataFrame, kind: str) -> List[Dict]:
        """Helper to find strongest correlations (returns list of dictionaries)"""
        matrix = corr_matrix.copy()
        np.fill_diagonal(matrix.values, np.nan)
        
        if kind == 'positive':
            threshold = matrix.stack().quantile(0.95)  # Top 5% positive correlations
            pairs = matrix.stack()[matrix.stack() >= threshold]
        else:
            threshold = matrix.stack().quantile(0.05)  # Bottom 5% negative correlations
            pairs = matrix.stack()[matrix.stack() <= threshold]
        
        # Return sorted list of correlation pairs
        return [
            {
                'column1': col1,
                'column2': col2,
                'value': float(corr_value)
            }
            for (col1, col2), corr_value in pairs.sort_values(ascending=False).items()
        ]
    
    def _pattern_detect(self, data: Any, params: Dict) -> Dict:
        """Pattern detection in text/numeric fields"""
        df = self._to_dataframe(data)
        patterns = defaultdict(list)
        
        # Numeric patterns
        for col in df.select_dtypes(include=np.number).columns:
            if (df[col] % 1 == 0).all():
                patterns['integer_columns'].append(col)
            if (df[col] > 0).all():
                patterns['positive_values'].append(col)
        
        # Text patterns
        for col in df.select_dtypes(include='object').columns:
            if df[col].str.contains(r'\d{4}-\d{2}-\d{2}').any():
                patterns['date_strings'].append(col)
            if df[col].str.contains(r'\$?\d+(,\d{3})*(\.\d+)?').any():
                patterns['currency_strings'].append(col)
        
        return {'detected_patterns': dict(patterns)}
    
    def _nested_aggregate(self, data: Any, params: Dict) -> Dict:
        """Handle nested JSON structures with aggregation"""
        if not isinstance(data, (dict, list)):
            raise ValueError("Data must be a dictionary or list for nested analysis")
        
        # For lists of records
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            df = pd.json_normalize(data)
            group_by = params.get('group_by', [col for col in df.columns if df[col].nunique() < 10])
            
            if not group_by:
                return {"message": "No suitable grouping columns found"}
            
            agg_results = {}
            for col in group_by:
                if col in df.columns:
                    agg_results[col] = df.groupby(col).size().to_dict()
            
            return {'grouped_counts': agg_results}
        
        # For deep nested structures
        elif isinstance(data, dict):
            flat = self._flatten_data(data)
            return {
                'structure_summary': {
                    'depth': max(len(k.split('.')) for k in flat.keys()),
                    'field_types': {k: type(v).__name__ for k, v in flat.items()}
                }
            }
        
        return {"message": "No aggregation performed on simple data"}