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
        """Statistical description of all fields with robust NaN handling and enhanced statistics"""
        df = self._to_dataframe(data)
        
        def clean_value(v):
            if pd.isna(v) or v is None:
                return None
            if isinstance(v, (np.floating, float)):
                return float(round(v, 4))
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                return v.isoformat()
            if isinstance(v, (list, tuple, np.ndarray)):
                return [clean_value(x) for x in v]
            return str(v)
        
        stats = {}
        for col in df.columns:
            col_stats = {}
            dtype = str(df[col].dtype)
            non_null_count = len(df[col]) - df[col].isna().sum()
            
            # Common stats for all types
            col_stats.update({
                'dtype': dtype,
                'total_count': len(df[col]),
                'null_count': int(df[col].isna().sum()),
                'non_null_count': int(non_null_count),
                'null_percentage': round(float(df[col].isna().mean() * 100), 2)
            })
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'type': 'numeric',
                    'min': clean_value(df[col].min()),
                    'max': clean_value(df[col].max()),
                    'mean': clean_value(df[col].mean()),
                    'median': clean_value(df[col].median()),
                    'std': clean_value(df[col].std()),
                    'variance': clean_value(df[col].var()),
                    'skewness': clean_value(df[col].skew()),
                    'kurtosis': clean_value(df[col].kurt()),
                    'quartiles': {
                        'q1': clean_value(df[col].quantile(0.25)),
                        'q2': clean_value(df[col].quantile(0.5)),
                        'q3': clean_value(df[col].quantile(0.75))
                    }
                })
                
                # Add histogram data for numeric columns
                if non_null_count > 0:
                    try:
                        hist, bins = np.histogram(df[col].dropna(), bins=10)
                        col_stats['histogram'] = {
                            'bins': clean_value(bins.tolist()),
                            'counts': clean_value(hist.tolist())
                        }
                    except Exception as e:
                        col_stats['histogram'] = f"Error calculating histogram: {str(e)}"
            
            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_stats.update({
                    'type': 'datetime',
                    'min': clean_value(df[col].min()),
                    'max': clean_value(df[col].max()),
                    'range_days': clean_value((df[col].max() - df[col].min()).days),
                    'weekday_counts': clean_value(df[col].dt.weekday.value_counts().to_dict())
                })
            
            # Categorical/string columns
            else:
                try:
                    # Handle mixed types by converting to string first
                    str_series = df[col].astype(str)
                    value_counts = str_series.value_counts()
                    
                    if not value_counts.empty:
                        top_value = value_counts.index[0]
                        # Use .iloc[0] to safely get the first value
                        top_freq = value_counts.iloc[0]
                        
                        # Calculate frequency percentage safely
                        freq_pct = None
                        if non_null_count > 0:
                            freq_pct = round((top_freq / non_null_count * 100), 2)
                        
                        col_stats.update({
                            'type': 'categorical',
                            'unique_values': int(str_series.nunique()),
                            'top_value': clean_value(top_value),
                            'top_frequency': int(top_freq),
                            'top_frequency_percentage': clean_value(freq_pct),
                            'value_counts': clean_value(
                                value_counts.head(10).to_dict()
                            )
                        })
                    else:
                        col_stats.update({
                            'type': 'categorical',
                            'unique_values': 0,
                            'top_value': None,
                            'top_frequency': None,
                            'top_frequency_percentage': None,
                            'value_counts': {}
                        })
                except Exception as e:
                    col_stats.update({
                        'type': 'categorical',
                        'unique_values': 'error',
                        'top_value': 'error',
                        'top_frequency': 'error',
                        'top_frequency_percentage': 'error',
                        'value_counts': 'error',
                        'error_message': str(e)
                    })
            
            stats[col] = col_stats
        
        # Add correlation matrix for numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) > 1:
            try:
                stats['_correlations'] = clean_value(
                    df[numeric_cols].corr().round(4).to_dict()
                )
            except Exception as e:
                stats['_correlations'] = f"Error calculating correlations: {str(e)}"
        
        return {
            'statistics': stats,
            'metadata': {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'numeric_columns': len(numeric_cols),
                'datetime_columns': len([col for col in df.columns 
                                    if pd.api.types.is_datetime64_any_dtype(df[col])]),
                'categorical_columns': len(df.columns) - len(numeric_cols) - 
                                    len([col for col in df.columns 
                                        if pd.api.types.is_datetime64_any_dtype(df[col])])
            }
        }
    
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
        df = self._to_dataframe(data)
        
        # Common datetime formats
        DATETIME_FORMATS = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M',
            '%Y%m%d', '%d%m%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        # Handle date column
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
            raise ValueError("No valid datetime column found or specified")
        
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Handle value column
        value_col = params.get('value_column')
        if not value_col:
            value_col = next(
                (col for col in df.columns 
                if pd.api.types.is_numeric_dtype(df[col]) 
                and col != time_col),
                None
            )
        
        if not value_col:
            raise ValueError("No numeric column found for analysis")
        
        # Get parameters
        agg_func = params.get('agg_function', 'sum')
        freq = params.get('frequency', 'D')
        
        # Resample data
        resampled = (
            df.set_index(time_col)[[value_col]]
            .resample(freq)
            .agg(agg_func)
            .ffill()
        )
        
        # Create plot with larger fonts
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(14, 8))  # Larger figure size
        
        # Set font sizes
        title_fontsize = 16
        axis_label_fontsize = 14
        tick_label_fontsize = 12
        annotation_fontsize = 16
        
        line, = ax.plot(
            resampled.index,
            resampled[value_col],
            label=value_col,
            marker='o',
            markersize=6,  # Slightly larger markers
            linewidth=2,
            alpha=0.8
        )
        
        # Configure title and labels with larger fonts
        ax.set_title(
            f"Time Series Analysis - {agg_func} of {value_col} ({freq} frequency)", 
            pad=20,
            fontsize=title_fontsize,
            fontweight='bold'
        )
        ax.set_xlabel(
            time_col, 
            labelpad=15,
            fontsize=axis_label_fontsize,
            fontweight='bold'
        )
        ax.set_ylabel(
            f"{agg_func} of {value_col}", 
            labelpad=15,
            fontsize=axis_label_fontsize,
            fontweight='bold'
        )
        
        # Configure tick labels
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (date, val) in enumerate(zip(resampled.index, resampled[value_col])):
            if i % max(1, len(resampled)//10) == 0:
                ax.annotate(
                    f"{val:.2f}",
                    (date, val),
                    textcoords="offset points",
                    xytext=(0,8),
                    ha='center',
                    fontsize=annotation_fontsize,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                )
        
        # Interactive hover effect
        annot = ax.annotate(
            "", xy=(0,0), 
            xytext=(20,20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
            arrowprops=dict(arrowstyle="->"),
            fontsize=annotation_fontsize
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            x, y = line.get_data()
            date = pd.to_datetime(x[ind["ind"][0]])
            value = y[ind["ind"][0]]
            annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
            annot.set_text(f"Date: {date.strftime('%Y-%m-%d')}\nValue: {value:.2f}")
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = line.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        # Rotate and adjust x-axis labels
        fig.autofmt_xdate(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare output
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
        
        return {
            'time_column': time_col,
            'value_column': value_col,
            'frequency': freq,
            'aggregation_function': agg_func,
            'data_points': len(resampled),
            'start_date': clean_value(resampled.index.min()),
            'end_date': clean_value(resampled.index.max()),
            'plot': plot_base64,
            'stats': {
                'mean': float(resampled[value_col].mean()),
                'median': float(resampled[value_col].median()),
                'min': float(resampled[value_col].min()),
                'max': float(resampled[value_col].max()),
                'std_dev': float(resampled[value_col].std()),
                'total': float(resampled[value_col].sum())
            }
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
        """Enhanced nested aggregation with multiple aggregation functions"""
        if not isinstance(data, (dict, list)):
            raise ValueError("Data must be a dictionary or list for nested analysis")

        # For tabular data (list of records)
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            df = pd.json_normalize(data)
            
            # Get parameters with defaults
            group_by = params.get('group_by', [])
            agg_columns = params.get('agg_columns', [
                col for col in df.columns 
                if pd.api.types.is_numeric_dtype(df[col])
            ])
            agg_functions = params.get('agg_functions', ['mean'])
            
            # Validate inputs
            if not group_by:
                group_by = [col for col in df.columns if df[col].nunique() < 10]
                if not group_by:
                    return {"error": "No suitable grouping columns found"}
            
            if not agg_columns:
                return {"error": "No numeric columns available for aggregation"}
            
            # Perform aggregations
            results = {}
            plots = []
            
            for func in agg_functions:
                try:
                    agg_result = df.groupby(group_by)[agg_columns].agg(func)
                    results[func] = agg_result.reset_index().to_dict('records')
                    
                    # Generate visualization for the first 3 columns
                    if len(agg_columns) > 0 and len(group_by) > 0:
                        for i, col in enumerate(agg_columns[:3]):
                            plt.switch_backend('Agg')
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Prepare data for plotting
                            groups = agg_result.index.tolist()
                            values = agg_result[col].values
                            
                            # Convert groups to strings if they aren't already
                            group_labels = [str(g[:20]) for g in groups]
                            
                            # Create bar plot
                            bars = ax.bar(group_labels, values)
                            
                            # Add title and labels
                            ax.set_title(f"{func.title()} of {col} by {group_by[0]}")
                            ax.set_xlabel(group_by[0])
                            ax.set_ylabel(col)
                            plt.xticks(rotation=45)
                            
                            # Add values on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * height,
                                        f'{height:.2f}',
                                        ha='center', va='bottom')
                            
                            # Add interactive hover effect
                            def format_tooltip(bar):
                                return f"{group_by[0]}: {group_labels[int(bar.get_x() + bar.get_width()/2)]}\nValue: {bar.get_height():.2f}"
                            
                            annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                                            textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w"),
                                            arrowprops=dict(arrowstyle="->"))
                            annot.set_visible(False)
                            
                            def update_annot(bar):
                                annot.xy = (bar.get_x() + bar.get_width()/2, bar.get_height())
                                annot.set_text(format_tooltip(bar))
                                annot.get_bbox_patch().set_alpha(0.8)
                            
                            def hover(event):
                                vis = annot.get_visible()
                                if event.inaxes == ax:
                                    for bar in bars:
                                        cont, ind = bar.contains(event)
                                        if cont:
                                            update_annot(bar)
                                            annot.set_visible(True)
                                            fig.canvas.draw_idle()
                                            return
                                if vis:
                                    annot.set_visible(False)
                                    fig.canvas.draw_idle()
                            
                            fig.canvas.mpl_connect("motion_notify_event", hover)
                            
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                            buf.seek(0)
                            plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                            plt.close()
                            
                            plots.append({
                                'function': func,
                                'column': col,
                                'group_by': group_by,
                                'agg_columns': agg_columns,
                                'results': results,
                                'plot': plot_base64
                            })
                except Exception as e:
                    results[func] = f"Aggregation failed: {str(e)}"
            
            return plots[0]
        
        # For deep nested structures
        elif isinstance(data, dict):
            flat = self._flatten_data(data)
            return {
                'structure_summary': {
                    'depth': max(len(k.split('.')) for k in flat.keys()),
                    'field_types': {k: type(v).__name__ for k, v in flat.items()},
                    'sample_values': {k: v for i, (k, v) in enumerate(flat.items()) if i < 5}
                }
            }
        
        return {"message": "No aggregation performed on simple data"}


    def _flatten_data(self, data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary structure"""
        items = {}
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_data(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items