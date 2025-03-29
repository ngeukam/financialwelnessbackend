import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')
from PyPDF2 import PdfReader
import re
from collections import Counter
from wordcloud import WordCloud  # pip install wordcloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def analyze_data(file_path, operation, parameters):
      # Determine file type
    if file_path.endswith('.pdf'):
        return analyze_pdf(file_path, operation, parameters)
    elif file_path.endswith(('.csv', '.xlsx', '.xls')):
        return analyze_tabular_data(file_path, operation, parameters)
    else:
        raise ValueError("Unsupported file format")

def analyze_tabular_data(file_path, operation, parameters):
    """Original tabular data analysis functions"""
    # Read file based on extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # Excel
        df = pd.read_excel(file_path)
    
    # Perform requested operation
    if operation == 'describe':
        result = df.describe().to_dict()
        return {'result': result}
    elif operation == 'correlation':
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
        df = df.dropna()  # Drop NaNs to avoid issues
        correlation_matrix = df.corr()
        
        # Convert numpy types to native Python types for JSON serialization
        result = {
            col: {
                other_col: float(value) if pd.notna(value) else None
                for other_col, value in row.items()
            }
            for col, row in correlation_matrix.to_dict().items()
        }
        
        return {'result': result}
    elif operation == 'value_counts':
        column = parameters.get('column')
        if not column or column not in df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        counts = df[column].value_counts().to_dict()
        
        # Generate and save plot to a bytes buffer
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        df[column].value_counts().plot(kind='barh')
        plt.title(f'Value Counts for {column}')
        plt.xlabel('Count')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64 for easy transfer
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'result': counts,
            'plot': image_base64
        }
    elif operation == 'linear_regression':
        x_col = parameters.get('x_column')
        y_col = parameters.get('y_column')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Both x_column and y_column must be valid columns")
        
        # Convert columns to numeric, forcing errors to NaN
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        
        # Drop NaN values
        df = df.dropna(subset=[x_col, y_col])
        
        # Ensure we have at least two valid data points
        if len(df) < 2:
            raise ValueError("Not enough valid data points for linear regression")
        
        # Ensure no infinite values
        if np.isinf(df[x_col]).any() or np.isinf(df[y_col]).any():
            raise ValueError("Columns contain infinite values")
        
        x = df[x_col].values
        y = df[y_col].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label='Data points')
        plt.plot(x, slope * x + intercept, color='red', label='Regression line')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Linear Regression: {y_col} vs {x_col}')
        plt.legend()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        result = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'plot': plot_base64
        }
        
        # else:
        #     raise ValueError(f"Unsupported operation: {operation}")
        
        return {
                'operation': operation,
                'parameters': parameters,
                'result': result
            }

    elif operation == 'cluster':
        n_clusters = parameters.get('n_clusters', 3)
        features = parameters.get('features', df.columns.tolist())
        
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        if df.shape[0] < n_clusters:
            raise ValueError("Not enough valid data points for clustering")
        
        X = df[features].values
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        df['cluster'] = clusters
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f'K-Means Clustering (k={n_clusters})')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        result = {
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'plot': plot_base64,
            'sample_clusters': df[[features[0], features[1], 'cluster']].head(10).to_dict('records')
        }
    
    elif operation == 'pca':
        n_components = parameters.get('n_components', 2)
        features = parameters.get('features', df.columns.tolist())
        
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        if df.shape[0] < n_components:
            raise ValueError("Not enough valid data points for PCA")
        
        X = df[features].values
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(components[:, 0], components[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Results')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        result = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'components': pca.components_.tolist(),
            'plot': plot_base64
        }
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return {
        'operation': operation,
        'parameters': parameters,
        'result': result
    }


def analyze_pdf(file_path, operation, parameters):
    """PDF-specific analysis functions"""
    # Read PDF file
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Perform requested operation
    if operation == "text_stats":
        # Basic text statistics
        words = re.findall(r'\w+', text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if len(s.strip()) > 0])
        
        # Most common words (excluding stopwords)
        stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for'])
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        common_words = Counter(filtered_words).most_common(10)
        
        return {
            "operation": operation,
            "result": {
                "page_count": len(reader.pages),
                "word_count": word_count,
                "unique_words": unique_words,
                "sentence_count": sentence_count,
                "avg_word_length": sum(len(w) for w in words)/word_count if word_count > 0 else 0,
                "most_common_words": common_words,
                "metadata": reader.metadata
            }
        }
    
    elif operation == "word_cloud":
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Save plot to base64
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            "operation": operation,
            "result": {
                "word_cloud": plot_base64
            }
        }
    
    elif operation == "search_text":
        # Search for specific text patterns
        search_term = parameters.get('search_term', '')
        if not search_term:
            raise ValueError("search_term parameter is required")
        
        # Case insensitive search
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        matches = pattern.findall(text)
        
        # Find context around matches
        contexts = []
        for match in set(matches):  # unique matches only
            for m in re.finditer(re.escape(match), text, re.IGNORECASE):
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 20)
                contexts.append({
                    "match": match,
                    "context": text[start:end].replace('\n', ' '),
                    "page": "Unknown"  # Could be enhanced to track pages
                })
        
        return {
            "operation": operation,
            "result": {
                "search_term": search_term,
                "total_matches": len(matches),
                "unique_matches": len(set(matches)),
                "sample_contexts": contexts[:5]  # Return first 5 for preview
            }
        }
    
    elif operation == "extract_tables":
        # Attempt to extract tabular data from PDF
        try:
            import pdfplumber  # pip install pdfplumber
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        # Clean table data
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell is not None else '' for cell in row]
                            cleaned_table.append(cleaned_row)
                        
                        if len(cleaned_table) > 1:  # Skip empty tables
                            tables.append({
                                "page": i + 1,
                                "table": cleaned_table,
                                "dimensions": f"{len(cleaned_table)} rows × {len(cleaned_table[0])} cols"
                            })
            
            return {
                "operation": operation,
                "result": {
                    "tables_found": len(tables),
                    "tables": tables[:3]  # Return first 3 tables for preview
                }
            }
        except ImportError:
            return {
                "operation": operation,
                "error": "pdfplumber package required for table extraction"
            }
    
    else:
        raise ValueError(f"Unsupported PDF operation: {operation}")