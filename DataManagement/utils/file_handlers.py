import pandas as pd
from PyPDF2 import PdfReader
import csv
import os

def handle_uploaded_file(file):
    file_type = file.name.split('.')[-1].lower()
    data = None
    
    try:
        if file_type == 'pdf':
            data = process_pdf(file)
        elif file_type in ['xlsx', 'xls']:
            data = process_excel(file)
        elif file_type == 'csv':
            data = process_csv(file)
        else:
            raise ValueError("Unsupported file type")
            
        return {'success': True, 'data': data, 'file_type': file_type}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Extract structured data from text (customize as needed)
    data = {
        'text_content': text,
        'page_count': len(reader.pages),
        'metadata': reader.metadata
    }
    return data

def process_excel(file):
    df = pd.read_excel(file)
    return {
        'headers': list(df.columns),
        'data': df.to_dict('records'),
        'shape': df.shape
    }

def process_csv(file):
    # Read as binary and decode to handle different encodings
    content = file.read().decode('utf-8-sig').splitlines()
    reader = csv.DictReader(content)
    data = list(reader)
    
    return {
        'headers': reader.fieldnames,
        'data': data,
        'row_count': len(data)
    }