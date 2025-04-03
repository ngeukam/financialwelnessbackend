
import json
from EcommerceInventory.Helpers import CommonListAPIMixin, createParsedCreatedAtUpdatedAt, renderResponse
from DataManagement.models import DataFile, ProcessingOptions
from DataManagement.utils.file_handlers import handle_uploaded_file
from DataManagement.utils.data_analysis import UniversalJSONAnalyzer
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db import transaction
import pandas as pd
from PyPDF2 import PdfReader
import re
from collections import Counter, defaultdict
import os
from django.conf import settings
import time
from django.http import FileResponse
import pdfplumber
import numpy as np
import datetime

@createParsedCreatedAtUpdatedAt
class ProcessingOptionsSerializer(serializers.ModelSerializer):
    domain_user_id = serializers.SerializerMethodField()
    added_by_user_id = serializers.SerializerMethodField()

    class Meta:
        model = ProcessingOptions
        fields = '__all__'
        extra_kwargs = {
            'delete_duplicate': {'required': False, 'default': True},
            'merge_existing': {'required': False, 'default': False},
        }


    def get_domain_user_id(self, obj):
        if obj.domain_user_id:
            return f"#{obj.domain_user_id.id} {obj.domain_user_id.username}"
        return None
    
    def get_added_by_user_id(self, obj):
        if obj.added_by_user_id:
            return f"#{obj.added_by_user_id.id} {obj.added_by_user_id.username}"
        return None

    def validate_processing_options(self, value):
        """Ensure processing_options is valid JSON"""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise serializers.ValidationError('Value must be valid JSON.')
        return value

class DataFileDisplaySerializer(serializers.ModelSerializer):
    data_name=serializers.SerializerMethodField()
    file_name = serializers.SerializerMethodField()
    file_size = serializers.SerializerMethodField()

    class Meta:
        model= DataFile
        fields = ['data_name', 'file_name', 'file', 'file_type', 'file_size', 'uploaded_at', 'id']
    
    def get_data_name(self, obj):
        if obj.processing_option:
            return f"{obj.processing_option.data_name}"
        
    def get_file_name(self, obj):
        if obj.file:
            return obj.file.name.split('/')[-1]
        return None

    def get_file_size(self, obj):
        if obj.file:
            size = obj.file.size
            for unit in ['bytes', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        return None

class DataFileSerializer(serializers.ModelSerializer):
    processing_option_display = serializers.SerializerMethodField()
    file_url = serializers.SerializerMethodField()
    file_name = serializers.SerializerMethodField()
    file_size = serializers.SerializerMethodField()

    class Meta:
        model = DataFile
        fields = [
            'id',
            'processing_option',
            'processing_option_display',
            'file',
            'processed_data',
            'file_url',
            'file_name',
            'file_size',
            'file_type',
            'uploaded_at',
        ]
        read_only_fields = ['uploaded_at', 'file_type']

    def get_processing_option_display(self, obj):
        if obj.processing_option:
            return f"#{obj.processing_option.id} {obj.processing_option.data_name}"
        return None

    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None

    def get_file_name(self, obj):
        if obj.file:
            return obj.file.name.split('/')[-1]
        return None

    def get_file_size(self, obj):
        if obj.file:
            size = obj.file.size
            for unit in ['bytes', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        return None

class DataFileDisplay2Serializer(serializers.ModelSerializer):
    data_name = serializers.SerializerMethodField()
    file_name = serializers.SerializerMethodField()
    file_url = serializers.SerializerMethodField()
    sample_data = serializers.SerializerMethodField()
    table_count = serializers.SerializerMethodField()  # New field for PDF table count

    class Meta:
        model = DataFile
        fields = ['id', 'data_name', 'file_name', 'file_url', 'file_type', 
                 'sample_data', 'table_count']  # Added table_count

    def get_data_name(self, obj):
        return obj.processing_option.data_name if obj.processing_option else None

    def get_file_name(self, obj):
        return obj.file.name.split('/')[-1] if obj.file else None

    def get_file_url(self, obj):
        request = self.context.get('request')
        return request.build_absolute_uri(obj.file.url) if obj.file and request else None

    def get_table_count(self, obj):
        if obj.file_type == 'pdf':
            return self._extract_pdf_tables(obj, count_only=True)
        return None

    def get_sample_data(self, obj):
        try:
            if obj.file_type == 'csv':
                df = pd.read_csv(obj.file.path)
                return self._clean_sample_data(df)
            
            elif obj.file_type in ['xlsx', 'xls']:
                # Try different Excel engines
                try:
                    df = pd.read_excel(obj.file.path, engine='openpyxl')
                except:
                    df = pd.read_excel(obj.file.path, engine='xlrd')
                return self._clean_sample_data(df)
            
            elif obj.file_type == 'pdf':
                tables = self._extract_pdf_tables(obj)
                if tables:
                    # Return sample from first table
                    return self._clean_sample_data(tables[0])
                return None
            
            return None
        except Exception as e:
            print(f"Error getting sample data: {str(e)}")
            return None

    def _extract_pdf_tables(self, obj, count_only=False):
        """Helper method to extract tables from PDF"""
        try:
            with pdfplumber.open(obj.file.path) as pdf:
                tables = []
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:  # Needs header and at least one row
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
                
                if count_only:
                    return len(tables)
                
                return tables if tables else None
                
        except Exception as e:
            print(f"PDF table extraction failed: {str(e)}")
            return 0 if count_only else None
        
    def _clean_sample_data(self, df):
        """Clean and format sample data for display with explicit error handling"""
        # Basic cleaning
        df = df.dropna(how='all').dropna(how='all', axis=1)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Convert numeric columns and handle potential text numbers
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Clean numeric strings (remove spaces, currency symbols, etc.)
                cleaned = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                
                try:
                    # Attempt conversion without errors='ignore'
                    converted = pd.to_numeric(cleaned)
                    
                    # Only update if conversion was fully successful
                    if not converted.isna().all():  # Check if we got at least some valid numbers
                        df[col] = converted
                        
                except (ValueError, TypeError) as e:
                    # Handle specific exceptions that to_numeric might raise
                    print(f"Could not convert column '{col}': {str(e)}")
                    continue
                except Exception as e:
                    print(f"Unexpected error converting column '{col}': {str(e)}")
                    continue
        
        # Return first 5 rows with proper NaN handling
        return df.head(5).replace([np.nan, None], '').to_dict('records')

class DataAnalysisSerializer(serializers.Serializer):
    operation = serializers.CharField(max_length=100)
    parameters = serializers.JSONField(required=False)

class ProcessAndUploadAPIView(generics.CreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    serializer_class = ProcessingOptionsSerializer

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        try:
            data = request.data.copy()
            
            # Créez ProcessingOptions
            serializer = self.get_serializer(data=data)
            serializer.is_valid(raise_exception=True)
            
            processing_option = serializer.save(
                domain_user_id=request.user,
                added_by_user_id=request.user
            )

            # Gérez les fichiers uploadés
            files = request.FILES.getlist('data_files')
            # 3. Process optional parameters
            # rows_to_remove = request.data.get('rows_to_remove')
            min_empty_values = self._clean_optional_param(request.data.get('min_empty_values'), 'minEmptyValues')
            columns_to_remove = request.data.get('columns_to_remove')
            if columns_to_remove and isinstance(columns_to_remove, str):
                columns_to_remove = [col.strip() for col in columns_to_remove.split(',')]
            if files:
                for file in files:
                    result = handle_uploaded_file(file, columns_to_remove=columns_to_remove, min_empty_values=min_empty_values)
                    if not result['success']:
                        return renderResponse(
                            data={'error': result['error']},
                            message=result['error'],
                            status=400
                        )
                    # print(result)
                    processed_data = result['data']
                    if not self._is_valid_json(processed_data):
                        processed_data = self._convert_to_valid_json(processed_data)
                    DataFile.objects.create(
                        processing_option=processing_option,
                        file=file,
                        processed_data=processed_data,
                        file_type=result['file_type']
                    )

            # response_data = serializer.data
            response_data = {
            'file': serializer.data,
            'processed_data': result['data']
        }
            if files:
                response_data['uploaded_files'] = [
                    {'name': f.name, 'size': f.size} 
                    for f in files
                ]

            return renderResponse(
                data=response_data,
                message='Data file.s upload successfully!',
                status=201
            )

        except Exception as e:
            return renderResponse(
                data={'error': str(e)},
                message=str(e),
                status=400
            )

    def _clean_optional_param(self, value, param_name):
        """Convert optional parameters to None if 0 or invalid"""
        try:
            if value is not None:
                num_value = int(value)
                return None if num_value == 0 else num_value
            return None
        except (ValueError, TypeError):
            raise serializers.ValidationError(
                {param_name: "Must be a valid integer"}
            )
    def _is_valid_json(self, data):
        """Check if data is JSON serializable"""
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError):
            return False

    def _convert_to_valid_json(self, data):
        """Convert data to JSON serializable format"""
        if isinstance(data, pd.DataFrame):
            return data.replace({np.nan: None}).to_dict('records')
        if isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        if isinstance(data, (datetime.date, datetime.datetime)):
            return data.isoformat()
        
        try:
            return json.loads(json.dumps(data, default=str))
        except:
            return str(data)
        
class FileListView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, file_id=None):
        # Get queryset with select_related to optimize database queries
        if(file_id):
             files = DataFile.objects.select_related(
            'processing_option', 
        ).filter(pk=file_id).filter(processing_option__domain_user_id_id=self.request.user.domain_user_id.id).order_by('-uploaded_at')[:5]
        else:
            files = DataFile.objects.select_related(
                'processing_option', 
            ).filter(processing_option__domain_user_id_id=self.request.user.domain_user_id.id).order_by('-uploaded_at')[:5]
        
        # Use the serializer with context
        serializer = DataFileSerializer(
            files, 
            many=True,
            context={'request': request}
        )
        
        return Response(serializer.data)

class FileStatsView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, file_id):
        try:
            file = DataFile.objects.get(pk=file_id)
        except DataFile.DoesNotExist:
            return Response({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

        # Initialize with basic file info
        serializer = DataFileSerializer(file, context={'request': request})
        stats = {
            **serializer.data,
            'analysis': {
                'validation_score': 0,
                'issues_count': 0,
                'issues': [],
                'stats': {}
            }
        }

        # Process the JSON data if it exists
        if file.processed_data:
            analysis_result = self.analyze_json_data(file.processed_data)
            stats['analysis'].update(analysis_result)
            stats['analysis']['issues_count'] = len(stats['analysis']['issues'])
            
        return Response(stats)

    def analyze_json_data(self, json_data):
        """Analyze the JSON structure and content"""
        result = {
            'record_count': 0,
            'column_count': 0,
            'empty_values': 0,
            'columns': [],
            'validation_score': 0,
            'issues': [],
            'stats': {}
        }

        try:
            # Handle different JSON structures
            if isinstance(json_data, dict) and 'tables' in json_data:
                all_columns = defaultdict(list)
                # Process multi-table structure
                for table in json_data['tables']:
                    table_stats = self.analyze_table(table)
                    result['stats'][f"table_{table.get('table_id', 'unknown')}"] = table_stats
                    result['record_count'] += table_stats.get('record_count', 0)
                    result['column_count'] = max(result['column_count'], table_stats.get('column_count', 0))
                    result['empty_values'] += table_stats.get('empty_values', 0)
                    result['issues'].extend(table_stats.get('issues', []))
                    # Aggregate column information across tables
                    for col in table_stats.get('columns', []):
                        all_columns[col['name']].append(col)
                # Include all unique columns in the result
                result['columns'] = [self.merge_column_stats(stats) for stats in all_columns.values()]
                if json_data['tables']:
                    result['validation_score'] = self.calculate_validation_score(result)
            
            elif isinstance(json_data, list):
                # Process single table as list of records
                df = pd.DataFrame(json_data)
                table_stats = self.analyze_dataframe(df)
                result.update(table_stats)
                result['validation_score'] = self.calculate_validation_score(result)
                
            elif isinstance(json_data, dict):
                # Process single record
                df = pd.DataFrame([json_data])
                table_stats = self.analyze_dataframe(df)
                result.update(table_stats)
                result['validation_score'] = self.calculate_validation_score(result)
                
        except Exception as e:
            result['issues'].append({
                'type': 'analysis_error',
                'message': str(e),
                'severity': 'high'
            })

        return result
    
    def merge_column_stats(self, column_instances):
        """Merge statistics from multiple instances of the same column"""
        if not column_instances:
            return {}
        
        base_stats = column_instances[0].copy()
        
        if len(column_instances) > 1:
            base_stats['appears_in_tables'] = len(column_instances)
            
            # Merge numeric stats if available
            if all('numeric_stats' in c for c in column_instances):
                base_stats['numeric_stats'] = {
                    'combined_min': min(c['numeric_stats']['min'] for c in column_instances),
                    'combined_max': max(c['numeric_stats']['max'] for c in column_instances),
                    'combined_mean': sum(
                        c['numeric_stats']['mean'] * c['record_count'] 
                        for c in column_instances
                    ) / sum(c['record_count'] for c in column_instances),
                }
        
        return base_stats
    
    def analyze_table(self, table_data):
        """Analyze a single table from processed_data"""
        result = {
            'record_count': 0,
            'column_count': 0,
            'empty_values': 0,
            'columns': [],
            'issues': []
        }

        try:
            if 'data' in table_data and isinstance(table_data['data'], list):
                df = pd.DataFrame(table_data['data'])
                table_stats = self.analyze_dataframe(df)
                result.update(table_stats)
                
                # Add table-specific issues
                if 'headers' in table_data and len(table_data['headers']) != result['column_count']:
                    result['issues'].append({
                        'type': 'header_mismatch',
                        'message': f"Header count ({len(table_data['headers'])}) doesn't match data columns ({result['column_count']})",
                        'severity': 'medium'
                    })
                    
        except Exception as e:
            result['issues'].append({
                'type': 'table_analysis_error',
                'message': str(e),
                'severity': 'high'
            })

        return result

    def analyze_dataframe(self, df):
        """Analyze a pandas DataFrame"""
        result = {
            'record_count': len(df),
            'column_count': len(df.columns),
            'empty_values': int(df.isnull().sum().sum()),
            'columns': [],
            'issues': []
        }

        # Analyze each column
        for col in df.columns:
            col_stats = {
                'name': str(col),
                'type': str(df[col].dtype),
                'unique_values': int(df[col].nunique()),
                'empty_values': int(df[col].isnull().sum()),
                'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'sum': df[col].sum() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'distribution': 1 - (df[col].isnull().sum() / len(df)) if len(df) > 0 else 0,
                'sample_data': self.get_sample_values(df[col])
            }

            # Detect potential issues
            if col_stats['empty_values'] > 0:
                result['issues'].append({
                    'type': 'missing_values',
                    'column': str(col),
                    'count': col_stats['empty_values'],
                    'severity': 'low'
                })

            # Check for mixed data types
            if df[col].apply(type).nunique() > 1:
                result['issues'].append({
                    'type': 'mixed_types',
                    'column': str(col),
                    'severity': 'medium'
                })

            result['columns'].append(col_stats)

        return result

    def calculate_validation_score(self, result):
        """Calculate an overall data quality score (0-100)"""
        max_score = 100
        deductions = 0
        
        # Deduct for issues
        for issue in result.get('issues', []):
            if issue['severity'] == 'high':
                deductions += 5
            elif issue['severity'] == 'medium':
                deductions += 2
            else:
                deductions += 1
                
        # Deduct for empty values
        if result['record_count'] > 0:
            empty_percentage = result['empty_values'] / (result['record_count'] * result['column_count'])
            deductions += min(30, empty_percentage * 100)
            
        return max(0, max_score - deductions)

    def get_sample_values(self, series):
        """Get representative sample values from a series"""
        try:
            # Get non-null values
            values = series.dropna().unique()
            
            # Convert to native Python types
            samples = []
            for v in values[:5]:  # Return up to 5 sample values
                if pd.api.types.is_datetime64_any_dtype(series):
                    samples.append(v.strftime('%Y-%m-%d'))
                elif pd.api.types.is_numeric_dtype(series):
                    samples.append(float(v))
                else:
                    samples.append(str(v))
            return samples
        except:
            return []
        
class FileSampleView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, file_id):
        try:
            file = DataFile.objects.get(pk=file_id)
        except DataFile.DoesNotExist:
            return Response({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

        # Initialize response with basic file info
        response_data = {
            'file': DataFileSerializer(file, context={'request': request}).data,
            'sample_data': None,
            'structure_type': None,
            'metadata': {
                'has_processed_data': file.processed_data is not None
            }
        }

        if not file.processed_data:
            return Response({
                **response_data,
                'warning': 'No processed data available in JSONField'
            }, status=status.HTTP_200_OK)

        try:
            # Analyze and extract sample data from JSONField
            sample_data, structure_type = self.extract_sample_data(file.processed_data)
            
            return Response({
                **response_data,
                'sample_data': sample_data,
                'structure_type': structure_type,
                'metadata': {
                    **response_data['metadata'],
                    'record_count': self.count_records(file.processed_data),
                    'table_count': self.count_tables(file.processed_data)
                }
            })
            
        except Exception as e:
            return Response({
                **response_data,
                'error': f'Failed to extract sample data: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)

    def extract_sample_data(self, json_data):
        """Extract representative sample data from JSON structure"""
        if isinstance(json_data, dict) and 'tables' in json_data:
            # Multi-table structure (like bank statements)
            sample = {
                'structure_type': 'multi_table',
                'tables': []
            }
            for table in json_data['tables'][:2]:  # Sample first 2 tables max
                if 'data' in table and isinstance(table['data'], list):
                    sample['tables'].append({
                        'table_id': table.get('table_id'),
                        'headers': table.get('headers', []),
                        'sample_rows': table['data'][:10]  # First 5 rows per table
                    })
            return sample, 'multi_table'
        
        elif isinstance(json_data, list):
            # List of records
            return {
                'structure_type': 'record_list',
                'sample_records': json_data[:10]  # First 10 records
            }, 'record_list'
        
        elif isinstance(json_data, dict):
            # Single complex object
            return {
                'structure_type': 'single_object',
                'object_sample': json_data
            }, 'single_object'
        
        else:
            # Fallback for other JSON types
            return {
                'structure_type': 'raw_data',
                'data_sample': str(json_data)[:500]  # Truncate if too large
            }, 'raw_data'

    def count_records(self, json_data):
        """Count total records across all structures"""
        if isinstance(json_data, dict) and 'tables' in json_data:
            return sum(len(table.get('data', [])) for table in json_data['tables'])
        elif isinstance(json_data, list):
            return len(json_data)
        return 1  # Single object counts as 1 record

    def count_tables(self, json_data):
        """Count tables in the structure"""
        if isinstance(json_data, dict) and 'tables' in json_data:
            return len(json_data['tables'])
        return 0

class ImportHistoryFilesListView(generics.ListAPIView):
    serializer_class = DataFileDisplaySerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset=DataFile.objects.filter(processing_option__domain_user_id_id=self.request.user.domain_user_id.id)
        return queryset

    @CommonListAPIMixin.common_list_decorator(DataFileDisplaySerializer)
    def list(self,request,*args,**kwargs):
        return super().list(request,*args,**kwargs)   

class DataAnalysisAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, file_id, format=None):
        """Perform data analysis using UniversalJSONAnalyzer on stored JSON data."""
        try:
            uploaded_file = DataFile.objects.get(pk=file_id)
        except DataFile.DoesNotExist:
            return Response({'error': 'File not found'}, status=404)

        if not uploaded_file.processed_data:
            return Response({'error': 'No processed data available for analysis'}, status=400)

        serializer = DataAnalysisSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        operation = serializer.validated_data['operation']
        parameters = serializer.validated_data.get('parameters', {})
        # Get JSON data from processed_data field
        json_data = uploaded_file.processed_data
        # Perform analysis using UniversalJSONAnalyzer
        analyzer = UniversalJSONAnalyzer()
        try:
            analysis_result = analyzer.analyze(json_data, operation, parameters)
        except ValueError as ve:
            return Response({'error': str(ve)}, status=400)
        return Response({
            'file': DataFileDisplaySerializer(uploaded_file, context={'request': request}).data,
            'analysis': analysis_result
        }, status=200)

class DataFileListView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        files = DataFile.objects.select_related('processing_option').filter(processing_option__domain_user_id_id=self.request.user.domain_user_id.id).order_by('-uploaded_at')
        serializer = DataFileDisplay2Serializer(files, many=True, context={'request': request})
        return Response(serializer.data)
                     
class DataFileDeleteView(generics.DestroyAPIView):
    queryset = DataFile.objects.all()
    serializer_class = DataFileSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, *args, **kwargs):
        try:
            datafile = self.get_object()

            # Get parent ProcessingOption safely
            processing_option = getattr(datafile, 'processing_option', None)

            # Ensure file path exists
            file_path = os.path.join(settings.MEDIA_ROOT, str(datafile.file)) if datafile.file else None

            if file_path and os.path.exists(file_path):
                try:
                    # Ensure the file is closed before deleting
                    with open(file_path, 'r') as f:
                        pass
                    time.sleep(1)  # Prevent race conditions
                    os.remove(file_path)
                except Exception as e:
                    return Response(
                        {"error": f"Failed to delete file: {str(e)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

            # Delete DataFile entry
            datafile.delete()

            # Delete processing option if it has no more associated files
            if processing_option and not processing_option.data_files.exists():
                processing_option.delete()

            return Response(
                {"message": "Data file and associated processing option deleted successfully"},
                status=status.HTTP_204_NO_CONTENT
            )

        except DataFile.DoesNotExist:
            return Response(
                {"error": "Data file not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"Unexpected error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )