
import json
from EcommerceInventory.Helpers import CommonListAPIMixin, createParsedCreatedAtUpdatedAt, renderResponse
from DataManagement.models import DataFile, ProcessingOptions
from DataManagement.utils.file_handlers import handle_uploaded_file
from DataManagement.utils.data_analysis import analyze_data
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
from collections import Counter
import os
from django.conf import settings
import time
from django.http import FileResponse

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

    class Meta:
        model = DataFile
        fields = ['id', 'data_name', 'file_name', 'file_url', 'file_type', 'sample_data']

    def get_data_name(self, obj):
        if obj.processing_option:
            return obj.processing_option.data_name
        return None

    def get_file_name(self, obj):
        if obj.file:
            return obj.file.name.split('/')[-1]
        return None

    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None

    def get_sample_data(self, obj):
        if obj.file_type in ['csv', 'xlsx', 'xls']:
            try:
                if obj.file_type == 'csv':
                    df = pd.read_csv(obj.file.path)
                else:
                    df = pd.read_excel(obj.file.path)
                return df.head(5).fillna('').to_dict('records')
            except:
                return None
        return None

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
            if files:
                for file in files:
                    result = handle_uploaded_file(file)
                    if not result['success']:
                        return renderResponse(
                            data={'error': result['error']},
                            message=result['error'],
                            status=400
                        )
                    DataFile.objects.create(
                        processing_option=processing_option,
                        file=file,
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

        # Initialize with basic file info using the serializer
        serializer = DataFileSerializer(file, context={'request': request})
        stats = {
            **serializer.data,
            'validation_score': 0,
            'issues_count': 0,
            'issues': []
        }

        # File type specific analysis
        if file.file_type == 'pdf':
            stats.update(self.analyze_pdf(file))
        else:
            stats.update(self.analyze_tabular(file))

        return Response(stats)

    def analyze_pdf(self, file):
        result = {
            'page_count': 0,
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'text_summary': '',
            'keywords': [],
            'validation_score': 0,
            'issues': []
        }
        
        try:
            reader = PdfReader(file.file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # Basic stats
            words = re.findall(r'\w+', text.lower())
            sentences = re.split(r'[.!?]+', text)
            
            result.update({
                'page_count': len(reader.pages),
                'word_count': len(words),
                'char_count': len(text),
                'sentence_count': len([s for s in sentences if len(s.strip()) > 0]),
                'text_summary': text[:500] + '...' if len(text) > 500 else text,
                'keywords': self.extract_keywords(text),
                'validation_score': self.calculate_pdf_quality(text)
            })
        except Exception as e:
            result['issues'].append({
                'title': 'PDF Processing Error',
                'description': str(e),
                'severity': 'high'
            })
            result['validation_score'] = 30
            
        return result

    def analyze_tabular(self, file):
        result = {
            'record_count': 0,
            'column_count': 0,
            'empty_values': 0,
            'columns': [],
            'validation_score': 0,
            'issues': []
        }
        
        try:
            if file.file_type == 'csv':
                df = pd.read_csv(file.file)
            else:  # Excel
                df = pd.read_excel(file.file)
            
            result.update({
                'record_count': len(df),
                'column_count': len(df.columns),
                'empty_values': df.isnull().sum().sum(),
                'columns': self.analyze_columns(df),
                'validation_score': self.calculate_data_quality(df)
            })
        except Exception as e:
            result['issues'].append({
                'title': 'Data Processing Error',
                'description': str(e),
                'severity': 'high'
            })
            result['validation_score'] = 40
            
        return result

    def extract_keywords(self, text):
        words = re.findall(r'\w+', text.lower())
        stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for'])
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        return [word for word, count in Counter(filtered_words).most_common(10)]

    def calculate_pdf_quality(self, text):
        if len(text) < 100:
            return 30
        word_count = len(re.findall(r'\w+', text))
        unique_words = len(set(re.findall(r'\w+', text.lower())))
        diversity = unique_words / word_count if word_count > 0 else 0
        score = min(100, int(diversity * 100 + 50))
        return max(30, score)

    def analyze_columns(self, df):
        columns = []
        for col in df.columns:
            col_data = {
                'name': col,
                'type': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'null_count': df[col].isnull().sum(),
                'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'distribution': 1 - (df[col].isnull().sum() / len(df)) if len(df) > 0 else 0
            }
            columns.append(col_data)
        return columns

    def calculate_data_quality(self, df):
        if len(df) == 0:
            return 0
        
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        uniqueness = df.apply(lambda x: x.nunique() / len(x)).mean()
        
        validity = 1
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
                validity *= 0.9
        
        consistency = 1
        for col in df.select_dtypes(include='number').columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
            consistency *= 1 - (outlier_count / len(df))
        
        return max(0, min(100, int(100 * (0.4 * completeness + 0.2 * uniqueness + 0.2 * validity + 0.2 * consistency))))

class FileSampleView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, file_id):
        try:
            file = DataFile.objects.get(pk=file_id)
        except DataFile.DoesNotExist:
            return Response({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

        if file.file_type == 'pdf':
            try:
                reader = PdfReader(file.file)
                text = ""
                for i, page in enumerate(reader.pages):
                    if i >= 3:
                        break
                    text += page.extract_text() + "\n\n"
                return Response({
                    'file': DataFileSerializer(file, context={'request': request}).data,
                    'text_sample': text[:2000] + '...' if len(text) > 2000 else text
                })
            except Exception as e:
                return Response({
                    'file': DataFileSerializer(file, context={'request': request}).data,
                    'error': str(e)
                }, status=status.HTTP_400_BAD_REQUEST)
        else:
            try:
                if file.file_type == 'csv':
                    df = pd.read_csv(file.file)
                else:
                    df = pd.read_excel(file.file)
                return Response({
                    'file': DataFileSerializer(file, context={'request': request}).data,
                    'sample_data': df.head(10).fillna('').to_dict('records')
                })
            except Exception as e:
                return Response({
                    'file': DataFileSerializer(file, context={'request': request}).data,
                    'error': str(e)
                }, status=status.HTTP_400_BAD_REQUEST)

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
        """Existing analysis functionality"""
        try:
            uploaded_file = DataFile.objects.get(pk=file_id)
        except DataFile.DoesNotExist:
            return Response({'error': 'File not found'}, status=404)
        
        if uploaded_file.file_type not in ['csv', 'xlsx', 'xls', 'pdf']:
            return Response(
                {'error': 'Data analysis only supported for CSV, Excel and PDF files'},
                status=400
            )
        
        serializer = DataAnalysisSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        
        analysis_result = analyze_data(
            uploaded_file.file.path,
            serializer.validated_data['operation'],
            serializer.validated_data.get('parameters', {})
        )
        
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
            processing_option = datafile.processing_option  # Récupérer le parent avant suppression
            # Assurer la fermeture du fichier avant suppression
            file_path = os.path.join(settings.MEDIA_ROOT, str(datafile.file))
            if os.path.exists(file_path):
                try:
                    # Essayer de fermer explicitement le fichier s'il est encore ouvert
                    with open(file_path, 'r'):
                        pass
                    time.sleep(1)  # Petite pause pour éviter un conflit d'accès
                    os.remove(file_path)
                except Exception as e:
                    return Response({"error": f"Failed to delete file: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Suppression du DataFile
            datafile.delete()
            # Vérifier s'il reste d'autres fichiers liés à ce ProcessingOption
            if not processing_option.data_files.exists():
                processing_option.delete()  # Supprimer le parent s'il n'a plus de DataFiles

            return Response({"message": "Data file and associated processing option deleted successfully"}, status=status.HTTP_204_NO_CONTENT)
        except DataFile.DoesNotExist:
            return Response({"error": "Data file not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)