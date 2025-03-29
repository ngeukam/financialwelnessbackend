# urls.py
from django.urls import path
from .Controller.DataManagementController import DataAnalysisAPIView, DataFileDeleteView, DataFileListView, FileListView, FileSampleView, FileStatsView, ProcessAndUploadAPIView, ImportHistoryFilesListView

urlpatterns = [
    path('process-and-upload/', ProcessAndUploadAPIView.as_view(), name='process_and_upload'),
    path('analyze/<int:file_id>/', DataAnalysisAPIView.as_view(), name='data_analyze'),
    path('files/', FileListView.as_view(), name='files_list'),
    path('files/<int:file_id>/', FileListView.as_view(), name='get_file'),
    path('files/<int:file_id>/stats/', FileStatsView.as_view(), name='files_stats'),
    path('files/<int:file_id>/sample/', FileSampleView.as_view(), name='files_sample'),
    path('upload-files-history/', ImportHistoryFilesListView.as_view(), name='upload-files-history'),
    path('detele/<int:pk>/',DataFileDeleteView.as_view(),name='data_file_delete'),
    path('data-files/',DataFileListView.as_view(),name='data_files'),
    path('data-analysis/<int:file_id>/',DataAnalysisAPIView.as_view(),name='data_analysis'),


    






]