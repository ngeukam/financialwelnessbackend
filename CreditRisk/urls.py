
# urls.py
from django.urls import path

from .Controller.CreditRiskController import CreditRiskAPIView, CreditRiskEvolutionAPIView, CreditRiskListView, CreditRiskRetrieveDestroyView, EarlyWarningSystemAPI
from .Controller.MarketRiskController import MarketRiskAPIView, MarketRiskListView, MarketRiskRetrieveDestroyView
from .Controller.SmartAnalysisController import FinancialRiskAnalysisAPI, FinancialRiskIndicatorListView, FinancialRiskIndicatorRetrieveDestroyView

urlpatterns = [
    
    path('creditrisk-list/', CreditRiskListView.as_view(), name='credit-risk-list'),
    path('creditrisk/retrieve-destroy/<int:id>/', CreditRiskRetrieveDestroyView.as_view(), name='credit-risk-retrieve-destroy'),
    path('creditrisk/', CreditRiskAPIView.as_view(), name='credit-risk-assessment'),
    path('creditrisk/<int:pk>/', CreditRiskAPIView.as_view(), name='credit-risk-assessment-update'),
    
    path('marketrisk-list/', MarketRiskListView.as_view(), name='market-risk-list'),
    path('marketrisk/retrieve-destroy/<int:id>/', MarketRiskRetrieveDestroyView.as_view(), name='market-risk-retrieve-destroy'),
    path('marketrisk/', MarketRiskAPIView.as_view(), name='market-risk-assessment'),
    path('marketrisk/<int:pk>/', MarketRiskAPIView.as_view(), name='market-risk-update'),
    
    path('financialrisk-list/', FinancialRiskIndicatorListView.as_view(), name='financial-risk-list'),
    path('financialrisk/retrieve-destroy/<int:id>/', FinancialRiskIndicatorRetrieveDestroyView.as_view(), name='financial-risk-retrieve-destroy'),
    path('financialrisk/', FinancialRiskAnalysisAPI.as_view(), name='financial-risk'),
    path('financialrisk/<int:pk>/', FinancialRiskAnalysisAPI.as_view(), name='financial-risk-update'),


    path('ews/', EarlyWarningSystemAPI.as_view(), name='ews-analyze'),
    path('creditrisk/evolution/', CreditRiskEvolutionAPIView.as_view(), name='credit-risk-evolution'),
]
