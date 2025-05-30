
# urls.py
from django.urls import path

from CreditRisk.Controller.RiskManagementController import CreditRiskAPIView, EarlyWarningSystemAPI, MarketRiskAPIView, RiskAnalysisAPI

urlpatterns = [
    path('creditrisk/', CreditRiskAPIView.as_view(), name='credit-risk-assessment'),
    path('marketrisk/', MarketRiskAPIView.as_view(), name='market-risk-assessment'),
    path('ews/', EarlyWarningSystemAPI.as_view(), name='ews-analyze'),
    path('risk-analysis/', RiskAnalysisAPI.as_view(), name='risk-analysis'),
]
