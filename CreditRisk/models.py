from django.db import models

from UserServices.models import Users

class CreditRiskAssessment(models.Model):
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_risk')
    created_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='created_by_user_id_risk')
    updated_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='updated_by_user_id_risk')
    income = models.FloatField()
    loan_amount = models.FloatField()
    started_period = models.DateField(null=True, blank=True)
    ended_period = models.DateField(null=True, blank=True)
    credit_score = models.IntegerField()
    risk_score = models.FloatField(null=True, blank=True)
    loan_duration_months = models.PositiveIntegerField(null=True, blank=True)
    prediction = models.BooleanField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Assessment {self.id}"


class MarketRiskAssessment(models.Model):
    domain_user_id = models.ForeignKey(Users, on_delete=models.CASCADE, blank=True, null=True, related_name='market_risk_assessments')
    created_by_user_id = models.ForeignKey(Users, on_delete=models.CASCADE, blank=True, null=True, related_name='created_market_risks')
    updated_by_user_id = models.ForeignKey(Users, on_delete=models.CASCADE, blank=True, null=True, related_name='updated_market_risks')
    started_period = models.DateField(null=True, blank=True)
    ended_period = models.DateField(null=True, blank=True)
    # Market data inputs
    stock_prices = models.JSONField()  # Array of historical prices
    exchange_rates = models.JSONField(null=True, blank=True)
    interest_rates = models.JSONField(null=True, blank=True)
    
    # Calculated metrics
    value_at_risk = models.FloatField(null=True, blank=True)
    expected_shortfall = models.FloatField(null=True, blank=True)
    volatility = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Market Risk Assessment {self.id}"

class EarlyWarningIndicator(models.Model):
    INDICATOR_TYPES = [
        ('SPENDING', 'Spending Pattern'),
        ('TRANSACTION', 'Transaction Pattern'),
        ('STOCK', 'Stock Movement'),
        ('LOAN', 'Loan Default Risk'),
        ('NEWS', 'News Sentiment')
    ]
    
    user = models.ForeignKey(Users, on_delete=models.CASCADE, related_name='ews_indicators')
    indicator_type = models.CharField(max_length=20, choices=INDICATOR_TYPES)
    value = models.JSONField()
    started_period = models.DateField(null=True, blank=True)
    ended_period = models.DateField(null=True, blank=True)
    threshold = models.JSONField(default=dict)
    is_anomaly = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.get_indicator_type_display()} Alert - {'Anomaly' if self.is_anomaly else 'Normal'}"

class FinancialIndicator(models.Model):
    RISK_LEVELS = [
        (0, 'Low Risk'),
        (1, 'Medium Risk'),
        (2, 'High Risk')
    ]
    
    started_period = models.DateField(null=True, blank=True)
    ended_period = models.DateField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    gdp_growth = models.FloatField()
    inflation = models.FloatField()
    interest_rate = models.FloatField()
    market_volatility = models.FloatField()
    predicted_risk = models.IntegerField(choices=RISK_LEVELS)
    confidence_score = models.FloatField()
    metadata = models.JSONField(default=dict)
    domain_user_id = models.ForeignKey(Users, on_delete=models.CASCADE, blank=True, null=True, related_name='financial_risk_assessments')
    created_by_user_id = models.ForeignKey(Users, on_delete=models.CASCADE, blank=True, null=True, related_name='created_financial_risks')
    updated_by_user_id = models.ForeignKey(Users, on_delete=models.CASCADE, blank=True, null=True, related_name='updated_financial_risks')
  

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.email} - {self.get_predicted_risk_display()} ({self.timestamp})"
