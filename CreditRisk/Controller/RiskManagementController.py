from rest_framework import serializers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
import numpy as np
import logging
from django.utils import timezone
from CreditRisk.Helper.ml_service import RiskPredictor
from CreditRisk.models import CreditRiskAssessment, EarlyWarningIndicator, FinancialIndicator, MarketRiskAssessment
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

class CreditRiskAssessmentSerializer(serializers.ModelSerializer):
    income = serializers.FloatField(min_value=0)
    loan_amount = serializers.FloatField(min_value=0)
    credit_score = serializers.IntegerField(min_value=300, max_value=850)
    
    class Meta:
        model = CreditRiskAssessment
        fields = ['income', 'loan_amount', 'credit_score']
        extra_kwargs = {
            'income': {'required': True},
            'loan_amount': {'required': True},
            'credit_score': {'required': True}
        }

class MarketRiskAssessmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketRiskAssessment
        fields = ['stock_prices', 'exchange_rates', 'interest_rates']
        extra_kwargs = {
            'stock_prices': {'required': True},
            'exchange_rates': {'required': False},
            'interest_rates': {'required': False}
        }
        
class EarlyWarningIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = EarlyWarningIndicator
        fields = ['id', 'indicator_type', 'value', 'threshold', 'is_anomaly', 'timestamp', 'metadata']
    
    def to_representation(self, instance):
        """Convert the JSON fields to proper format in API response"""
        representation = super().to_representation(instance)
        # Ensure value is always returned as array for consistency
        if not isinstance(representation['value'], list):
            representation['value'] = [representation['value']]
        return representation

class FinancialIndicatorSerializer(serializers.ModelSerializer):
    risk_level_display = serializers.CharField(source='get_predicted_risk_display', read_only=True)
    
    class Meta:
        model = FinancialIndicator
        fields = [
            'id',
            'timestamp',
            'gdp_growth',
            'inflation',
            'interest_rate',
            'market_volatility',
            'predicted_risk',
            'risk_level_display',
            'confidence_score',
            'metadata'
        ]
         
class CreditRiskAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    # Sample training data - replace with your actual data source
    def _get_training_data(self):
            """Fetch actual historical data from your database"""
            # Get approved assessments with known outcomes (defaults)
            historical_data = CreditRiskAssessment.objects.exclude(
                prediction__isnull=True
            ).values_list(
                'income',
                'loan_amount',
                'credit_score',
                'prediction'
            )
            print(historical_data)
            # Convert to pandas DataFrame
            if historical_data:
                df = pd.DataFrame(list(historical_data), 
                                columns=['income', 'loan_amount', 'credit_score', 'default'])
                return df
            else:
                # Fallback to sample data if no historical data exists
                return pd.DataFrame({
                    'income': [50000, 60000, 45000, 80000, 55000, 75000, 90000, 48000, 65000, 70000],
                    'loan_amount': [15000, 20000, 10000, 25000, 18000, 22000, 30000, 12000, 21000, 24000],
                    'credit_score': [700, 650, 720, 680, 710, 690, 730, 675, 705, 715],
                    'default': [0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
                })
    
    def __init__(self):
        super().__init__()
        self.model = self._train_model()
    
    def _train_model(self):
        """Train the risk assessment model with actual data"""
        df = self._get_training_data()
        X = df[['income', 'loan_amount', 'credit_score']]
        y = df['default']
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        return model
    
    def post(self, request):
        """Handle credit risk assessment requests"""
        serializer = CreditRiskAssessmentSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            
            try:
                # Prepare input data with proper feature names
                input_df = pd.DataFrame([[data['income'], data['loan_amount'], data['credit_score']]],
                                      columns=['income', 'loan_amount', 'credit_score'])
                
                # Make prediction
                prediction = self.model.predict(input_df)[0]
                
                # Calculate risk score (customize this formula as needed)
                risk_score = (data['income'] / data['loan_amount']) * (data['credit_score'] / 700)
                
                # Create assessment record with user associations
                assessment = CreditRiskAssessment(
                    domain_user_id=request.user,  # The user being assessed
                    created_by_user_id=request.user,  # The user creating the assessment
                    updated_by_user_id=request.user,  # The user updating (same as creator for new records)
                    income=data['income'],
                    loan_amount=data['loan_amount'],
                    credit_score=data['credit_score'],
                    risk_score=risk_score,
                    prediction=bool(prediction))
                assessment.save()
                
                # Prepare response data
                response_data = {
                    'risk_score': risk_score,
                    'prediction': prediction,
                    'assessment_id': assessment.id,
                    'created_at': assessment.created_at,
                    'message':'Save successfully!'
                }
                
                return Response(response_data, status=status.HTTP_201_CREATED)
            
            except Exception as e:
                return Response(
                    {"error": "Assessment failed", "details": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class MarketRiskAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def validate_input_data(self, data):
        """Validate input data meets minimum requirements"""
        if len(data.get('stock_prices', [])) < 2:
            raise ValueError("At least 2 stock prices are required to calculate returns")
        
        # Optional: Add validation for exchange_rates and interest_rates if needed
        return True
    
    def calculate_returns(self, prices):
        """Calculate logarithmic returns from price series"""
        prices = np.array(prices)
        return np.diff(np.log(prices))
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index])  # Return absolute value for risk magnitude
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """Calculate Expected Shortfall (CVaR)"""
        if len(returns) == 0:
            return 0
        var = self.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= -abs(var)]  # Consider only losses
        return abs(tail_returns.mean()) if len(tail_returns) > 0 else 0
    
    def calculate_volatility(self, returns):
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0
        return np.std(returns) * np.sqrt(252)  # 252 trading days
    
    def post(self, request):
        serializer = MarketRiskAssessmentSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            
            try:
                # Validate input data
                self.validate_input_data(data)
                
                # Calculate returns from stock prices
                stock_returns = self.calculate_returns(data['stock_prices'])
                
                # Calculate risk metrics (absolute values for risk measures)
                var_95 = self.calculate_var(stock_returns)
                expected_shortfall = self.calculate_expected_shortfall(stock_returns)
                volatility = self.calculate_volatility(stock_returns)
                
                # Create assessment record
                assessment = MarketRiskAssessment(
                    domain_user_id=request.user,
                    created_by_user_id=request.user,
                    updated_by_user_id=request.user,
                    stock_prices=data['stock_prices'],
                    exchange_rates=data.get('exchange_rates', []),
                    interest_rates=data.get('interest_rates', []),
                    value_at_risk=var_95,
                    expected_shortfall=expected_shortfall,
                    volatility=volatility
                )
                assessment.save()
                
                # Prepare response with formatted values
                response_data = {
                    'value_at_risk': round(var_95, 6),
                    'expected_shortfall': round(expected_shortfall, 6),
                    'volatility': round(volatility, 6),
                    'assessment_id': assessment.id,
                    'created_at': assessment.created_at,
                    'message': 'Risk assessment completed successfully'
                }
                
                return Response(response_data, status=status.HTTP_201_CREATED)
            
            except ValueError as e:
                return Response(
                    {"error": "Invalid input data", "details": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )
            except Exception as e:
                return Response(
                    {"error": "Market risk assessment failed", "details": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

logger = logging.getLogger(__name__)
class EarlyWarningSystemAPI(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    DEFAULT_CONTAMINATION = 0.1
    
    def detect_anomalies(self, data, contamination=None):
        """Enhanced anomaly detection with configurable contamination"""
        contamination = contamination or self.DEFAULT_CONTAMINATION
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(data)
        return model.predict(data)
    
    def calculate_financial_metrics(self, income, expenses):
        """Calculate additional financial health metrics"""
        return {
            'savings_rate': (income - expenses) / income if income > 0 else 0,
            'expense_ratio': expenses / income if income > 0 else float('inf'),
            'disposable_income': income - expenses
        }
    
    def validate_input(self, indicator_type, values, thresholds):
        """Validate all input parameters"""
        if not indicator_type or indicator_type not in dict(EarlyWarningIndicator.INDICATOR_TYPES):
            return False, "Invalid indicator type"
        
        if not isinstance(values, list) or len(values) == 0:
            return False, "Values must be a non-empty array"
            
        if indicator_type == 'SPENDING':
            if any(not isinstance(item, list) or len(item) != 2 for item in values):
                return False, "Spending data must be arrays of [income, expenses] pairs"
        
        if thresholds and not isinstance(thresholds, dict):
            return False, "Thresholds must be a dictionary"
            
        return True, None

    def analyze_spending_pattern(self, spending_data, contamination=None):
        data = np.array(spending_data)
        
        # 1. Apply absolute business rules
        biz_rules_violated = np.array([
            -1 if income < expenses else 1 
            for income, expenses in data
        ])
        
        # 2. Prepare enhanced features for ML
        processed_data = np.array([
            [income, expenses, expenses/max(income,1), (income-expenses)/max(income,1)]
            for income, expenses in data
        ])
        
        # 3. Dynamic contamination
        base_contamination = contamination or self.DEFAULT_CONTAMINATION
        biz_anomaly_ratio = sum(biz_rules_violated == -1)/len(biz_rules_violated)
        final_contamination = min(0.5, base_contamination + biz_anomaly_ratio)
        
        # 4. ML detection
        ml_anomalies = self.detect_anomalies(processed_data, final_contamination)
        
        # 5. Combine results (business rules take precedence)
        combined_anomalies = np.where(biz_rules_violated == -1, -1, ml_anomalies)
        
        return combined_anomalies, final_contamination
    
    def post(self, request):
        indicator_type = request.data.get('indicator_type')
        values = request.data.get('values', [])
        thresholds = request.data.get('thresholds', {})
        contamination = request.data.get('contamination')
        if contamination is not None:
            contamination = float(contamination)
        else:
            contamination = self.DEFAULT_CONTAMINATION
        
        # Input validation
        is_valid, error_msg = self.validate_input(indicator_type, values, thresholds)
        if not is_valid:
            return Response(
                {"error": error_msg},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            anomalies = []
            metrics = []
            final_contamination = contamination
            debug_info = {}
            
            if indicator_type == 'SPENDING':
                anomalies, final_contamination = self.analyze_spending_pattern(values, contamination)
                metrics = [self.calculate_financial_metrics(inc, exp) for inc, exp in values]
                
                # Debug information
                debug_info = {
                    'business_rules': [
                        'expenses > income' 
                        for income, expenses in values 
                        if income < expenses
                    ],
                    'contamination_used': final_contamination,
                    'input_values': values
                }
            
            results = []
            for i, value in enumerate(values):
                metadata = {
                    'analysis_method': 'IsolationForest',
                    'contamination': final_contamination,
                    'full_data': value,
                    'metrics': metrics[i] if metrics else None
                }
                
                indicator = EarlyWarningIndicator(
                    user=request.user,
                    indicator_type=indicator_type,
                    value=value,
                    threshold=thresholds.get(str(i), 0),
                    is_anomaly=anomalies[i] == -1 if i < len(anomalies) else False,
                    metadata=metadata
                )
                indicator.save()
                results.append(indicator)
            
            response_data = {
                'results': EarlyWarningIndicatorSerializer(results, many=True).data,
                'summary': {
                    'total_anomalies': sum(1 for a in anomalies if a == -1),
                    'anomaly_indices': [i for i, a in enumerate(anomalies) if a == -1],
                    'contamination_used': final_contamination
                },
                'metadata': {
                    'model': 'IsolationForest',
                    'analysis_date': timezone.now().isoformat(),
                    'debug': debug_info
                }
            }
            
            return Response(response_data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(
                f"EWS analysis failed for user {request.user.id}",
                exc_info=True,
                extra={'request': request}
            )
            return Response(
                {
                    "error": "EWS analysis failed",
                    "details": str(e),
                    "contamination": contamination
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

logger = logging.getLogger(__name__)

class RiskAnalysisAPI(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def __init__(self):
        self.predictor = RiskPredictor()
        super().__init__()

    def post(self, request):
        try:
            # Validate input
            required_fields = ['gdp_growth', 'inflation', 'interest_rate', 'market_volatility']
            if not all(field in request.data for field in required_fields):
                return Response(
                    {"error": "Missing required fields"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Make prediction
            prediction = self.predictor.predict_risk(request.data)
            
            # Save to database
            indicator = FinancialIndicator.objects.create(
                user=request.user,
                gdp_growth=request.data['gdp_growth'],
                inflation=request.data['inflation'],
                interest_rate=request.data['interest_rate'],
                market_volatility=request.data['market_volatility'],
                predicted_risk=prediction['risk_level'],
                confidence_score=prediction['confidence'],
                metadata={
                    'probabilities': prediction['probabilities'],
                    'model': prediction['model_type'],
                    'version': '1.0'
                }
            )
            
            return Response({
                'result': FinancialIndicatorSerializer(indicator).data,
                'prediction_details': prediction
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {str(e)}", exc_info=True)
            return Response(
                {"error": "Risk analysis failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get(self, request):
        # Get historical predictions for the user
        indicators = FinancialIndicator.objects.filter(user=request.user)
        serializer = FinancialIndicatorSerializer(indicators, many=True)
        return Response({
            'count': indicators.count(),
            'results': serializer.data
        })