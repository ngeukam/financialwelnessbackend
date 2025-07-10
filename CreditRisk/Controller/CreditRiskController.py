from dataclasses import fields
from datetime import datetime
from rest_framework import serializers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination
import numpy as np
import logging
from django.utils import timezone
from CreditRisk.Helper.ml_service import RiskPredictor
from CreditRisk.models import CreditRiskAssessment, EarlyWarningIndicator, FinancialIndicator
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.db.models.functions import Cast
from django.db.models import (
    Avg, Count, Min, Max, StdDev, 
    Case, When, Value, CharField,
    ExpressionWrapper, DurationField, 
    F, IntegerField
)
from django.db.models.functions import TruncDate, TruncWeek, TruncMonth
from django.db.models import Func, FloatField

class CreditRiskAssessmentSerializer(serializers.ModelSerializer):
    income = serializers.FloatField(min_value=0)
    loan_amount = serializers.FloatField(min_value=0)
    credit_score = serializers.IntegerField(min_value=300, max_value=850)
    
    class Meta:
        model = CreditRiskAssessment
        fields = ['id','income', 'loan_amount', 'credit_score', 'started_period', 'ended_period', 'loan_duration_months']
        extra_kwargs = {
            'income': {'required': True},
            'loan_amount': {'required': True},
            'credit_score': {'required': True}
        }

        
class EarlyWarningIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = EarlyWarningIndicator
        fields = ['id', 'indicator_type', 'value', 'started_period', 'ended_period', 'threshold', 'is_anomaly', 'timestamp', 'metadata']
    
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
            'metadata',
            'started_period', 
            'ended_period'
        ]

class CreditRiskListView(generics.ListAPIView):
    serializer_class = CreditRiskAssessmentSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        queryset=CreditRiskAssessment.objects.filter(domain_user_id=self.request.user.domain_user_id.id)
        return queryset
    
    @CommonListAPIMixin.common_list_decorator(CreditRiskAssessmentSerializer)
    def list(self,request,*args,**kwargs):
        return super().list(request,*args,**kwargs)

class CreditRiskRetrieveDestroyView(generics.RetrieveDestroyAPIView):
    serializer_class = CreditRiskAssessmentSerializer
    queryset = CreditRiskAssessment.objects.all()
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = None
    lookup_field = 'id'
    
    def get_queryset(self):
        return CreditRiskAssessment.objects.filter(
            domain_user_id=self.request.user.domain_user_id.id
        )
          
class CreditRiskAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    # Define feature columns as class-level constant
    FEATURE_COLUMNS = ['income', 'loan_amount', 'credit_score', 'duration_months']
    
    def _get_training_data(self):
        """Fetch historical data with precise duration calculation"""
        try:
            historical_data = CreditRiskAssessment.objects.filter(
                prediction__isnull=False,
                started_period__isnull=False,
                ended_period__isnull=False
            ).annotate(
                duration_days=ExpressionWrapper(
                    F('ended_period') - F('started_period'),
                    output_field=DurationField()
                )
            ).annotate(
                duration_months=Cast(
                    F('duration_days') / (365.25/12),
                    output_field=IntegerField()
                )
            ).values_list(
                'income',
                'loan_amount',
                'credit_score',
                'duration_months',
                'prediction'
            )

            if historical_data:
                df = pd.DataFrame(list(historical_data),
                                columns=self.FEATURE_COLUMNS + ['default'])
                df['default'] = df['default'].astype(int)  # Convert bool to 0/1
                return df
            return self._get_fallback_data()
            
        except Exception as e:
            logger.warning(f"Training data fetch failed: {str(e)}")
            return self._get_fallback_data()

    def _get_fallback_data(self):
        """Sample data for initial model training"""
        return pd.DataFrame({
            'income': [
                30000, 45000, 60000, 80000, 55000, 70000,
                40000, 65000, 75000, 50000, 90000, 38000
            ],
            'loan_amount': [
                20000, 15000, 25000, 10000, 18000, 20000,
                22000, 16000, 27000, 12000, 30000, 24000
            ],
            'credit_score': [
                580, 620, 700, 750, 680, 710,
                630, 695, 720, 660, 740, 610
            ],
            'duration_months': [
                12, 18, 24, 6, 12, 36,
                30, 18, 48, 12, 60, 24
            ],
            'default': [
                1, 1, 0, 0, 0, 0,
                1, 0, 0, 1, 0, 1
            ]
        })


    def __init__(self):
        super().__init__()
        self.model = self._train_model()
    
    def _train_model(self):
        """Train and cache a RandomForest classifier"""
        df = self._get_training_data()
        X = df[self.FEATURE_COLUMNS]
        y = df['default']
        
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced'
        )
        model.fit(X, y)
        return model
        
    def post(self, request):
        """
        Process credit risk assessment:
        1. Validates input
        2. Calculates loan duration
        3. Predicts default risk (prediction field)
        4. Computes risk score
        5. Saves assessment
        """
        serializer = CreditRiskAssessmentSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            data = serializer.validated_data
            start_date = data['started_period']
            end_date = data['ended_period']
            
            # Calculate duration
            duration_months = max(1, round((end_date - start_date).days / (365.25/12)))
            
            # Prepare model input
            input_data = pd.DataFrame([[
                data['income'],
                data['loan_amount'],
                data['credit_score'],
                duration_months
            ]], columns=self.FEATURE_COLUMNS)
            
            # Generate prediction (0=False=Low Risk, 1=True=High Risk)
            prediction = int(self.model.predict(input_data)[0])  # Convert to 0/1
            print('prediction', prediction)
            # Calculate explanatory risk score (0-1 scale)
            risk_score = self._calculate_risk_score(
                data['income'],
                data['loan_amount'],
                data['credit_score'],
                duration_months
            )
            
            # Persist assessment
            assessment = serializer.save(
                domain_user_id=request.user,
                created_by_user_id=request.user,
                updated_by_user_id=request.user,
                risk_score=risk_score,
                prediction=bool(prediction),  # Store as Boolean
                loan_duration_months=duration_months
            )
            
            return Response({
                'risk_score': round(risk_score, 4),
                'prediction': prediction,
                'duration_months': duration_months,
                'assessment_id': assessment.id,
                'warning': bool(prediction)  # True if high risk
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Assessment failed: {str(e)}")
            return Response(
                {"error": "Assessment processing failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def put(self, request, pk=None):
        """
        Update credit risk assessment:
        1. Validates input
        2. Calculates loan duration
        3. Predicts default risk (prediction field)
        4. Computes risk score
        5. Updates assessment
        """
        try:
            assessment = CreditRiskAssessment.objects.get(pk=pk)
        except CreditRiskAssessment.DoesNotExist:
            return Response(
                {"error": "Assessment not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = CreditRiskAssessmentSerializer(assessment, data=request.data, partial=False)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            data = serializer.validated_data
            start_date = data['started_period']
            end_date = data['ended_period']
            
            # Calculate duration (same logic as POST)
            duration_months = max(1, round((end_date - start_date).days / (365.25/12)))
            
            # Prepare model input (same logic as POST)
            input_data = pd.DataFrame([[
                data['income'],
                data['loan_amount'],
                data['credit_score'],
                duration_months
            ]], columns=self.FEATURE_COLUMNS)
            
            # Generate prediction (same logic as POST)
            prediction = int(self.model.predict(input_data)[0])
            
            # Calculate risk score (same logic as POST)
            risk_score = self._calculate_risk_score(
                data['income'],
                data['loan_amount'],
                data['credit_score'],
                duration_months
            )
            
            # Update assessment (same logic but with update fields)
            updated_assessment = serializer.save(
                updated_by_user_id=request.user,
                risk_score=risk_score,
                prediction=bool(prediction),
                loan_duration_months=duration_months
            )
            
            return Response({
                'risk_score': round(risk_score, 4),
                'prediction': prediction,
                'duration_months': duration_months,
                'assessment_id': updated_assessment.id,
                'warning': bool(prediction)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Assessment update failed: {str(e)}")
            return Response(
                {"error": "Assessment processing failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
    def _calculate_risk_score(self, income, loan_amount, credit_score, duration_months):
        """
        Calculate a normalized risk score (0-1) where:
        - Higher values indicate greater risk
        - Factors: Income-to-debt ratio, credit score, and duration
        """
        debt_ratio = loan_amount / max(income, 1)  # Prevent division by zero
        score_factor = 1 - (min(credit_score, 850) / 850)  # Normalize to 0-1
        duration_factor = min(duration_months / 60, 1)  # Cap at 5 years
        
        return (0.4 * debt_ratio) + (0.4 * score_factor) + (0.2 * duration_factor)
    
logger = logging.getLogger(__name__)
class CreditRiskEvolutionAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get comprehensive risk evolution analytics"""
        try:
            # Get and validate parameters
            user_id = request.query_params.get('user_id')
            start_date = request.query_params.get('started_period')
            end_date = request.query_params.get('ended_period')
            time_grouping = request.query_params.get('group_by', 'day')
            
            # Validate required parameters
            if not all([start_date, end_date]):
                return Response(
                    {"message": "Both started_period and ended_period are required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convert dates
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            except ValueError:
                return Response(
                    {"message": "Invalid date format. Use YYYY-MM-DD"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if start_date > end_date:
                return Response(
                    {"message": "Start date must be before end date"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Build base queryset - REMOVED __date lookup
            queryset = CreditRiskAssessment.objects.filter(
                started_period__gte=start_date,  # Changed from started_period__date__gte
                ended_period__lte=end_date,      # Changed from ended_period__date__lte
                risk_score__isnull=False,
                prediction__isnull=False
            )
            
            # Apply user filter if provided
            if user_id:
                queryset = queryset.filter(domain_user_id=user_id)
            
            if not queryset.exists():
                return Response(
                    {"message": "No records found for the specified criteria"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Determine date truncation
            if time_grouping == 'week':
                date_trunc = TruncWeek('created_at')
            elif time_grouping == 'month':
                date_trunc = TruncMonth('created_at')
            else:
                date_trunc = TruncDate('created_at')

            # Generate time-series data
            evolution_data = list(
                queryset.annotate(
                    period=date_trunc
                ).values('period').annotate(
                    avg_risk=Avg('risk_score'),
                    default_rate=Avg('prediction'),
                    assessment_count=Count('id'),
                    min_risk=Min('risk_score'),
                    max_risk=Max('risk_score')
                ).order_by('period')
            )

            # Calculate statistics
            def calculate_percentiles(queryset, field_name):
                values = list(queryset.annotate(
                    float_value=Cast(field_name, FloatField())
                ).values_list('float_value', flat=True))
                
                if not values:
                    return {'q1': None, 'median': None, 'q3': None}
                    
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                return {
                    'q1': float(q1),
                    'median': float(median),
                    'q3': float(q3)
                }

            # Calculate basic stats
            stats = queryset.aggregate(
                overall_avg_risk=Avg('risk_score'),
                overall_default_rate=Avg('prediction'),
                risk_std_dev=StdDev('risk_score'),
                total_assessments=Count('id')
            )
            
            # Add percentiles
            percentiles = calculate_percentiles(queryset, 'risk_score')
            stats.update(percentiles)

            # Risk distribution
            risk_distribution = (
                queryset.annotate(
                    risk_bucket=Case(
                        When(risk_score__lte=0.3, then=Value('low')),
                        When(risk_score__lte=0.6, then=Value('medium')),
                        default=Value('high'),
                        output_field=CharField()
                    )
                ).values('risk_bucket').annotate(
                    count=Count('id')
                ).annotate(
                    percentage=ExpressionWrapper(
                        Count('id') * 100.0 / len(queryset),
                        output_field=FloatField()
                    )
                )
            )
            print(risk_distribution)
            # Prepare response
            response = {
                "metadata": {
                    "time_grouping": time_grouping,
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "user_filter": bool(user_id)
                },
                "time_series": evolution_data,
                "statistics": {
                    "risk_score": {
                        "average": stats['overall_avg_risk'],
                        "std_dev": stats['risk_std_dev'],
                        "quartiles": {
                            "q1": stats['q1'],
                            "median": stats['median'],
                            "q3": stats['q3']
                        },
                        "distribution": {
                            bucket['risk_bucket']: {
                                "count": bucket['count'],
                                "percentage": bucket['percentage']
                            } for bucket in risk_distribution
                        }
                    },
                    "default_rate": stats['overall_default_rate'],
                    "total_assessments": stats['total_assessments']
                }
            }

            return Response(response, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Risk evolution analysis failed: {str(e)}", exc_info=True)
            return Response(
                {"error": "Analysis failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

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
