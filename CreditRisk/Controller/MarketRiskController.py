from rest_framework import serializers
from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination
import numpy as np
from CreditRisk.models import MarketRiskAssessment
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication


class MarketRiskAssessmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketRiskAssessment
        fields = ['id', 'stock_prices', 'exchange_rates', 'interest_rates', 'started_period', 'ended_period']
        extra_kwargs = {
            'stock_prices': {'required': True},
            'exchange_rates': {'required': False},
            'interest_rates': {'required': False}
        }
     
class MarketRiskListView(generics.ListAPIView):
    serializer_class = MarketRiskAssessmentSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        queryset=MarketRiskAssessment.objects.filter(domain_user_id=self.request.user.domain_user_id.id)
        return queryset
    
    @CommonListAPIMixin.common_list_decorator(MarketRiskAssessmentSerializer)
    def list(self,request,*args,**kwargs):
        return super().list(request,*args,**kwargs)

class MarketRiskRetrieveDestroyView(generics.RetrieveDestroyAPIView):
    serializer_class = MarketRiskAssessmentSerializer
    queryset = MarketRiskAssessment.objects.all()
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = None
    lookup_field = 'id'
    
    def get_queryset(self):
        return MarketRiskAssessment.objects.filter(
            domain_user_id=self.request.user.domain_user_id.id
        )
    
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
                    started_period = data.get('started_period'),
                    ended_period = data.get('ended_period'),
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

    def put(self, request, pk=None):
        try:
            # Get existing assessment
            assessment = MarketRiskAssessment.objects.get(pk=pk, domain_user_id=request.user.domain_user_id)
        except MarketRiskAssessment.DoesNotExist:
            return Response(
                {"error": "Market risk assessment not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = MarketRiskAssessmentSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            
            try:
                # Validate input data (same as POST)
                self.validate_input_data(data)
                
                # Calculate returns from stock prices (same as POST)
                stock_returns = self.calculate_returns(data['stock_prices'])
                
                # Calculate risk metrics (same as POST)
                var_95 = self.calculate_var(stock_returns)
                expected_shortfall = self.calculate_expected_shortfall(stock_returns)
                volatility = self.calculate_volatility(stock_returns)
                
                # Update assessment record
                assessment.stock_prices = data['stock_prices']
                assessment.exchange_rates = data.get('exchange_rates', [])
                assessment.interest_rates = data.get('interest_rates', [])
                assessment.started_period = data.get('started_period')
                assessment.ended_period = data.get('ended_period')
                assessment.value_at_risk = var_95
                assessment.expected_shortfall = expected_shortfall
                assessment.volatility = volatility
                assessment.updated_by_user_id = request.user
                assessment.save()
                
                # Prepare response with formatted values (same as POST)
                response_data = {
                    'value_at_risk': round(var_95, 6),
                    'expected_shortfall': round(expected_shortfall, 6),
                    'volatility': round(volatility, 6),
                    'assessment_id': assessment.id,
                    'updated_at': assessment.updated_at,
                    'message': 'Risk assessment updated successfully'
                }
                
                return Response(response_data, status=status.HTTP_200_OK)
            
            except ValueError as e:
                return Response(
                    {"error": "Invalid input data", "details": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )
            except Exception as e:
                return Response(
                    {"error": "Market risk assessment update failed", "details": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

