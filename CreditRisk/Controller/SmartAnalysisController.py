from rest_framework import serializers
from rest_framework.response import Response
from rest_framework.views import APIView
import logging
from CreditRisk.Helper.ml_service import RiskPredictor
from CreditRisk.models import FinancialIndicator
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import status, generics

from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination

        
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

class FinancialRiskIndicatorListView(generics.ListAPIView):
    serializer_class = FinancialIndicatorSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        queryset=FinancialIndicator.objects.filter(domain_user_id=self.request.user.domain_user_id.id)
        return queryset
    
    @CommonListAPIMixin.common_list_decorator(FinancialIndicatorSerializer)
    def list(self,request,*args,**kwargs):
        return super().list(request,*args,**kwargs)

class FinancialRiskIndicatorRetrieveDestroyView(generics.RetrieveDestroyAPIView):
    serializer_class = FinancialIndicatorSerializer
    queryset = FinancialIndicator.objects.all()
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = None
    lookup_field = 'id'
    
    def get_queryset(self):
        return FinancialIndicator.objects.filter(
            domain_user_id=self.request.user.domain_user_id.id
        )
        
logger = logging.getLogger(__name__)
class FinancialRiskAnalysisAPI(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def __init__(self):
        self.predictor = RiskPredictor()
        super().__init__()

    def post(self, request):
        try:
            # Validate input
            required_fields = ['gdp_growth', 'inflation', 'interest_rate', 'market_volatility', 'ended_period', 'started_period']
            if not all(field in request.data for field in required_fields):
                return Response(
                    {"error": "Missing required fields"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Make prediction
            prediction = self.predictor.predict_risk(request.data)
            
            # Save to database
            indicator = FinancialIndicator.objects.create(
                domain_user_id=request.user,
                created_by_user_id=request.user,
                updated_by_user_id=request.user,
                started_period = request.data['started_period'],
                ended_period = request.data['ended_period'],
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
    def put(self, request, pk=None):
        try:
            # Get existing assessment
            try:
                indicator = FinancialIndicator.objects.get(
                    pk=pk,
                    domain_user_id=request.user.domain_user_id
                )
            except FinancialIndicator.DoesNotExist:
                return Response(
                    {"error": "Financial risk assessment not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Validate input
            required_fields = ['gdp_growth', 'inflation', 'interest_rate', 'market_volatility', 'ended_period', 'started_period']
            if not all(field in request.data for field in required_fields):
                return Response(
                    {"error": "Missing required fields"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Make new prediction with updated data
            prediction = self.predictor.predict_risk(request.data)
            
            # Update the existing record
            indicator.started_period = request.data.get('started_period', indicator.started_period)
            indicator.ended_period = request.data.get('ended_period', indicator.ended_period)
            indicator.gdp_growth = request.data['gdp_growth']
            indicator.inflation = request.data['inflation']
            indicator.interest_rate = request.data['interest_rate']
            indicator.market_volatility = request.data['market_volatility']
            indicator.predicted_risk = prediction['risk_level']
            indicator.confidence_score = prediction['confidence']
            indicator.metadata = {
                'probabilities': prediction['probabilities'],
                'model': prediction['model_type'],
                'version': '1.0'
            }
            indicator.updated_by_user_id = request.user
            indicator.save()
            
            return Response({
                'result': FinancialIndicatorSerializer(indicator).data,
                'prediction_details': prediction
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Risk assessment update failed: {str(e)}", exc_info=True)
            return Response(
                {"error": "Risk assessment update failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    def get(self, request):
        # Get historical predictions for the user
        indicators = FinancialIndicator.objects.filter(domain_user_id=request.user)
        serializer = FinancialIndicatorSerializer(indicators, many=True)
        return Response({
            'count': indicators.count(),
            'results': serializer.data
        })