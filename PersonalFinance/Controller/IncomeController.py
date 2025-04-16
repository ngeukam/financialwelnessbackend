from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination, getDynamicFormFields, renderResponse
from rest_framework import generics
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from UserServices.models import Users
from PersonalFinance.models import  Incomes
from django.db.models import (
    FloatField,
    Sum,
)
from django.db.models.functions import Coalesce
from django.utils import timezone

class IncomeListSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by_user_id.username', read_only=True)
    domain_username = serializers.CharField(source='domain_user_id.username', read_only=True)
    income_total = serializers.SerializerMethodField()
    
    class Meta:
        model = Incomes
        fields = "__all__"
        extra_fields = ['created_by_username', 'domain_username', 'income_total']
        
    def get_income_total(self, obj):
        # Use the annotated value if available, otherwise calculate
        if hasattr(obj, 'income_total'):
            return obj.income_total
        return obj.amount or 0

class IncomeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Incomes
        fields = "__all__"
    
    def create(self, validated_data):
        return Incomes.objects.create(**validated_data)
    
    def update(self, instance, validated_data):
        return super().update(instance, validated_data)

class CreateIncomeView(generics.CreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, id=None):
        income = Incomes.objects.filter(
            domain_user_id=request.user.domain_user_id.id, 
            id=id
        ).first() if id else Incomes()
        
        income_fields = getDynamicFormFields(
            income, 
            request.user.domain_user_id.id, 
            skip_fields=['status']
        )
        
        return renderResponse(
            data={'incomeFields': income_fields},
            message='Income Fields',
            status=200
        )

    def post(self, request, id=None):
        data = request.data.copy()
        data.update({
            'added_by_user_id': request.user.id,
            'domain_user_id': request.user.domain_user_id.id
        })
        # Validation: Check wallet balance for income amount
        try:
            amount = float(data.get('amount', 0))            
            # For income, we might want different validation than expense
            # For example, check if amount is positive
            if amount <= 0:
                return renderResponse(
                    data={'amount': amount},
                    message='Income amount must be positive',
                    status=400
                )
                
        except (ValueError, TypeError) as e:
            return renderResponse(
                data={'error': str(e)},
                message='Invalid amount value',
                status=400
            )

        # Create/update logic
        if id:
            income = Incomes.objects.filter(
                domain_user_id=request.user.domain_user_id.id,
                id=id,
                status='ACTIVE'
            ).first()
            
            if not income:
                return renderResponse(
                    data={},
                    message='Income Not Found',
                    status=404
                )
            data['updated_by_user_id'] = request.user.id
            serializer = IncomeSerializer(income, data=data)
        else:
            serializer = IncomeSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            
            return renderResponse(
                data=serializer.data,
                message='Income created successfully' if not id else 'Income updated successfully',
                status=201
            )

        return renderResponse(
            data=serializer.errors,
            message='Validation error',
            status=400
        )
        
class IncomeListView(generics.ListAPIView):
    serializer_class = IncomeListSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        current_user = self.request.user
        now = timezone.now()
        
        queryset = Incomes.objects.filter(
            domain_user_id=current_user.domain_user_id.id,
        ).annotate(
            income_total=Coalesce(
                Sum('amount'),
                0.0,
                output_field=FloatField()
            )
        ).select_related('domain_user_id', 'added_by_user_id')

        # Calculate total income for current month
        monthly_income = Incomes.objects.filter(
            domain_user_id=current_user.domain_user_id.id,
            created_at__year=now.year,
            created_at__month=now.month
        ).aggregate(total=Sum('amount'))['total'] or 0

        self.extra_data = {
            'total_income': float(monthly_income),
        }

        return queryset

    def finalize_response(self, request, response, *args, **kwargs):
        response = super().finalize_response(request, response, *args, **kwargs)
        if hasattr(self, 'extra_data'):
            response.data.update(self.extra_data)
        return response

    @CommonListAPIMixin.common_list_decorator(IncomeListSerializer)
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)