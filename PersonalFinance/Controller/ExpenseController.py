from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination, getDynamicFormFields, renderResponse
from rest_framework import generics
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from PersonalFinance.models import ExpenseItems, Expenses, Wallet
from django.db.models import (
    FloatField,
    Sum,
    F,
    ExpressionWrapper,
    Prefetch
)
from django.db.models.functions import Coalesce
from django.utils import timezone

class ExpenseItemSerializer(serializers.ModelSerializer):
    category_name=serializers.CharField(source='category_id.name',read_only=True)
    class Meta:
        model=ExpenseItems
        fields="__all__"

class ExpenseListSerializer(serializers.ModelSerializer):
    created_by_user_id=serializers.CharField(source='created_by_user_id.username',read_only=True)
    domain_user_id=serializers.CharField(source='domain_user_id.username',read_only=True)
    expense_total = serializers.SerializerMethodField()
    items = ExpenseItemSerializer(many=True, source='expense_id_item', read_only=True)
    # category_names = serializers.SerializerMethodField()
    class Meta:
        model=Expenses
        fields="__all__"
        
    def get_expense_total(self, obj):
        # Calculate total for this specific expense
        return obj.expense_id_item.aggregate(total=Sum('price'))['total'] or 0

class ExpenseSerializer(serializers.ModelSerializer):
    items=ExpenseItemSerializer(many=True,source='expense_id_item')
    class Meta:
        model=Expenses
        fields="__all__"
    
    def create(self,validated_data):
        items_data=validated_data.pop('expense_id_item')
        expense=Expenses.objects.create(**validated_data)
        for item_data in items_data:
            item_data.update({'domain_user_id':validated_data.get('domain_user_id')})
            ExpenseItems.objects.create(expense_id=expense,**item_data)

        return expense
    
    def update(self,instance,validated_data):
        items_data=validated_data.pop('expense_id_item')
        instance=super().update(instance,validated_data)
        items=[item_data.get('id') for item_data in items_data if 'id' in item_data]
        ExpenseItems.objects.filter(expense_id=instance).exclude(id__in=items).delete()

        for item_data in items_data:
            item_data.update({'domain_user_id':validated_data.get('domain_user_id')})

            if 'expense_id' in item_data:
                item_data.pop('expense_id')

            if 'id' in item_data:
                expenseItem=ExpenseItems.objects.filter(id=item_data.get('id'))
                expense_item_serializer=ExpenseItemSerializer(expenseItem.first(),data=item_data)
                if expense_item_serializer.is_valid():
                    expense_item_serializer.save()
            else:
                ExpenseItems.objects.create(expense_id=instance,**item_data)

        return instance

class CreateExpenseView(generics.CreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self,request,id=None):
        expense=Expenses.objects.filter(domain_user_id=request.user.domain_user_id.id,id=id).first() if id else Expenses()
        expenseItems=ExpenseItems.objects.filter(expense_id=id) if id else []
        expenseItems=ExpenseItemSerializer(expenseItems,many=True).data
      
        expenseFields=getDynamicFormFields(expense,request.user.domain_user_id.id, skip_fields=['status'])
        expenseItemFields=getDynamicFormFields(ExpenseItems(),request.user.domain_user_id.id,skip_related=['expense_id'],skip_fields=['expense_id','status'])
        return renderResponse(data={'expenseItems':expenseItems,'expenseFields':expenseFields,'expenseItemFields':expenseItemFields},message='Expense Fields',status=200)

    def post(self, request, id=None):
        data = request.data.copy()
        data.update({
            'created_by_user_id': request.user.id,
            'domain_user_id': request.user.domain_user_id.id
        })

         # Validation 1: Check for duplicate categories in items
        # if 'items' in data and isinstance(data['items'], list):
        #     category_ids = []
        #     duplicate_categories = []
            
        #     for item in data['items']:
        #         if 'category_id' in item:
        #             category_id = item['category_id']
        #             if category_id in category_ids:
        #                 duplicate_categories.append(category_id)
        #             category_ids.append(category_id)
            
        #     if duplicate_categories:
        #         return renderResponse(
        #             data={'items': f'Duplicate categories found in items: {", ".join(map(str, duplicate_categories))}'},
        #             message='Each category can only be used once per expense',
        #             status=400
        #         )

        # Validation 2: Budget sum check
        if 'items' in data:
            try:
                items_sum = sum(float(item.get('price', 0)) for item in data['items'])
                
                # Get the user's wallet balance
                wallet = Wallet.objects.get(user=request.user)
                wallet.update_balance()  # Ensure balance is up-to-date
                
                if items_sum > wallet.current_balance:
                    return renderResponse(
                        data={
                            'items_sum': f'Items sum: {items_sum}',
                            'difference': f'Exceeds by: {items_sum - wallet.current_balance}',
                            'wallet_balance': wallet.current_balance
                        },
                        message='Sum of items price exceeds your wallet balance',
                        status=400
                    )
  
            except (ValueError, TypeError) as e:
                return renderResponse(
                    data={'error': str(e)},
                    message='Invalid prices values',
                    status=400
                )
            except Wallet.DoesNotExist:
                return renderResponse(
                    data={'error': 'Wallet not found'},
                    message='User wallet does not exist',
                    status=404
                )

        # Existing create/update logic
        if id:
            expense = Expenses.objects.filter(
                domain_user_id=request.user.domain_user_id.id,
                id=id
            ).first()
            if not expense:
                return renderResponse(
                    data={},
                    message='Expense Not Found',
                    status=404
                )
            serializer = ExpenseSerializer(expense, data=data)
        else:
            serializer = ExpenseSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            return renderResponse(
                data=serializer.data,
                message='Expense created successfully' if not id else 'Expense updated successfully',
                status=201
            )

        return renderResponse(
            data=serializer.errors,
            message='Validation error',
            status=400
        )

class ExpenseListView(generics.ListAPIView):
    serializer_class = ExpenseListSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        now = timezone.now()
        queryset = Expenses.objects.filter(
            domain_user_id=self.request.user.domain_user_id.id,
        ).prefetch_related(
            Prefetch(
                'expense_id_item',
                queryset=ExpenseItems.objects.select_related('category_id')
            )
        ).annotate(
            expense_total=Coalesce(
                Sum('expense_id_item__price'),
                0.0,
                output_field=FloatField()
            )
        )

        expense_ids = queryset.values_list('id', flat=True)

        # total_expense du mois courant
        monthly_expense_qs = ExpenseItems.objects.filter(
            expense_id__in=expense_ids,
            created_at__year=now.year,
            created_at__month=now.month,
            expense_done='YES'
        )

        total_expense = monthly_expense_qs.aggregate(total=Sum('price'))['total'] or 0

        categories_summary = monthly_expense_qs.values(
            'category_id', 'category_id__name'
        ).annotate(
            total_amount=Sum('price')
        ).annotate(
            percentage=ExpressionWrapper(
                F('total_amount') * 100.0 / (total_expense or 1),
                output_field=FloatField()
            )
        ).order_by('-total_amount')

        self.extra_data = {
            'total_expense': float(total_expense),
            'categories_summary': [
                {
                    'category_id': item['category_id'],
                    'category_name': item['category_id__name'],
                    'total_amount': float(item['total_amount']),
                    'percentage': round(float(item['percentage']), 1)
                }
                for item in categories_summary
            ]
        }

        return queryset

    def finalize_response(self, request, response, *args, **kwargs):
        response = super().finalize_response(request, response, *args, **kwargs)
        if isinstance(response.data, dict) and hasattr(self, 'extra_data'):
            response.data.update(self.extra_data)
        return response

    @CommonListAPIMixin.common_list_decorator(ExpenseListSerializer)
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
