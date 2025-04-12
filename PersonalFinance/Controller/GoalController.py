from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination, getDynamicFormFields, renderResponse
from rest_framework import generics
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from PersonalFinance.models import Goals, GoalsItems, Wallet
from datetime import date
from django.utils import timezone
from django.db.models import (
    FloatField,
    Sum,
    OuterRef,
    Subquery,
    F,
    Q,
    ExpressionWrapper,
    Value,
    Count,
    Case,
    When
)
from django.db.models.functions import Coalesce

class GoalItemSerializer(serializers.ModelSerializer):
    expense_description=serializers.CharField(source='expense_id.description',read_only=True)
    class Meta:
        model=GoalsItems
        fields="__all__"
class GoalListSerializer(serializers.ModelSerializer):
    created_by_user = serializers.CharField(source='created_by_user_id.username', read_only=True)
    updated_by_user = serializers.CharField(source='updated_by_user_id.username', read_only=True)
    domain_user = serializers.CharField(source='domain_user_id.username', read_only=True)
    total_expenses = serializers.FloatField(read_only=True)
    progress_percentage = serializers.FloatField(read_only=True)
    expenses_count = serializers.IntegerField(read_only=True)
    linked_expenses = serializers.SerializerMethodField()
    total_budget = serializers.FloatField(read_only=True)
    total_budget_current_month = serializers.FloatField(read_only=True)

    class Meta:
        model = Goals
        fields = [
            'id', 'description', 'budget', 'status', 'reached',
            'begin_date', 'end_date', 'priority', 'created_by_user', 'total_budget',
            'updated_by_user', 'domain_user', 'created_at', 'updated_at',
            'total_expenses', 'progress_percentage', 'expenses_count', 'linked_expenses', 'total_budget_current_month'
        ]

    def get_linked_expenses(self, obj):
        linked_expenses = []
        for goal_item in obj.goal_id_item.all():
            # Get all expense items related to this expense
            for expense_item in goal_item.expense_id.expense_id_item.all():
                linked_expenses.append({
                    'expense_id': goal_item.expense_id.id,
                    'expense_item_id': expense_item.id,
                    'price': expense_item.price,  # Using price from ExpenseItems
                    'description': expense_item.expense_id.description,
                    'date': expense_item.date_of_expense,
                    'category': expense_item.category_id.name if expense_item.category_id else None
                })
        return linked_expenses

class GoalSerializer(serializers.ModelSerializer):
    items = GoalItemSerializer(many=True, source='goal_id_item')
    
    class Meta:
        model = Goals
        fields = "__all__"
    
    def create(self, validated_data):
        items_data = validated_data.pop('goal_id_item')
        goal = Goals.objects.create(**validated_data)
        for item_data in items_data:
            item_data.update({'domain_user_id': validated_data.get('domain_user_id')})
            GoalsItems.objects.create(goal_id=goal, **item_data)
        return goal
    
    def update(self, instance, validated_data):
        items_data = validated_data.pop('goal_id_item', [])
        instance = super().update(instance, validated_data)
        
        # Get list of item IDs to keep
        keep_items = [item_data.get('id') for item_data in items_data if 'id' in item_data]
        
        # Delete items not in the keep list
        GoalsItems.objects.filter(goal_id=instance).exclude(id__in=keep_items).delete()

        for item_data in items_data:
            item_data.update({'domain_user_id': validated_data.get('domain_user_id')})

            if 'expense_id' in item_data:
                item_data.pop('expense_id')

            if 'id' in item_data:
                goal_item = GoalsItems.objects.filter(id=item_data.get('id')).first()
                if goal_item:
                    goal_item_serializer = GoalItemSerializer(goal_item, data=item_data)
                    if goal_item_serializer.is_valid():
                        goal_item_serializer.save()
            else:
                GoalsItems.objects.create(goal_id=instance, **item_data)

        return instance

class CreateGoalView(generics.CreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, id=None):
        goal = Goals.objects.filter(domain_user_id=request.user.domain_user_id.id, id=id).first() if id else Goals()
        goalItems = GoalsItems.objects.filter(goal_id=id) if id else []  # Changed expense_id to goal_id
        goalItems = GoalItemSerializer(goalItems, many=True).data
        goalFields = getDynamicFormFields(goal, request.user.domain_user_id.id, skip_fields=['status'])
        goalItemFields = getDynamicFormFields(GoalsItems(), request.user.domain_user_id.id, skip_related=['goal_id'], skip_fields=['goal_id', 'status'])
        return renderResponse(data={'goalItems': goalItems, 'goalFields': goalFields, 'goalItemFields': goalItemFields}, message='Goal Fields', status=200)
        
    def post(self, request, id=None):
        data = request.data.copy()
        data.update({
            'created_by_user_id': request.user.id,
            'domain_user_id': request.user.domain_user_id.id
        })

         # Validation 1: Check for duplicate categories in items
        if 'items' in data and isinstance(data['items'], list):
            expense_ids = []
            duplicate_expenses = []
            
            for item in data['items']:
                if 'expense_id' in item:
                    expense_id = item['expense_id']
                    if expense_id in expense_ids:
                        duplicate_expenses.append(expense_id)
                    expense_ids.append(expense_id)
            
            if duplicate_expenses:
                return renderResponse(
                    data={'items': f'Duplicate expenses found in items: {", ".join(map(str, duplicate_expenses))}'},
                    message='Each expense can only be used once per goal',
                    status=400
                )

        # Validation 2: Budget sum check
        if 'budget' in data:
            try:
                budget = float(data['budget'])
                # Get the user's wallet balance
                wallet = Wallet.objects.get(user=request.user)
                wallet.update_balance()  # Ensure balance is up-to-date
                    
                if budget > wallet.current_balance:
                    return renderResponse(
                        data={
                            'budget': budget,
                            'wallet_balance': wallet.current_balance,
                            'difference': budget - wallet.current_balance
                        },
                        message='Budget cannot exceed your current wallet balance',
                        status=400
                    )
                    
            except (ValueError, TypeError) as e:
                return renderResponse(
                    data={'error': str(e)},
                    message='Invalid budget values',
                    status=400
                )
            except Wallet.DoesNotExist:
                return renderResponse(
                    data={'error': 'Wallet not found'},
                    message='User wallet does not exist',
                    status=404
                )

        # Validation 3: Date validation
        date_errors = {}
        today = timezone.now().date()
        if 'begin_date' in data:
            begin_date = date.fromisoformat(data['begin_date'])
        
        # if 'begin_date' in data:
        #     try:
        #         begin_date = date.fromisoformat(data['begin_date'])
        #         if begin_date < today:
        #             date_errors['begin_date'] = 'Cannot set begin date in the past'
        #     except (ValueError, TypeError):
        #         date_errors['begin_date'] = 'Invalid date format (YYYY-MM-DD required)'

        if 'end_date' in data:
            try:
                end_date = date.fromisoformat(data['end_date'])
                if 'begin_date' in data and not date_errors.get('begin_date'):
                    if end_date < begin_date:
                        date_errors['end_date'] = 'End date must be after begin date'
            except (ValueError, TypeError):
                date_errors['end_date'] = 'Invalid date format (YYYY-MM-DD required)'

        if date_errors:
            return renderResponse(
                data=date_errors,
                message=f"Invalid date: {', '.join(date_errors.values())}",
                status=400
            )

        # Existing create/update logic
        if id:
            goal = Goals.objects.filter(
                domain_user_id=request.user.domain_user_id.id,
                id=id
            ).first()
            if not goal:
                return renderResponse(
                    data={},
                    message='Goal Not Found',
                    status=404
                )
            if goal.reached == 'YES':
                return renderResponse(
                    data={},
                    message='Cannot update a goal that has already been reached',
                    status=400
                )
            serializer = GoalSerializer(goal, data=data)
        else:
            serializer = GoalSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            return renderResponse(
                data=serializer.data,
                message='Goal created successfully' if not id else 'Goal updated successfully',
                status=201
            )

        return renderResponse(
            data=serializer.errors,
            message='Validation error',
            status=400
        )

class GoalListView(generics.ListAPIView):
    serializer_class = GoalListSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        now = timezone.now()

        # Base queryset
        queryset = Goals.objects.all()

        # Sous-requête : budget total du mois courant
        total_budget_month_subquery = Goals.objects.filter(
            domain_user_id=OuterRef('domain_user_id'),
            created_at__year=now.year,
            created_at__month=now.month
        ).values('domain_user_id') \
        .annotate(total=Sum('budget')) \
        .values('total')[:1]

        # Annotations
        queryset = queryset.annotate(
            total_expenses=Coalesce(
                Sum('goal_id_item__expense_id__expense_id_item__price', 
                    filter=Q(goal_id_item__expense_id__expense_id_item__expense_done='YES'),
                    output_field=FloatField()
                ),
                0.0
            ),
            expenses_count=Count('goal_id_item__expense_id__expense_id_item', 
                filter=Q(goal_id_item__expense_id__expense_id_item__expense_done='YES'),
                distinct=True
            ),
            total_budget_current_month=Coalesce(
                Subquery(total_budget_month_subquery),
                Value(0.0),
                output_field=FloatField()
            ),
            progress_percentage=Case(
            When(budget=0, then=Value(0)),
            default=ExpressionWrapper(
                F('total_expenses') / F('budget') * 100,
                output_field=FloatField()
            ),
            output_field=FloatField()
        )
        )
        # Mise à jour des objectifs atteints
        reached_goals = []
        for goal in queryset:
            if goal.progress_percentage >= 100 and goal.reached != 'YES':
                reached_goals.append(goal.id)
                goal.reached = 'YES'

        if reached_goals:
            Goals.objects.filter(id__in=reached_goals).update(reached='YES')

        return queryset


    @CommonListAPIMixin.common_list_decorator(GoalListSerializer)
    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        return response