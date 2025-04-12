from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination, getDynamicFormFields, renderResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from PersonalFinance.models import  ExpenseItems, Goals, GoalsItems, Incomes
from django.db.models import (
    FloatField,
    Sum,
    Count,
    OuterRef,
    Subquery,
    Value
)
from django.db.models.functions import Coalesce
from django.utils import timezone

class IncomeSerializer(serializers.ModelSerializer):
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

class IncomeSummaryView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        current_user = request.user
        now = timezone.now()
        
        # Calculate total income for current month
        current_month_income = Incomes.objects.filter(
            domain_user_id=current_user.domain_user_id.id,
            created_at__year=now.year,
            created_at__month=now.month
        ).aggregate(total=Sum('amount'))['total'] or 0

        # Calculate total income for previous month
        first_day_of_current_month = now.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timezone.timedelta(days=1)
        
        previous_month_income = Incomes.objects.filter(
            domain_user_id=current_user.domain_user_id.id,
            created_at__year=last_day_of_previous_month.year,
            created_at__month=last_day_of_previous_month.month
        ).aggregate(total=Sum('amount'))['total'] or 0

        response_data = {
            'current_month_income': float(current_month_income),
            'previous_month_income': float(previous_month_income),
            'income_change': float(current_month_income) - float(previous_month_income),
            'income_change_percentage': (
                ((float(current_month_income) - float(previous_month_income)) / float(previous_month_income)) * 100
                if previous_month_income != 0 else 0
            )
        }

        return Response(response_data)

class ExpenseSummaryView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        current_user = request.user
        now = timezone.now()
        
        # Calculate total expenses for current month
        current_month_expenses = ExpenseItems.objects.filter(
            domain_user_id=current_user.domain_user_id.id,
            date_of_expense__year=now.year,
            date_of_expense__month=now.month,
            expense_done='YES'
        ).aggregate(total=Sum('price'))['total'] or 0

        # Calculate total expenses for previous month
        first_day_of_current_month = now.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timezone.timedelta(days=1)
        
        previous_month_expenses = ExpenseItems.objects.filter(
            domain_user_id=current_user.domain_user_id.id,
            date_of_expense__year=last_day_of_previous_month.year,
            date_of_expense__month=last_day_of_previous_month.month,
            expense_done='YES'
        ).aggregate(total=Sum('price'))['total'] or 0

        # Get category breakdown
        category_data = self.get_expense_by_category(current_user, now, current_month_expenses)

        response_data = {
            'current_month_expenses': float(current_month_expenses),
            'previous_month_expenses': float(previous_month_expenses),
            'expense_change': float(current_month_expenses) - float(previous_month_expenses),
            'expense_change_percentage': (
                ((float(current_month_expenses) - float(previous_month_expenses)) / float(previous_month_expenses)) * 100
                if previous_month_expenses != 0 else 0
            ),
            'expense_by_category': category_data
        }

        return Response(response_data)

    def get_expense_by_category(self, user, now, current_month_total):
        # Get expenses grouped by category for current month
        category_expenses = ExpenseItems.objects.filter(
            domain_user_id=user.domain_user_id.id,
            date_of_expense__year=now.year,
            date_of_expense__month=now.month
        ).values('category_id__name').annotate(
            total=Sum('price')
        ).order_by('-total')

        return [
            {
                'category': item['category_id__name'],
                'amount': float(item['total'] or 0),
                'percentage': (
                    (float(item['total']) / float(current_month_total)) * 100
                    if current_month_total != 0 else 0
                )
            }
            for item in category_expenses
        ]

class GoalsBudgetSummaryView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        current_user = request.user
        now = timezone.now()
        
        # Current month budget subquery
        current_month_subquery = Goals.objects.filter(
            domain_user_id=OuterRef('domain_user_id'),
            created_at__year=now.year,
            created_at__month=now.month
        ).values('domain_user_id') \
        .annotate(total=Sum('budget')) \
        .values('total')[:1]

        # Previous month budget subquery
        first_day_current = now.replace(day=1)
        last_day_previous = (first_day_current - timezone.timedelta(days=1)).replace(day=1)
        
        previous_month_subquery = Goals.objects.filter(
            domain_user_id=OuterRef('domain_user_id'),
            created_at__year=last_day_previous.year,
            created_at__month=last_day_previous.month
        ).values('domain_user_id') \
        .annotate(total=Sum('budget')) \
        .values('total')[:1]

        # Get the budget values
        budget_data = Goals.objects.filter(
            domain_user_id=current_user.domain_user_id.id
        ).annotate(
            current_month_budget=Coalesce(
                Subquery(current_month_subquery),
                Value(0.0),
                output_field=FloatField()
            ),
            previous_month_budget=Coalesce(
                Subquery(previous_month_subquery),
                Value(0.0),
                output_field=FloatField()
            )
        ).values('current_month_budget', 'previous_month_budget').first()

        if not budget_data:
            budget_data = {
                'current_month_budget': 0.0,
                'previous_month_budget': 0.0
            }

        current = float(budget_data['current_month_budget'])
        previous = float(budget_data['previous_month_budget'])
        
        response_data = {
            'current_month_budget': current,
            'previous_month_budget': previous,
            'budget_change': current - previous,
            'budget_change_percentage': (
                ((current - previous) / previous * 100) if previous != 0 else 0
            )
        }

        return Response(response_data)