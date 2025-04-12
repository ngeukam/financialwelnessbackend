from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Sum, Count
from datetime import date, datetime, timedelta
import json
from PersonalFinance.models import Categories, ExpenseItems, Incomes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
import random
from django.db.models import F, CharField, Value, ExpressionWrapper
from django.db.models.functions import Coalesce
from rest_framework import serializers

class FinancialSummarySerializer(serializers.Serializer):
    total_balance = serializers.FloatField(read_only=True)
    total_income = serializers.FloatField(read_only=True)
    total_expenses = serializers.FloatField(read_only=True)
    
class FinancialChartsAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        time_period = request.query_params.get('period', 'monthly')  # monthly, yearly, weekly
        months_back = int(request.query_params.get('months', 12))    # default 12 months
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30*months_back)
        
        # 1. Income vs Expenses Over Time
        income_data = self._get_time_series_data(
            model=Incomes,
            user=user,
            amount_field='amount',
            date_field='date_of_received',
            period=time_period,
            start_date=start_date,
            end_date=end_date
        )
        
        expenses_data = self._get_time_series_data(
            model=ExpenseItems,
            user=user,
            amount_field='price',
            date_field='date_of_expense',
            period=time_period,
            start_date=start_date,
            end_date=end_date,
            extra_filters={'expense_done': 'YES', 'expense_id__status': 'ACTIVE'}
        )
        print('expenses_data',expenses_data)
        # 2. Spending by Category
        category_spending = self._get_category_spending(user, start_date, end_date)
        
        return Response({
            'income_vs_expenses': {
                'labels': list(income_data.keys()),
                'income': list(income_data.values()),
                'expenses': list(expenses_data.values())
            },
            'spending_by_category': category_spending
        })
    
    def _parse_date_string(self, date_str):
        """Helper to parse YYYY-MM-DD string into date object"""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            return None
    
    def _generate_random_color(self):
        """Generate a random hex color"""
        return f"#{''.join([random.choice('0123456789ABCDEF') for _ in range(6)])}"
    
    def _get_time_series_data(self, model, user, amount_field, date_field, period, start_date, end_date, extra_filters=None):
        """Process time series data for CharField dates"""
        filters = {
            'domain_user_id': user.domain_user_id,
            # 'status': 'ACTIVE',
        }
        # Only add status filter for Incomes model
        if model == Incomes:
            filters['status'] = 'ACTIVE'

        if extra_filters:
            filters.update(extra_filters)
            
        queryset = model.objects.filter(**filters).values(date_field, amount_field)
        print('queryset',queryset)
        
        period_data = {}
        for entry in queryset:
            date_str = entry.get(date_field)
            if not date_str:
                continue
            
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:  # Assume it's already a date object
                date_obj = date_str
            # date_obj = self._parse_date_string(date_str)
            if not date_obj or not (start_date <= date_obj <= end_date):
                continue
            
            if period == 'monthly':
                key = date_obj.strftime('%b %Y')
            elif period == 'yearly':
                key = date_obj.strftime('%Y')
            else:  # weekly
                key = f"Week {date_obj.isocalendar()[1]}, {date_obj.year}"
            
            amount = float(entry.get(amount_field, 0))
            period_data[key] = period_data.get(key, 0) + amount
        
        # Fill in missing periods with 0
        filled_data = {}
        current_date = start_date
        while current_date <= end_date:
            if period == 'monthly':
                key = current_date.strftime('%b %Y')
                next_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            elif period == 'yearly':
                key = current_date.strftime('%Y')
                next_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:  # weekly
                key = f"Week {current_date.isocalendar()[1]}, {current_date.year}"
                next_date = current_date + timedelta(days=7)
            
            filled_data[key] = period_data.get(key, 0)
            current_date = next_date
        
        return filled_data
    
    def _get_category_spending(self, user, start_date, end_date):
        """Get spending by category using ExpenseItems"""
        expense_items = ExpenseItems.objects.filter(
            domain_user_id=user.domain_user_id,
            expense_done='YES',
            expense_id__status='ACTIVE',
            category_id__isnull=False
        ).values('category_id', 'price', 'date_of_expense')
        # Filter expenses by date range and collect category IDs
        category_ids = set()
        filtered_expenses = []
        for item in expense_items:
            date_value = item.get('date_of_expense')
            if not date_value:
                continue
                
            try:
                # Handle both string and date objects
                if isinstance(date_value, str):
                    date_obj = datetime.strptime(date_value, '%Y-%m-%d').date()
                else:  # Assume it's already a date object
                    date_obj = date_value
                if start_date <= date_obj <= end_date:
                    filtered_expenses.append(item)
                    category_ids.add(item['category_id'])
            except (ValueError, TypeError):
                continue
        # Get categories
        categories = Categories.objects.filter(
            id__in=category_ids,
            domain_user_id=user.domain_user_id,
        ).values('id', 'name')

        # Calculate totals
        category_totals = {}
        for item in filtered_expenses:
            cat_id = item['category_id']
            amount = float(item.get('price', 0))
            category_totals[cat_id] = category_totals.get(cat_id, 0) + amount

        # Prepare response
        result = {
            'labels': [],
            'data': [],
            'colors': [],
            'category_ids': []
        }

        category_info = {cat['id']: cat for cat in categories}
        for cat_id, total in category_totals.items():
            if cat_id in category_info:
                cat = category_info[cat_id]
                result['labels'].append(cat['name'])
                result['data'].append(total)
                result['colors'].append(self._generate_random_color())
                result['category_ids'].append(cat_id)

        return result

class LatestTransactionsAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def _parse_date(self, date_value):
        """Helper to parse date from string or return date object"""
        if isinstance(date_value, str):
            try:
                return datetime.strptime(date_value, '%Y-%m-%d').date()
            except ValueError:
                return date.min
        elif isinstance(date_value, date):
            return date_value
        return date.min

    def get(self, request):
        user = request.user
        limit = int(request.query_params.get('limit', 4))

        # Get latest incomes
        incomes = Incomes.objects.filter(
            domain_user_id=user.domain_user_id,
            status='ACTIVE'
        ).order_by('-created_at')[:limit].annotate(
            transaction_type=Value('income', output_field=CharField()),
            description_field=Coalesce(
                F('source'), 
                F('description'),
                output_field=CharField()
            ),
        ).values(
            'id',
            'transaction_type',
            'description_field',
            'amount',
            'date_of_received',
        )

        # Get latest expenses
        expenses = ExpenseItems.objects.filter(
            domain_user_id=user.domain_user_id,
            expense_id__status='ACTIVE',
            expense_done='YES'
        ).select_related(
            'category_id',
            'expense_id'
        ).order_by('-date_of_expense')[:limit].annotate(
            transaction_type=Value('expense', output_field=CharField()),
            category_name=Coalesce(
                F('category_id__name'), 
                Value('Uncategorized'),
                output_field=CharField()
            ),
            description=ExpressionWrapper(
                F('expense_id__description'),
                output_field=CharField()
            ),
        ).values(
            'id',
            'transaction_type',
            'description',
            'price',
            'date_of_expense',
            'category_name',
        )

        # Format and combine results
        transactions = []
        
        for income in incomes:
            transactions.append({
                'id': income['id'],
                'type': income['transaction_type'],
                'description': income['description_field'],
                'amount': float(income['amount']),
                'date': self._parse_date(income['date_of_received']),
                'category': None
            })
        
        for expense in expenses:
            transactions.append({
                'id': expense['id'],
                'type': expense['transaction_type'],
                'description': expense['description'],
                'amount': float(expense['price']),
                'date': self._parse_date(expense['date_of_expense']),
                'category': expense['category_name']
            })

        # Sort combined transactions by date (newest first)
        transactions.sort(key=lambda x: x['date'], reverse=True)
        
        return Response(transactions[:limit])

class FinancialSummaryAPIView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Get all active incomes (no date filtering)
        total_income = Incomes.objects.filter(
            domain_user_id=user.domain_user_id,
            status='ACTIVE'
        ).aggregate(total=Sum('amount'))['total'] or 0
        
        # Get all active expenses (no date filtering)
        total_expenses = ExpenseItems.objects.filter(
            domain_user_id=user.domain_user_id,
            expense_id__status='ACTIVE',
            expense_done='YES'
        ).aggregate(total=Sum('price'))['total'] or 0
        
        # Calculate balance
        total_balance = total_income - total_expenses
        
        # Prepare response data
        data = {
            'total_balance': float(total_balance),
            'total_income': float(total_income),
            'total_expenses': float(total_expenses),
        }
        
        # Serialize and return
        serializer = FinancialSummarySerializer(data)
        return Response(serializer.data)
