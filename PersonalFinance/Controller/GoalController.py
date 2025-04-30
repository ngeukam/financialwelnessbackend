from EcommerceInventory.Helpers import CommonListAPIMixin, CustomPageNumberPagination, getDynamicFormFields, renderResponse
from rest_framework import generics
from datetime import date, datetime, timedelta
from django.utils import timezone
from rest_framework import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from rest_framework_simplejwt.authentication import JWTAuthentication
from PersonalFinance.models import Goals, Wallet
from django.db.models import (
    FloatField,
    Sum,
)

class GoalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Goals
        fields = "__all__"
    
    def create(self, validated_data):
        return Goals.objects.create(**validated_data)
    
    def update(self, instance, validated_data):
        return super().update(instance, validated_data)


    
class GoalCreateAPIView(generics.CreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, id=None):
        goal = Goals.objects.filter(
            domain_user_id=request.user.domain_user_id.id, 
            id=id
        ).first() if id else Goals()
        
        goal_fields = getDynamicFormFields(
            goal, 
            request.user.domain_user_id.id, 
            skip_fields=['status', 'reached', 'closed', 'allocated_amount', 'last_applied']
        )
        
        return renderResponse(
            data={'goalFields': goal_fields},
            message='Goal Fields',
            status=200
        )

    def post(self, request, id=None):
        data = request.data.copy()
        data.update({
            'created_by_user_id': request.user.id,
            'domain_user_id': request.user.domain_user_id.id
        })
        
        # Validation: Check budget amount
        try:
            budget = float(data.get('budget', 0))
            if budget <= 0:
                return renderResponse(
                    data={'budget': budget},
                    message='Budget must be positive',
                    status=400
                )
            percentage = float(data.get('percentage', 0))
            if percentage <= 0:
                return renderResponse(
                    data={'percentage': percentage},
                    message='percentage must be positive',
                    status=400
                )
                
            # Validate date range
            begin_date = datetime.strptime(data.get('begin_date'), '%Y-%m-%d').date()
            end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d').date()
            
            if begin_date >= end_date:
                return renderResponse(
                    data={'begin_date': begin_date, 'end_date': end_date},
                    message='End date must be after begin date',
                    status=400
                )
                
        except (ValueError, TypeError) as e:
            return renderResponse(
                data={'error': str(e)},
                message='Invalid value',
                status=400
            )
        except KeyError as e:
            return renderResponse(
                data={'error': f'Missing required field: {str(e)}'},
                message='Missing required field',
                status=400
            )

        # Create/update logic
        if id:
            goal = Goals.objects.filter(
                domain_user_id=request.user.domain_user_id.id,
                id=id,
                status='ACTIVE'
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
                    message='Unable to update a completed goal.',
                    status=404
                )
            data['updated_by_user_id'] = request.user.id
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

def is_application_due(goal, today):
    if not goal.last_applied:
        return True
    delta = (today - goal.last_applied).days
    if goal.frequency == 'DAYLY' and delta >= 1:
        return True
    elif goal.frequency == 'WEEKLY' and delta >= 7:
        return True
    elif goal.frequency == 'MONTHLY' and delta >= 30:
        return True
    elif goal.frequency == 'ANNUALY' and delta >= 365:
        return True
    return False

class ApplyWalletToGoalsAPIView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request):
        user = request.user
        today = date.today()

        try:
            wallet = Wallet.objects.get(user=user)
        except Wallet.DoesNotExist:
            return Response({"error": "The user's wallet could not be found."}, status=404)

        if wallet.current_balance <= 0:
            return Response({"message": "No balance available in the wallet."}, status=200)

        priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        goals = Goals.objects.filter(
            domain_user_id=user,
            status='ACTIVE',
            closed='NO'
        )
        goals = sorted(goals, key=lambda g: priority_order.get(g.priority, 3))

        total_applied = 0
        goal_updates = []

        for goal in goals:
            if goal.closed == 'YES' or goal.reached == 'YES':
                continue

            if goal.allocated_amount >= goal.budget or today >= goal.end_date:
                goal.closed = 'YES'
                goal.save()
                continue
            
            if goal.allocated_amount >= goal.budget:
                goal.reached = 'YES'
                goal.save()
                continue

            if not is_application_due(goal, today):
                continue

            amount_to_apply = (goal.percentage / 100) * wallet.current_balance
            remaining_budget = goal.budget - goal.allocated_amount
            apply_amount = min(amount_to_apply, remaining_budget)

            if apply_amount > 0:
                goal.allocated_amount += apply_amount
                goal.last_applied = today
                total_applied += apply_amount

                if goal.allocated_amount >= goal.budget or today >= goal.end_date:
                    goal.closed = 'YES'
                    
                if goal.allocated_amount >= goal.budget:
                    goal.reached = 'YES'

                goal.save()
                goal_updates.append({
                    "goal_id": goal.id,
                    "applied_amount": round(apply_amount, 2),
                    "new_allocated_amount": round(goal.allocated_amount, 2),
                    "closed": goal.closed,
                    "frequency": goal.frequency
                })

        wallet.current_balance -= total_applied
        wallet.save()

        return Response({
            "message": "Amounts successfully applied by frequency.",
            "total_applied": round(total_applied, 2),
            "wallet_balance": round(wallet.current_balance, 2),
            "details": goal_updates
        }, status=status.HTTP_200_OK)

class GoalListSerializer(serializers.ModelSerializer):
    progression = serializers.SerializerMethodField()
    allocated_amount = serializers.SerializerMethodField()

    class Meta:
        model = Goals
        fields = [
            'id', 'description', 'budget', 'priority', 'last_applied',
            'status', 'reached', 'begin_date', 'end_date',
            'progression', 'allocated_amount', 'percentage', 'closed', 'frequency'
        ]

    def get_allocated_amount(self, obj):
        return round(obj.allocated_amount or 0, 2)

    def get_progression(self, obj):
        if obj.budget and obj.allocated_amount:
            return round((obj.allocated_amount / obj.budget) * 100, 2)
        return 0.0


class GoalListAPIView(generics.ListAPIView):
    serializer_class = GoalListSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPageNumberPagination

    def get_queryset(self):
        # Store calculated metrics in the view instance for later use
        user = self.request.user
        now = timezone.now()
        
        # Calculate date ranges
        first_day_current_month = now.replace(day=1)
        last_day_last_month = first_day_current_month - timezone.timedelta(days=1)
        
        # Calculate and store metrics
        current_month_achieved = Goals.objects.filter(
            domain_user_id=user.domain_user_id.id,
            reached='YES',
            created_at__year=now.year,
            created_at__month=now.month
        ).count()
        last_month_achieved = Goals.objects.filter(
            domain_user_id=user.domain_user_id,
            reached='YES',
            created_at__year=last_day_last_month.year,
            created_at__month=last_day_last_month.month
        ).count()
        total_budget_achieved = Goals.objects.filter(
            domain_user_id=user.domain_user_id,
            reached='YES',
        ).aggregate(total=Sum('budget'))['total'] or 0

        total_goals = Goals.objects.filter(
            domain_user_id=user.domain_user_id
        ).count()
        
        self.extra_data = {
            'current_month_achieved': current_month_achieved,
            'last_month_achieved': last_month_achieved,
            'total_goals': total_goals,
            'total_budget_achieved':total_budget_achieved
        }
        # Return the base queryset
        return Goals.objects.filter(domain_user_id=user.domain_user_id)
    def finalize_response(self, request, response, *args, **kwargs):
            response = super().finalize_response(request, response, *args, **kwargs)
            if hasattr(self, 'extra_data'):
                response.data.update(self.extra_data)
            return response

    @CommonListAPIMixin.common_list_decorator(GoalListSerializer)
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)