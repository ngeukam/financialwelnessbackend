from django.urls import path

from PersonalFinance.Controller.GoalController import ApplyWalletToGoalsAPIView, GoalCreateAPIView, GoalListAPIView
from PersonalFinance.Controller.OverviewController import ExpenseSummaryView, GoalsBudgetSummaryView, IncomeSummaryView
from PersonalFinance.Controller.IncomeController import CreateIncomeView, IncomeListView
from PersonalFinance.Controller.ExpenseController import CreateExpenseView, ExpenseListView
from PersonalFinance.Controller.WalletController import FinancialChartsAPIView, FinancialSummaryAPIView, LatestTransactionsAPIView
from PersonalFinance.Controller.CategoryController import CategoryListView


urlpatterns = [
    path('categories/',CategoryListView.as_view(),name='category_list'),
    path('financial-charts/', FinancialChartsAPIView.as_view(), name='financial_charts'),
    path('latest-transactions/', LatestTransactionsAPIView.as_view(), name='latest_transactions'),
    path('financial-summary/', FinancialSummaryAPIView.as_view(), name='financial_summary'),
    
    path('goal/',GoalCreateAPIView.as_view(),name='goal'),
    path('goal/<str:id>/',GoalCreateAPIView.as_view(),name='goal_detail'),
    path('goals-apply-wallet/', ApplyWalletToGoalsAPIView.as_view(), name='apply_wallet_to_goals'),
    path('goals/', GoalListAPIView.as_view(), name='goals'),

    path('goals-summary/',GoalsBudgetSummaryView.as_view(),name='goals_summary'),

    path('expense/',CreateExpenseView.as_view(),name='expense'),
    path('expense/<str:id>/',CreateExpenseView.as_view(),name='expense_detail'),
    path('expenses/',ExpenseListView.as_view(),name='expenses_list'),
    path('expenses-summary/',ExpenseSummaryView.as_view(),name='expenses_summary'),
    
    path('incomes/',IncomeListView.as_view(),name='incomes_list'),
    path('income/',CreateIncomeView.as_view(),name='expense'),
    path('income/<str:id>/',CreateIncomeView.as_view(),name='expense_detail'),
    path('incomes-summary/',IncomeSummaryView.as_view(),name='incomes_summary'),

]