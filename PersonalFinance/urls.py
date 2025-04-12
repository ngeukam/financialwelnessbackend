from django.urls import path

from PersonalFinance.Controller.OverviewController import ExpenseSummaryView, GoalsBudgetSummaryView, IncomeSummaryView
from PersonalFinance.Controller.IncomeController import IncomeListView
from PersonalFinance.Controller.ExpenseController import CreateExpenseView, ExpenseListView
from PersonalFinance.Controller.GoalController import CreateGoalView, GoalListView
from PersonalFinance.Controller.WalletController import FinancialChartsAPIView, FinancialSummaryAPIView, LatestTransactionsAPIView
from PersonalFinance.Controller.CategoryController import CategoryListView


urlpatterns = [
    path('categories/',CategoryListView.as_view(),name='category_list'),
    path('financial-charts/', FinancialChartsAPIView.as_view(), name='financial_charts'),
    path('latest-transactions/', LatestTransactionsAPIView.as_view(), name='latest_transactions'),
    path('financial-summary/', FinancialSummaryAPIView.as_view(), name='financial_summary'),
    path('goal/',CreateGoalView.as_view(),name='goal'),
    path('goal/<str:id>/',CreateGoalView.as_view(),name='goal_detail'),
    path('goals-list/',GoalListView.as_view(),name='goal_list'),
    
    path('expense/',CreateExpenseView.as_view(),name='expense'),
    path('expense/<str:id>/',CreateExpenseView.as_view(),name='expense_detail'),
    path('expenses-list/',ExpenseListView.as_view(),name='expenses_list'),
    path('incomes-list/',IncomeListView.as_view(),name='incomes_list'),
    
    path('incomes-summary/',IncomeSummaryView.as_view(),name='incomes_summary'),
    path('expenses-summary/',ExpenseSummaryView.as_view(),name='expenses_summary'),
    path('goals-summary/',GoalsBudgetSummaryView.as_view(),name='goals_summary'),

]