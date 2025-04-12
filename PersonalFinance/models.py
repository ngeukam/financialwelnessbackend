from django.db import models
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from UserServices.models import Users
from django.db.models import Sum

# Create your models here.
class Categories(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=255)
    image=models.JSONField(blank=True,null=True)
    description=models.TextField()
    display_order=models.IntegerField(default=0)
    parent_id=models.ForeignKey('self',on_delete=models.CASCADE,blank=True,null=True)
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_finance_category')
    added_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='added_by_user_id_finance_category')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)

    def defaultkey():
        return "name"

class Expenses(models.Model):
    id=models.AutoField(primary_key=True)
    # name=models.CharField(max_length=255)
    description=models.TextField()
    image=models.JSONField(blank=True,null=True)
    status=models.CharField(max_length=255,choices=[('ACTIVE','ACTIVE'),('INACTIVE','INACTIVE')],default='ACTIVE')
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_expenses')
    created_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='created_by_user_id_expense')
    updated_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='updated_by_user_id_expense')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)
    
    def defaultkey():
        return "description"

class ExpenseItems(models.Model):
    id=models.AutoField(primary_key=True)
    price=models.FloatField()
    expense_id=models.ForeignKey(Expenses,on_delete=models.CASCADE,blank=True,null=True,related_name='expense_id_item')
    category_id=models.ForeignKey(Categories,on_delete=models.CASCADE,blank=True,null=True,related_name='category_id_expense_item')
    date_of_expense=models.DateField()
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_expense_item')
    # image=models.JSONField(blank=True,null=True)
    note=models.TextField(null=True, blank=True)
    expense_done=models.CharField(max_length=255,choices=[('YES','YES'),('NO','NO')],default='NO')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)
    
class Incomes(models.Model):
    id=models.AutoField(primary_key=True)
    source=models.CharField(max_length=255,blank=True,null=True)
    description=models.TextField()
    amount=models.FloatField()
    status=models.CharField(max_length=255,choices=[('ACTIVE','ACTIVE'),('INACTIVE','INACTIVE')],default='ACTIVE')
    frequency=models.CharField(max_length=255,choices=[('DAYLY','DAYLY'),('WEEKLY','WEEKLY'),('MONTHLY','MONTHLY'),('ANNUALY','ANNUALY')],default='MONTHLY')
    date_of_received=models.CharField(max_length=50, blank=True,null=True)
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_incomes')
    added_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='added_by_user_id_incomes')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)

class Wallet(models.Model):
    user = models.ForeignKey(Users, on_delete=models.CASCADE, related_name='wallets')
    current_balance = models.FloatField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    
    def update_balance(self):
        # Calculate total active incomes for the user
        total_income = Incomes.objects.filter(
            domain_user_id=self.user,
            status='ACTIVE'
        ).aggregate(Sum('amount'))['amount__sum'] or 0
        
        # Calculate total completed expenses for the user
        total_expenses = ExpenseItems.objects.filter(
            domain_user_id=self.user,
            expense_done='YES',
            expense_id__status='ACTIVE'  # Only count items from active expenses
        ).aggregate(Sum('price'))['price__sum'] or 0
        
        self.current_balance = total_income - total_expenses
        self.save()

@receiver([post_save, post_delete], sender=Incomes)
@receiver([post_save, post_delete], sender=Expenses)
def update_wallet(sender, instance, **kwargs):
    user = instance.domain_user_id
    wallet, created = Wallet.objects.get_or_create(user=user)
    wallet.update_balance()

class Goals(models.Model):
    id=models.AutoField(primary_key=True)
    # name=models.CharField(max_length=255)
    description=models.TextField()
    budget=models.FloatField()
    status=models.CharField(max_length=255,choices=[('ACTIVE','ACTIVE'),('INACTIVE','INACTIVE')],default='ACTIVE')
    reached=models.CharField(max_length=255,choices=[('YES','YES'),('NO','NO')],default='NO', help_text='Goal is reached ?')
    begin_date=models.DateField()
    end_date=models.DateField()
    priority=models.CharField(max_length=255,choices=[('HIGH','HIGH'), ('MEDIUM','MEDIUM'), ('LOW','LOW')],default='MEDIUM')
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_goals')
    created_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='created_by_user_id_goal')
    updated_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='updated_by_user_id_goal')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)

class GoalsItems(models.Model):
    id=models.AutoField(primary_key=True)
    goal_id=models.ForeignKey(Goals,on_delete=models.CASCADE,blank=True,null=True,related_name='goal_id_item')
    expense_id=models.ForeignKey(Expenses,on_delete=models.CASCADE,blank=True,null=True,related_name='goal_expense_id_item')
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_item')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)