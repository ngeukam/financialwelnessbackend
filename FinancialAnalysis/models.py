from django.db import models

from UserServices.models import Users

# Create your models here.
class Finances(models.Model):
    id=models.AutoField(primary_key=True)
    started_period = models.DateField(null=True, blank=True)
    ended_period = models.DateField(null=True, blank=True)
    current_assets=models.FloatField(blank=True, null=True)
    total_assets=models.FloatField(blank=True, null=True)
    current_liabilities=models.FloatField(blank=True, null=True)
    total_liabilities=models.FloatField(blank=True, null=True)
    revenue=models.FloatField(blank=True, null=True)
    cost_goods_sold=models.FloatField(blank=True, null=True)
    operating_expense=models.FloatField(blank=True, null=True)
    interest_expense=models.FloatField(blank=True, null=True)
    net_income=models.FloatField(blank=True, null=True)
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_finance')
    created_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='created_by_user_id_finance')
    updated_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='updated_by_user_id_finance')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)

class RefValues(models.Model):
    id=models.AutoField(primary_key=True)
    sector=models.CharField(max_length=255, default="Construction, civil engineering, real estate")
    label=models.CharField(max_length=255, blank=True, null=True)
    value=models.CharField(max_length=255, blank=True, null=True)
    description=models.CharField(max_length=255, blank=True, null=True)
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)
    
'''
Ratio	Valeur idéale	Signification
Current Ratio (Ratio de liquidité générale)	1.2 – 1.8	Capacité à payer les dettes court terme avec les actifs courants.
Quick Ratio (Ratio de liquidité immédiate)	0.8 – 1.2	Mesure plus stricte (exclut les stocks).

Ratio	Valeur idéale	Signification
Debt-to-Equity (D/E)	1.0 – 2.0	Équilibre dette / capitaux propres.
Interest Coverage Ratio	> 3.0	Capacité à payer les intérêts de la dette.

Ratio	Valeur idéale	Signification
Inventory Turnover (Rotation des stocks)	4 – 6	Fréquence de renouvellement des stocks/an.
Accounts Receivable Turnover (Rotation des créances clients)	8 – 12	Efficacité à recouvrer les paiements.

Ratio	Valeur idéale	Signification
Gross Profit Margin (Marge brute)	15% – 25%	Profit après coûts directs.
Net Profit Margin (Marge nette)	5% – 10%	Profit après tous les coûts.

'''