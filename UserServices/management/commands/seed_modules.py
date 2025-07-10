from django.core.management.base import BaseCommand
from django.db import connection

from UserServices.models import ModuleUrls, Modules

class Command(BaseCommand):
    help = 'Resets the database and seeds the Modules and ModuleUrls models'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.WARNING('Resetting database...'))
        
        # Supprimer les anciennes données
        ModuleUrls.objects.all().delete()
        Modules.objects.all().delete()
        
        # Réinitialiser les séquences d'ID
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='userservices_modules';")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='userservices_moduleurls';")
        
        self.stdout.write(self.style.SUCCESS('Database reset completed.'))
        # Define module data
        modules_data = [
            {"key": "dashboard", "module_name": "Dashboard", "module_icon": "Dashboard", "is_menu": True, "module_url": "", "parent_key": None},
            {"key": "finance", "module_name": "Personal Finance", "module_icon": "Finance", "is_menu": True, "module_url": "", "parent_key": None},
            {"key": "finance_analysis", "module_name": "Financial Analysis", "module_icon": "FinanceAnalysis", "is_menu": True, "module_url": "/overview/financial-analysis", "parent_key": None},

            {"key": "data", "module_name": "Data Management", "module_icon": "Warehouse", "is_menu": True, "module_url": "/manage/data", "parent_key": None},
            
            {"key": "risk_management", "module_name": "Risk Management", "module_icon": "Finance", "is_menu": True, "module_url": "", "parent_key": None},
            {"key": "credit_risk", "module_name": "Credit Risk", "module_icon": "Finance", "is_menu": True, "module_url": "/manage/credit-risk", "parent_key": "risk_management"},
            {"key": "market_risk", "module_name": "Market Risk", "module_icon": "Finance", "is_menu": True, "module_url": "/manage/market-risk", "parent_key": "risk_management"},
            # {"key": "early_warning", "module_name": "Early Warning", "module_icon": "Finance", "is_menu": True, "module_url": "/manage/early-warning", "parent_key": "risk_management"},
            {"key": "smart_analysis", "module_name": "Smart Analysis", "module_icon": "Finance", "is_menu": True, "module_url": "/manage/smart-analysis", "parent_key": "risk_management"},


            {"key": "goal", "module_name": "Create Goal", "module_icon": "attendance", "is_menu": False, "module_url": "/create/goal ", "parent_key": "finance"},

            {"key": "wallet", "module_name": "Wallet", "module_icon": "Wallet", "is_menu": True, "module_url": "/pf/wallet", "parent_key": "finance"},            
            {"key": "fin_mgmt", "module_name": "Finance Management", "module_icon": "Money", "is_menu": True, "module_url": "/pf/manage/finance", "parent_key": "finance"},

        ]


        modules = {}

        for data in modules_data:
            key = data.pop("key")
            parent_key = data.pop("parent_key")
            parent_instance = modules.get(parent_key) if parent_key else None
            module = Modules.objects.create(parent_id=parent_instance, **data)
            modules[key] = module


        
        self.stdout.write(self.style.SUCCESS('Modules created successfully.'))

        # Define module URLs
        module_urls_data = [
            {"module": None, "url": "/api/v1/getMenus/"},
            {"module": None, "url": "/api/v1/getForm/"},
            {"module": None, "url": "/api/v1/auth/login/"},
            {"module": None, "url": "/api/v1/auth/signup/"},
        ]




        # Create or update module URLs
        for data in module_urls_data:
            ModuleUrls.objects.create(**data)
        
        self.stdout.write(self.style.SUCCESS('Module URLs created successfully.'))