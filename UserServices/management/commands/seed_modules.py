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
            {"key": "data", "module_name": "Data Management", "module_icon": "Wharehouse", "is_menu": True, "module_url": "manage/data", "parent_key": None},
            {"key": "users", "module_name": "Users Management", "module_icon": "attendance", "is_menu": True, "module_url": "", "parent_key": None},
                       
            {"key": "goal", "module_name": "Create Goal", "module_icon": "attendance", "is_menu": False, "module_url": "/create/goal ", "parent_key": "finance"},

            {"key": "wallet", "module_name": "Wallet", "module_icon": "Wallet", "is_menu": True, "module_url": "pf/wallet", "parent_key": "finance"},            
            {"key": "cat_mgmt", "module_name": "Categories Management", "module_icon": "Category", "is_menu": True, "module_url": "pf/manage/category", "parent_key": "finance"},
            {"key": "fin_mgmt", "module_name": "Finance Management", "module_icon": "Money", "is_menu": True, "module_url": "pf/manage/finance", "parent_key": "finance"},


            {"key": "users_list", "module_name": "Users List", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/users", "parent_key": "users"},
            {"key": "users_add", "module_name": "Add User", "module_icon": "Add", "is_menu": True, "module_url": "/form/users", "parent_key": "users"},
            
            {"key": "get_menus", "module_name": "getMenus", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_key": None},
            {"key": "transactions", "module_name": "Transactions", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_key": None},
            {"key": "products", "module_name": "Products", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_key": None},
            {"key": "auth", "module_name": "Auth", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_key": None},
            {"key": "inventory", "module_name": "Inventory", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_key": None},
            {"key": "get_form", "module_name": "getForm", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_key": None},
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
            {"module": modules["dashboard"], "url": "/"},
            {"module": modules["finance"], "url": ""},
            {"module": modules["data"], "url": ""},
            {"module": modules["users"], "url": ""},
            {"module": modules["goal"], "url": ""},
            {"module": modules["cat_mgmt"], "url": ""},
            {"module": modules["fin_mgmt"], "url": ""},
            {"module": modules["wallet"], "url": ""},
            {"module": modules["users_list"], "url": ""},
            {"module": modules["users_add"], "url": ""},
            {"module": modules["get_menus"], "url": "api/v1/getMenus/"},
            {"module": modules["transactions"], "url": "api/v1/transactions/"},
            {"module": modules["products"], "url": "api/v1/products/"},
            {"module": modules["auth"], "url": "api/v1/auth/"},
            {"module": modules["inventory"], "url": "api/v1/inventory/"},
            {"module": modules["get_form"], "url": "api/v1/getForm/"},
        ]




        # Create or update module URLs
        for data in module_urls_data:
            ModuleUrls.objects.create(**data)
        
        self.stdout.write(self.style.SUCCESS('Module URLs created successfully.'))