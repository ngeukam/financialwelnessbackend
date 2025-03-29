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
            {"module_name": "Dashboard", "module_icon": "Dashboard", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Inventaire", "module_icon": "Inventory", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Fournisseurs", "module_icon": "AccountCircle", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Achats", "module_icon": "Store", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Ventes", "module_icon": "Retail", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Emplacements", "module_icon": "Location", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Catégories", "module_icon": "Category", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Produits", "module_icon": "Redeem", "is_menu": True, "module_url": "", "parent_id": None},
            {"module_name": "Gestion des Utilisateurs", "module_icon": "attendance", "is_menu": True, "module_url": "", "parent_id": None},

            {"module_name": "Gestion inventaires", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/invetory", "parent_id": 2},
            {"module_name": "Ajouter un inventaire", "module_icon": "Add", "is_menu": True, "module_url": "/form/stock", "parent_id": 2},
             
            {"module_name": "Gestion des fournisseurs", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/suppliers", "parent_id": 3},
            {"module_name": "Ajouter un fournisseur", "module_icon": "Add", "is_menu": True, "module_url": "/form/users", "parent_id": 3},

            {"module_name": "Gestion des achats", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/purchaseorder", "parent_id": 4},
            {"module_name": "Ajouter un achat", "module_icon": "Add", "is_menu": True, "module_url": "/create/po", "parent_id": 4},

            {"module_name": "Gestion des ventes", "module_icon": "Dashboard", "is_menu": True, "module_url": "", "parent_id": 5},
            {"module_name": "Ajouter une vente", "module_icon": "Add", "is_menu": True, "module_url": "/form/sale", "parent_id": 5},

            {"module_name": "Gestion des emplacements", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/location", "parent_id": 6},
            {"module_name": "Ajouter un emplacement", "module_icon": "Add", "is_menu": True, "module_url": "/form/location", "parent_id": 6},
            
            {"module_name": "Gestion des catégories", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/categories", "parent_id": 7},
            {"module_name": "Ajouter une catégorie", "module_icon": "Add", "is_menu": True, "module_url": "/form/category", "parent_id": 7},
            
            {"module_name": "Gestion des produits", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/product", "parent_id": 8},
            {"module_name": "Ajouter un produit", "module_icon": "Add", "is_menu": True, "module_url": "/form/product", "parent_id": 8},
            
            {"module_name": "Gestion des utilisateurs", "module_icon": "Dashboard", "is_menu": True, "module_url": "/manage/users", "parent_id": 9},
            {"module_name": "Ajouter un utilisateur", "module_icon": "Add", "is_menu": True, "module_url": "/form/users", "parent_id": 9},
            
            {"module_name": "getMenus", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_id": None},
            {"module_name": "Transactions", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_id": None},
            {"module_name": "Products", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_id": None},
            {"module_name": "Auth", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_id": None},
            {"module_name": "Inventory", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_id": None},
            {"module_name": "getForm", "module_icon": "attendance", "is_menu": False, "module_url": "", "parent_id": None},
            
            {"module_name": "Gestion des modules", "module_icon": "attendance", "is_menu": True, "module_url": "", "parent_id": None},

        ]

        modules = {}  # Stocker les modules par ID pour les références des enfants

        for data in modules_data:
            parent_id = data.pop("parent_id")  # Récupérer l'ID du parent et le retirer du dictionnaire

            # Si un parent_id est défini, récupérer son instance
            parent_instance = modules.get(parent_id) if parent_id else None

            # Créer le module avec l'instance du parent
            module = Modules.objects.create(parent_id=parent_instance, **data)

            # Stocker l'instance dans le dictionnaire avec son ID
            modules[module.id] = module

        
        self.stdout.write(self.style.SUCCESS('Modules created successfully.'))

        # Define module URLs
        module_urls_data = [
            {"module": modules[1], "url": "/"},
            {"module": modules[2], "url": ""},
            {"module": modules[3], "url": ""},
            {"module": modules[4], "url": ""},
            {"module": modules[5], "url": ""},
            {"module": modules[6], "url": ""},
            {"module": modules[7], "url": ""},
            {"module": modules[8], "url": ""},
            {"module": modules[9], "url": ""},
            
            {"module": modules[10], "url": ""},
            {"module": modules[11], "url": ""},
            
            {"module": modules[12], "url": ""},
            {"module": modules[13], "url": ""},
            
            {"module": modules[14], "url": ""},
            {"module": modules[14], "url": ""},
            
            {"module": modules[16], "url": ""},
            {"module": modules[17], "url": ""},
            
            {"module": modules[18], "url": ""},
            {"module": modules[19], "url": ""},
            
            {"module": modules[20], "url": ""},
            {"module": modules[21], "url": ""},
            
            {"module": modules[22], "url": ""},
            {"module": modules[23], "url": ""},
            
            {"module": modules[24], "url": ""},
            {"module": modules[25], "url": ""},
            
            {"module": modules[26], "url": "api/v1/getMenus/"},
            {"module": modules[27], "url": "api/v1/transactions/"},
            {"module": modules[28], "url": "api/v1/products/"},
            {"module": modules[29], "url": "api/v1/auth/"},
            {"module": modules[30], "url": "api/v1/inventory/"},
            {"module": modules[31], "url": "api/v1/getForm/"},
            {"module": modules[32], "url": "api/v1/superAdminForm/*/"},
        ]

        # Create or update module URLs
        for data in module_urls_data:
            ModuleUrls.objects.create(**data)
        
        self.stdout.write(self.style.SUCCESS('Module URLs created successfully.'))