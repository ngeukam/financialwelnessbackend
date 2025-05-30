"""
URL configuration for EcommerceInventory project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path

from EcommerceInventory.views import index,FileUploadViewInS3
from EcommerceInventory import settings
from UserServices.Controller.DynamicFormController import DynamicFormController
from UserServices.Controller.SuperAdminDynamicFormController import SuperAdminDynamicFormController
from UserServices.Controller.SidebarController import ModuleUrlsListAPIView, ModuleView
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/auth/', include('UserServices.urls')),
    path('api/v1/getForm/<str:modelName>/',DynamicFormController.as_view(),name='dynamicForm'),
    path('api/v1/getForm/<str:modelName>/<str:id>/',DynamicFormController.as_view(),name='dynamicForm'),
    path('api/v1/superAdminForm/<str:modelName>/',SuperAdminDynamicFormController.as_view(),name='superadmindynamicForm'),
    path('api/v1/moduleUrls/',ModuleUrlsListAPIView.as_view(),name='moduleUrls_superadmin'),
    path('api/v1/getMenus/',ModuleView.as_view(),name='sidebarmenu'),
    # path('api/v1/products/',include('ProductServices.urls')),
    # path('api/v1/inventory/',include('InventoryServices.urls')),
    # path('api/v1/orders/',include('OrderService.urls')),
    path('api/v1/uploads/',FileUploadViewInS3.as_view(),name='fileupload'),
    path('api/v1/datamanagement/', include('DataManagement.urls')),
    path('api/v1/personalfinance/', include('PersonalFinance.urls')),
    path('api/v1/assess/', include('CreditRisk.urls')),
]

if settings.DEBUG:
    urlpatterns+=static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)

urlpatterns+=[
    re_path(r'^(?:.*)/?$',index,name='index')
]