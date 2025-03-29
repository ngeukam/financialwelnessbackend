from django.http import JsonResponse
from UserServices.models import ModuleUrls, UserPermissions
from rest_framework_simplejwt.authentication import JWTAuthentication
import re
from django.db.models import Q

class PermissionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        current_url = request.path
        should_skip = urlToSkip()

        if should_skip(current_url):
            return response  # Skip authentication

        jwt_auth=JWTAuthentication()
        try:
            user,token=jwt_auth.authenticate(request)
            if not user:
                return JsonResponse({'message':'Unauthorized'},status=401)
        except:
            return JsonResponse({'message':'Unauthorized'},status=401)
        
        # Skip Permission Logic for Super Admin and Top Domain Level User
        if user.role=='Super Admin' or user.domain_user_id.id==user.id:
            return response

        module=find_matching_module(current_url)
        
        if not module:
            return JsonResponse({'message':'Module not Exist'},status=400)
        

        permission=UserPermissions.objects.filter(user=user.id,module=module.module).first()
        if not permission or permission.is_permission==False:
            return JsonResponse({'message':'Permission Denied'},status=403)
        

        return response
    

def urlToSkip():
    modules = list(ModuleUrls.objects.filter(module__isnull=True).values_list('url', flat=True))
    def should_skip(url):
        return any(url.startswith(skip_url) for skip_url in modules)

    return should_skip


def find_matching_module(url):
    # regex_pattern=re.sub(r'\d+','[^\/]+',url.replace('/','\/'))
    regex_pattern = re.sub(r'\d+', '[^/]+', url.replace('/', '/'))

    print('Regex Pattern:', regex_pattern)
    match_patter=ModuleUrls.objects.filter(Q(url__iregex=f'^{regex_pattern}$')).first()
    return match_patter