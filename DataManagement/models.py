from django.db import models

from UserServices.models import Users

# Create your models here.
class ProcessingOptions(models.Model):
    id=models.AutoField(primary_key=True)
    currency=models.CharField(max_length=5, blank=True,null=True, default='XAF', choices=(('XAF','XAF'), ('EUR','EUR'), ('USD','USD') ))
    data_name=models.CharField(max_length=50, blank=True,null=True)
    date_format=models.CharField(max_length=50, blank=True,null=True)
    # automatic_categorization = models.BooleanField(default=True)
    delete_duplicate = models.BooleanField(default=True)
    merge_existing = models.BooleanField(default=False)
    # advanced_validation = models.BooleanField(default=False)
    domain_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='domain_user_id_processing_options')
    added_by_user_id=models.ForeignKey(Users,on_delete=models.CASCADE,blank=True,null=True,related_name='added_by_user_id_processing_options')
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.data_name or f"ProcessingOption #{self.id}"
    
class DataFile(models.Model):
    processing_option = models.ForeignKey(
        ProcessingOptions,
        on_delete=models.CASCADE,
        related_name='data_files'
    )
    file = models.FileField(upload_to='DataFiles/')  # or ImageField if only images
    processed_data = models.JSONField(blank=True, null=True)
    file_type = models.CharField(max_length=50, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)