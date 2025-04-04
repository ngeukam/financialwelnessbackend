# Generated by Django 5.0.6 on 2025-03-26 21:27

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ProcessingOptions',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('currency', models.CharField(blank=True, choices=[('XAF', 'XAF'), ('EUR', 'EUR'), ('USD', 'USD')], default='XAF', max_length=5, null=True)),
                ('data_name', models.CharField(blank=True, max_length=50, null=True)),
                ('date_format', models.CharField(blank=True, max_length=50, null=True)),
                ('processing_options', models.JSONField(default={'advanced_validation': False, 'automatic_categorization': True, 'detect_duplicate': True, 'merge_existing': False})),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('added_by_user_id', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='added_by_user_id_processing_options', to=settings.AUTH_USER_MODEL)),
                ('domain_user_id', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='domain_user_id_processing_options', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='DataFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='DataFiles/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('processing_option', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='data_files', to='DataManagement.processingoptions')),
            ],
        ),
    ]
