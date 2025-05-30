# Generated by Django 5.0.6 on 2025-04-09 21:08

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('DataManagement', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='processingoptions',
            name='added_by_user_id',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='added_by_user_id_processing_options', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='processingoptions',
            name='domain_user_id',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='domain_user_id_processing_options', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='datafile',
            name='processing_option',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='data_files', to='DataManagement.processingoptions'),
        ),
    ]
