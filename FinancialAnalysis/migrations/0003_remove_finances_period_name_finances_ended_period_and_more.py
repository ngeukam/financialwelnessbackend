# Generated by Django 5.0.6 on 2025-07-01 13:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('FinancialAnalysis', '0002_rename_assets_finances_current_assets_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='finances',
            name='period_name',
        ),
        migrations.AddField(
            model_name='finances',
            name='ended_period',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='finances',
            name='started_period',
            field=models.DateField(blank=True, null=True),
        ),
    ]
