# Generated by Django 5.0.6 on 2025-04-29 14:05

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
            name='RefValues',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('label', models.CharField(blank=True, max_length=255, null=True)),
                ('value', models.FloatField(blank=True, null=True)),
                ('min', models.FloatField(blank=True, null=True)),
                ('max', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='Finances',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('period_name', models.CharField(blank=True, max_length=255, null=True)),
                ('assets', models.FloatField(blank=True, null=True)),
                ('cash_cash_equivalents', models.FloatField(blank=True, null=True)),
                ('inventory', models.FloatField(blank=True, null=True)),
                ('total_assets', models.FloatField(blank=True, null=True)),
                ('current_liabilities', models.FloatField(blank=True, null=True)),
                ('total_liabilities', models.FloatField(blank=True, null=True)),
                ('share_holder_equity', models.FloatField(blank=True, null=True)),
                ('revenue', models.FloatField(blank=True, null=True)),
                ('cost_goods_sold', models.FloatField(blank=True, null=True)),
                ('operating_expense', models.FloatField(blank=True, null=True)),
                ('interest_expense', models.FloatField(blank=True, null=True)),
                ('net_income', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by_user_id', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='created_by_user_id_finance', to=settings.AUTH_USER_MODEL)),
                ('domain_user_id', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='domain_user_id_finance', to=settings.AUTH_USER_MODEL)),
                ('updated_by_user_id', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='updated_by_user_id_finance', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
