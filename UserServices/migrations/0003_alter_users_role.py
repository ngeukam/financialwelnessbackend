# Generated by Django 5.0.6 on 2025-04-15 23:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('UserServices', '0002_remove_users_addition_details_remove_users_pincode_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='users',
            name='role',
            field=models.CharField(blank=True, choices=[('Admin', 'Admin'), ('Customer', 'Customer'), ('Staff', 'Staff')], default='Admin', max_length=50, null=True),
        ),
    ]
