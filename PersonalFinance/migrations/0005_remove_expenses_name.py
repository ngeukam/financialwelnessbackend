# Generated by Django 5.0.6 on 2025-04-10 20:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('PersonalFinance', '0004_alter_goalsitems_domain_user_id_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='expenses',
            name='name',
        ),
    ]
