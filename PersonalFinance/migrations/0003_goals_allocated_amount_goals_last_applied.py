# Generated by Django 5.0.6 on 2025-04-15 06:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PersonalFinance', '0002_goals_frequency'),
    ]

    operations = [
        migrations.AddField(
            model_name='goals',
            name='allocated_amount',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='goals',
            name='last_applied',
            field=models.DateField(blank=True, null=True),
        ),
    ]
