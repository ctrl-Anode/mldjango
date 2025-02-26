from django.db import models
from django.contrib.auth.models import User
import pandas as pd
from django.utils import timezone
import json
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import Group, Permission
from django.utils.translation import gettext as _


mental_disorder_df = pd.read_csv('static/mentalDisorder.csv')


from django.contrib.auth.models import AbstractUser


class userHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    test_type = models.CharField(max_length = 120)
    symptoms = models.CharField(max_length = 500)
    result = models.CharField(max_length = 120)
    date = models.DateField(default=timezone.now)
    
    def set_symptoms(self, symptoms_list):
        self.symptoms = json.dumps(symptoms_list)

    def get_symptoms(self):
        return json.loads(self.symptoms)

class UserProfile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dob = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=(('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')))
    height = models.FloatField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    profession = models.CharField(max_length=100, null=True, blank=True)

    @property
    def bmi(self):
        if self.height and self.weight:
            return round(self.weight / ((self.height / 100) ** 2), 2)
        return None

class obesityDisorder(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    activityLevel = models.CharField(max_length = 10, choices = (('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')))
    
class ObesityData(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length = 6)
    height = models.FloatField()
    weight = models.FloatField()
    bmi = models.FloatField()
    activityLevel = models.FloatField()
    ObesityCategory = models.CharField(max_length = 20)
    
    def __str__(self) -> str:
        return self.ObesityCategory