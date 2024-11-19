from django.shortcuts import render, HttpResponse, redirect
from django.conf import settings
from .models import ObesityData

import pandas as pd
from datetime import date

from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from .forms import obesityDisorderForm
from .models import UserProfile, userHistory
from django.contrib.auth.models import User
from django import forms


import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from django.utils import timezone
import json

from django.contrib import messages

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for CSS Properties
        self.fields['username'].widget.attrs.update({'class': 'col-md-10 form-control'})
        self.fields['email'].widget.attrs.update({'class': 'col-md-10 form-control'})
        self.fields['first_name'].widget.attrs.update({'class': 'col-md-10 form-control'})
        self.fields['last_name'].widget.attrs.update({'class': 'col-md-10 form-control'})
        self.fields['password1'].widget.attrs.update({'class': 'col-md-10 form-control'})
        self.fields['password2'].widget.attrs.update({'class': 'col-md-10 form-control'})

        self.fields['username'].help_text = '<span class="text-muted">Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.</span>'
        self.fields['email'].help_text = '<span class="text-muted">Required. Inform a valid email address.</span>'
        self.fields['password2'].help_text = '<span class="text-muted">Enter the same password as before, for verification.</span>'
        self.fields['password1'].help_text = '<span class="text-muted"><ul class="small"><li class="text-muted">Your password can not be too similar to your other personal information.</li><li class="text-muted">Your password must contain at least 8 characters.</li><li class="text-muted">Your password can not be a commonly used password.</li><li class="text-muted">Your password can not be entirely numeric.</li></ul></span>' 

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        
        try:
            if form.is_valid():
                form.save()
                return redirect('login')
        except:
            form = UserRegistrationForm()
            messages.error(request, "Something went wrong. Try again!")
            
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('profile')
        else:
            messages.error(request, "Something went wrong. Try again!")
    return render(request, 'login.html')

@login_required
def complete_profile(request):
    if UserProfile.objects.filter(user=request.user).exists():
        # If profile exists, redirect to the dashboard profile page
        return redirect('dashboard')
    
    if request.method == 'POST':
        user_profile = UserProfile.objects.create(
            user=request.user,
            dob=request.POST['dob'],
            gender=request.POST['gender'],
            height=request.POST['height'],
            weight=request.POST['weight'],
            profession=request.POST['profession']
        )
        return redirect('dashboard')
    return render(request, 'profile.html', {'user_name': request.user.first_name + " " + request.user.last_name})

@login_required
def user_dashboard(request):
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        user_profile = None

    return render(request, 'user_dashboard.html', {'user_name': request.user.first_name + " " + request.user.last_name, 
                                                'user_profile': user_profile, 
                                                'user_username': request.user.username
                                                })

@login_required
def health_prediction(request):
    return render(request, 'health_test.html', {'user_name': request.user.first_name + " " + request.user.last_name})

obesity_encoder = joblib.load('static/encoders/obesity_encoder.pkl')
obesity_output_encoder = joblib.load('static/encoders/obesity_output_encoder.pkl')
obesity_model = joblib.load('static/models/obesity_prediction.pkl')

@login_required
def obesity(request):
    user_data = UserProfile.objects.get(user=request.user)
    weight, height, bmi, gender = user_data.weight, user_data.height, user_data.bmi, user_data.gender.capitalize()
    age = date.today().year - user_data.dob.year - ((date.today().month, date.today().day) < (user_data.dob.month, user_data.dob.day))
    
    if request.method == 'POST':
        
        form = obesityDisorderForm(request.POST)
        if form.is_valid():
            activityLevel = form.cleaned_data['activityLevel']
            
            new_data = [[age, gender, height, weight, bmi, int(activityLevel)]]
            new_data[0][1] = obesity_encoder.transform(np.array(new_data[0][1]).reshape(-1, 1))[0][0]
            predicted_data = obesity_model.predict(new_data)
            predicted_output = obesity_output_encoder.inverse_transform((np.array(predicted_data)).reshape(-1, 1))[0][0]
            
            symp = [int(activityLevel)]
            
            my_instance = userHistory(
                user=request.user,
                test_type='Obesity Test',
                symptoms=json.dumps(symp),
                result=predicted_output,
                date=timezone.now()
            )
            my_instance.save()
            
            return render(request, 'obesity.html', {'age': age, 'user_data': user_data, 'form': form, 'prediction_result': predicted_output, 'user_name': request.user.first_name + " " + request.user.last_name})
    else:
        form = obesityDisorderForm()

    return render(request, 'obesity.html', {'age': age, 'user_data': user_data, 'form': form, 'user_name': request.user.first_name + " " + request.user.last_name})


@login_required
def report(request):
    user_data = userHistory.objects.last()
    
    user_info = {}
    user_info['test_type'] = user_data.test_type
    user_info['result'] = user_data.result
    user_info['date'] = user_data.date
    
    user_profile = UserProfile.objects.get(user=user_data.user)

    user_info['dob']= user_profile.dob
    user_info['gender'] = user_profile.gender
    user_info['height'] = user_profile.height
    user_info['weight'] = user_profile.weight
    user_info['profession'] = user_profile.profession
    
    given_list = user_data.get_symptoms()
    
    if user_data.test_type == 'Obesity Test':
        attributes = ['Activity Level (1-4)']
    
    attributes_values = {}
    for i in range(len(given_list)):
        attributes_values[attributes[i]] = given_list[i]
    
    if user_data.test_type == 'Obesity Test':
        advice = {
            'Normal weight': 'Maintain your healthy lifestyle habits, including balanced nutrition and regular exercise, to support overall well-being.',
            'Obese': 'Seek professional guidance to develop a personalized weight management plan focusing on sustainable changes in diet and physical activity.',
            'Overweight': 'Implement small, gradual changes such as portion control and incorporating more fruits and vegetables into your diet to achieve a healthier weight.',
            'Underweight': 'Consult with a healthcare provider to identify potential underlying causes and develop a nutrition plan to reach and maintain a healthy weight.'
        }
    
    user_info['advice'] = advice[user_data.result]
    
    return render(request, 'report.html', {'user_info': user_info, 'attributes_values': attributes_values, 'user_name': request.user.first_name + " " + request.user.last_name})


@login_required
def test_history(request):
    user_medical_history = userHistory.objects.filter(user=request.user)
    return render(request, 'test_history.html', {'user_name': request.user.first_name + " " + request.user.last_name,
                                                'user_medical_history': user_medical_history})

@login_required
def download_receipt(request):
    receipt = Receipt.objects.get(user=request.user)
    # Logic to generate/download receipt file
    return redirect('dashboard')  # Redirect to dashboard or any other page

@login_required
def user_logout(request):
    logout(request)
    return redirect('login')

def index(request):
    return render(request, 'index.html')

