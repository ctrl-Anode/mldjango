from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name='home'),
    path("register", views.register, name="register"),
    path("login", views.user_login, name="login"),
    path("profile", views.complete_profile, name="profile"),
    path("dashboard", views.user_dashboard, name="dashboard"),
    path("health_prediction", views.health_prediction, name="health_prediction"),
    path("obesity", views.obesity, name="obesity"),
    path("report", views.report, name="report"),
    path("test_history", views.test_history, name="test_history"),
    path('logout', views.user_logout, name='logout'),
]