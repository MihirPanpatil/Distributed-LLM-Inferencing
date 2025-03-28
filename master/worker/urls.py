from django.urls import path
from . import views

app_name = 'worker'

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('load_model/', views.load_model, name='load_model'),
    path('load_shard/', views.load_shard, name='load_shard'),
    path('unload_model/', views.unload_model, name='unload_model'),
    path('inference/', views.run_inference, name='run_inference'),
    path('ssh_setup/', views.setup_ssh_connection, name='ssh_setup'),
]
