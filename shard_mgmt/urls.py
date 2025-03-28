from django.urls import path
from . import views

app_name = 'shard_mgmt'

urlpatterns = [
    # Node management views
    path('nodes/', views.node_management, name='node_management'),
    
    # API endpoints
    path('api/nodes/status/', views.node_status, name='node_status'),
    path('api/nodes/add/', views.add_node, name='add_node'),
    path('api/nodes/remove/<int:node_id>/', views.remove_node, name='remove_node'),
    path('api/inference/submit/', views.submit_inference, name='submit_inference'),
    path('api/inference/status/<int:request_id>/', views.inference_status, name='inference_status'),
    path('api/inference/recent/', views.recent_inferences, name='recent_inferences'),
]
