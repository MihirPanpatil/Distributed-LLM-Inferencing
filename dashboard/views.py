import json
import requests
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from shard_mgmt.models import WorkerNode, ModelShard, InferenceRequest
from shard_mgmt.forms import InferenceForm
import logging

# Set up logging
logger = logging.getLogger(__name__)

def dashboard(request):
    """Main dashboard view showing system status."""
    active_nodes = WorkerNode.objects.filter(is_active=True).count()
    total_nodes = WorkerNode.objects.count()
    
    pending_requests = InferenceRequest.objects.filter(status='pending').count()
    processing_requests = InferenceRequest.objects.filter(status='processing').count()
    completed_requests = InferenceRequest.objects.filter(status='completed').count()
    
    recent_requests = InferenceRequest.objects.order_by('-created_at')[:5]
    
    context = {
        'active_nodes': active_nodes,
        'total_nodes': total_nodes,
        'pending_requests': pending_requests,
        'processing_requests': processing_requests,
        'completed_requests': completed_requests,
        'recent_requests': recent_requests,
    }
    
    return render(request, 'dashboard/dashboard.html', context)

def inference(request):
    """View for submitting inference requests."""
    form = InferenceForm()
    recent_requests = InferenceRequest.objects.order_by('-created_at')[:10]
    
    # Check for status messages from previous operations
    status_message = request.session.pop('status_message', None)
    status_type = request.session.pop('status_type', 'info')
    
    context = {
        'form': form,
        'recent_requests': recent_requests,
        'status_message': status_message,
        'status_type': status_type,
    }
    
    return render(request, 'dashboard/inference.html', context)
