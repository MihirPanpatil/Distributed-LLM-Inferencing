import json
import requests
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import WorkerNode, ModelShard, InferenceRequest
from .forms import WorkerNodeForm, InferenceForm
import threading

def dashboard(request):
    """Main dashboard view showing system status."""
    active_nodes = WorkerNode.objects.filter(is_active=True).count()
    total_nodes = WorkerNode.objects.count()
    
    pending_requests = InferenceRequest.objects.filter(status='pending').count()
    processing_requests = InferenceRequest.objects.filter(status='processing').count()
    
    recent_requests = InferenceRequest.objects.order_by('-created_at')[:5]
    
    context = {
        'active_nodes': active_nodes,
        'total_nodes': total_nodes,
        'pending_requests': pending_requests,
        'processing_requests': processing_requests,
        'recent_requests': recent_requests,
    }
    
    return render(request, 'dashboard/dashboard.html', context)

def node_management(request):
    """View for managing worker nodes."""
    form = WorkerNodeForm()
    nodes = WorkerNode.objects.all()
    
    context = {
        'form': form,
        'nodes': nodes,
    }
    
    return render(request, 'dashboard/node_management.html', context)

def inference(request):
    """View for submitting inference requests."""
    form = InferenceForm()
    recent_requests = InferenceRequest.objects.order_by('-created_at')[:10]
    
    context = {
        'form': form,
        'recent_requests': recent_requests,
    }
    
    return render(request, 'dashboard/inference.html', context)

def node_status(request):
    """API endpoint to get the status of all worker nodes."""
    nodes = []
    
    for node in WorkerNode.objects.all():
        node_data = {
            'id': node.id,
            'hostname': node.hostname,
            'ip_address': node.ip_address,
            'port': node.port,
            'is_active': node.is_active,
            'last_heartbeat': node.last_heartbeat.isoformat() if node.last_heartbeat else None,
        }
        
        # Try to get node health information
        if node.is_active:
            try:
                response = requests.get(f"{node.get_url()}/health", timeout=2)
                if response.status_code == 200:
                    health_data = response.json()
                    node_data['resources'] = health_data.get('resources', {})
                    node_data['loaded_shards'] = health_data.get('loaded_shards', [])
            except Exception as e:
                node_data['error'] = str(e)
                # Mark node as inactive if we can't reach it
                if node.is_active:
                    node.is_active = False
                    node.save()
        
        nodes.append(node_data)
    
    return JsonResponse({'nodes': nodes})

@csrf_exempt
@require_POST
def add_node(request):
    """API endpoint to add a new worker node."""
    form = WorkerNodeForm(request.POST)
    
    if form.is_valid():
        node = form.save(commit=False)
        
        # Check if the node is reachable
        try:
            response = requests.get(f"{node.get_url()}/health", timeout=5)
            if response.status_code == 200:
                node.is_active = True
                node.last_heartbeat = timezone.now()
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f"Could not connect to node: {str(e)}"
            }, status=400)
        
        node.save()
        
        return JsonResponse({
            'status': 'success',
            'node_id': node.id,
            'hostname': node.hostname
        })
    else:
        return JsonResponse({
            'status': 'error',
            'errors': form.errors
        }, status=400)

@csrf_exempt
@require_POST
def remove_node(request, node_id):
    """API endpoint to remove a worker node."""
    node = get_object_or_404(WorkerNode, id=node_id)
    
    # Unload any model shards from this node
    for shard in node.shards.all():
        try:
            requests.post(
                f"{node.get_url()}/unload_model",
                json={'model_name': shard.model_name},
                timeout=5
            )
        except Exception:
            # Continue even if unloading fails
            pass
    
    node.delete()
    
    return JsonResponse({
        'status': 'success',
        'message': f"Node {node.hostname} removed successfully"
    })

@csrf_exempt
@require_POST
def submit_inference(request):
    """API endpoint to submit an inference request."""
    form = InferenceForm(request.POST)
    
    if form.is_valid():
        inference_request = form.save()
        
        # Start processing in a background thread
        threading.Thread(
            target=process_inference_request,
            args=(inference_request.id,)
        ).start()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Inference request submitted successfully',
            'request_id': inference_request.id
        })
    else:
        return JsonResponse({
            'status': 'error',
            'errors': form.errors
        }, status=400)

def inference_status(request, request_id):
    """API endpoint to check the status of an inference request."""
    inference_request = get_object_or_404(InferenceRequest, id=request_id)
    
    return JsonResponse({
        'id': inference_request.id,
        'status': inference_request.status,
        'model_name': inference_request.model_name,
        'prompt': inference_request.prompt,
        'result': inference_request.result,
        'error': inference_request.error,
        'created_at': inference_request.created_at.isoformat(),
        'completed_at': inference_request.completed_at.isoformat() if inference_request.completed_at else None
    })

def recent_inferences(request):
    """API endpoint to get recent inference requests."""
    recent = InferenceRequest.objects.order_by('-created_at')[:10]
    
    requests_data = []
    for req in recent:
        requests_data.append({
            'id': req.id,
            'model_name': req.model_name,
            'status': req.status,
            'created_at': req.created_at.isoformat(),
            'completed_at': req.completed_at.isoformat() if req.completed_at else None
        })
    
    return JsonResponse({'requests': requests_data})

def process_inference_request(request_id):
    """Background task to process an inference request."""
    inference_request = InferenceRequest.objects.get(id=request_id)
    inference_request.status = 'processing'
    inference_request.save()
    
    model_name = inference_request.model_name
    prompt = inference_request.prompt
    
    # Check if we have shards for this model
    shards = ModelShard.objects.filter(model_name=model_name, is_loaded=True)
    
    if shards.exists():
        # Distributed inference with shards
        try:
            # Group shards by node
            node_shards = {}
            for shard in shards:
                if shard.node.is_active:
                    if shard.node.id not in node_shards:
                        node_shards[shard.node.id] = {
                            'node': shard.node,
                            'shard_ids': []
                        }
                    node_shards[shard.node.id]['shard_ids'].append(shard.shard_id)
            
            if not node_shards:
                raise Exception("No active nodes with loaded shards found")
            
            # For simplicity, use the first node that has shards
            node_info = list(node_shards.values())[0]
            node = node_info['node']
            shard_ids = node_info['shard_ids']
            
            # Send inference request to the node
            response = requests.post(
                f"{node.get_url()}/inference",
                json={
                    'model_name': model_name,
                    'prompt': prompt,
                    'shard_ids': shard_ids,
                    'max_length': 100  # Default value, could be configurable
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result_data = response.json()
                if result_data.get('status') == 'success':
                    inference_request.mark_completed(result_data.get('result', ''))
                else:
                    inference_request.mark_failed(result_data.get('message', 'Unknown error'))
            else:
                inference_request.mark_failed(f"Node returned status code {response.status_code}")
        
        except Exception as e:
            inference_request.mark_failed(str(e))
    
    else:
        # Try to find a node that can load the complete model
        active_nodes = WorkerNode.objects.filter(is_active=True)
        
        if not active_nodes.exists():
            inference_request.mark_failed("No active worker nodes available")
            return
        
        # For simplicity, use the first active node
        # In a real system, you would select based on load, capabilities, etc.
        node = active_nodes.first()
        
        try:
            # First, try to load the model if not already loaded
            load_response = requests.post(
                f"{node.get_url()}/load_model",
                json={'model_name': model_name},
                timeout=300  # Loading can take time
            )
            
            if load_response.status_code != 200:
                inference_request.mark_failed(f"Failed to load model: {load_response.text}")
                return
            
            # Now run inference
            response = requests.post(
                f"{node.get_url()}/inference",
                json={
                    'model_name': model_name,
                    'prompt': prompt,
                    'max_length': 100  # Default value, could be configurable
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result_data = response.json()
                if result_data.get('status') == 'success':
                    inference_request.mark_completed(result_data.get('result', ''))
                else:
                    inference_request.mark_failed(result_data.get('message', 'Unknown error'))
            else:
                inference_request.mark_failed(f"Node returned status code {response.status_code}")
        
        except Exception as e:
            inference_request.mark_failed(str(e))

