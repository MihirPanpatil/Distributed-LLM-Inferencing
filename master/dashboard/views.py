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

def node_management(request):
    """View for managing worker nodes."""
    form = WorkerNodeForm()
    nodes = WorkerNode.objects.all()
    
    # Check for status messages from previous operations
    status_message = request.session.pop('status_message', None)
    status_type = request.session.pop('status_type', 'info')
    
    context = {
        'form': form,
        'nodes': nodes,
        'status_message': status_message,
        'status_type': status_type,
    }
    
    return render(request, 'dashboard/node_management.html', context)

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
                response = requests.get(f"{node.get_url()}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    node_data['resources'] = health_data.get('resources', {})
                    node_data['loaded_shards'] = health_data.get('loaded_shards', [])
                    # Update heartbeat time when successful health check
                    node.last_heartbeat = timezone.now()
                    node.save(update_fields=['last_heartbeat'])
            except requests.RequestException as e:
                node_data['error'] = f'Connection error: {str(e)}'
                logger.warning(f'Health check failed for node {node.hostname}: {str(e)}')
                # Mark node as inactive if we can't reach it
                if node.is_active:
                    node.is_active = False
                    node.save(update_fields=['is_active'])
        
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
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Node returned status code {response.status_code}. Response: {response.text}'
                }, status=400)
        except requests.RequestException as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Could not connect to node: {str(e)}. Please check the hostname, IP, and port.'
            }, status=400)
        except Exception as e:
            logger.error(f'Unexpected error adding node: {str(e)}')
            return JsonResponse({
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}'
            }, status=500)
        
        node.save()
        
        # Store success message in session for UI feedback
        request.session['status_message'] = f'Node {node.hostname} ({node.ip_address}) added successfully'
        request.session['status_type'] = 'success'
        
        logger.info(f'Node {node.hostname} ({node.ip_address}) added successfully')
        
        return JsonResponse({
            'status': 'success',
            'node_id': node.id,
            'hostname': node.hostname,
            'message': f'Node {node.hostname} added successfully'
        })
    else:
        # Provide detailed form validation errors
        error_details = {field: errors for field, errors in form.errors.items()}
        logger.warning(f'Node addition failed due to form validation: {error_details}')
        return JsonResponse({
            'status': 'error',
            'message': 'Form validation failed. Please correct the errors and try again.',
            'errors': error_details
        }, status=400)

@csrf_exempt
@require_POST
def remove_node(request, node_id):
    """API endpoint to remove a worker node."""
    try:
        node = get_object_or_404(WorkerNode, id=node_id)
        hostname = node.hostname
        
        # Unload any model shards from this node
        unload_errors = []
        for shard in node.shards.all():
            try:
                response = requests.post(
                    f"{node.get_url()}/unload_model",
                    json={'model_name': shard.model_name},
                    timeout=10
                )
                if response.status_code != 200:
                    unload_errors.append(f"Failed to unload shard {shard.shard_id} of model {shard.model_name}: {response.text}")
            except requests.RequestException as e:
                unload_errors.append(f"Connection error while unloading shard {shard.shard_id}: {str(e)}")
            except Exception as e:
                unload_errors.append(f"Unexpected error unloading shard {shard.shard_id}: {str(e)}")
        
        # Delete the node
        node.delete()
        
        # Record success/warning message for UI feedback
        if unload_errors:
            message = f"Node {hostname} removed, but with warnings: {'; '.join(unload_errors)}"
            request.session['status_type'] = 'warning'
            logger.warning(message)
        else:
            message = f"Node {hostname} removed successfully"
            request.session['status_type'] = 'success'
            logger.info(message)
        
        request.session['status_message'] = message
        
        return JsonResponse({
            'status': 'success',
            'message': message,
            'warnings': unload_errors if unload_errors else None
        })
    
    except Exception as e:
        error_message = f"Failed to remove node: {str(e)}"
        logger.error(error_message)
        request.session['status_message'] = error_message
        request.session['status_type'] = 'error'
        
        return JsonResponse({
            'status': 'error',
            'message': error_message
        }, status=500)

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
        
        # Store success message in session for UI feedback
        request.session['status_message'] = 'Inference request submitted successfully'
        request.session['status_type'] = 'success'
        
        logger.info(f'Inference request {inference_request.id} submitted successfully')
        
        return JsonResponse({
            'status': 'success',
            'message': 'Inference request submitted successfully',
            'request_id': inference_request.id
        })
    else:
        # Provide detailed form validation errors
        error_details = {field: errors for field, errors in form.errors.items()}
        logger.warning(f'Inference submission failed due to form validation: {error_details}')
        
        return JsonResponse({
            'status': 'error',
            'message': 'Form validation failed. Please correct the errors and try again.',
            'errors': error_details
        }, status=400)

def inference_status(request, request_id):
    """API endpoint to check the status of an inference request."""
    try:
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
    except Exception as e:
        logger.error(f'Error getting inference status: {str(e)}')
        return JsonResponse({
            'status': 'error',
            'message': f'Error retrieving inference status: {str(e)}'
        }, status=500)

def recent_inferences(request):
    """API endpoint to get recent inference requests."""
    try:
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
    except Exception as e:
        logger.error(f'Error getting recent inferences: {str(e)}')
        return JsonResponse({
            'status': 'error',
            'message': f'Error retrieving recent inferences: {str(e)}'
        }, status=500)

def process_inference_request(request_id):
    """Background task to process an inference request."""
    try:
        inference_request = InferenceRequest.objects.get(id=request_id)
        inference_request.status = 'processing'
        inference_request.save(update_fields=['status'])
        
        logger.info(f'Processing inference request {request_id}')
        
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
                
                logger.info(f'Using node {node.hostname} with shards {shard_ids} for request {request_id}')
                
                # Send inference request to the node
                response = requests.post(
                    f"{node.get_url()}/inference",
                    json={
                        'model_name': model_name,
                        'prompt': prompt,
                        'shard_ids': shard_ids,
                        'max_length': 100,  # Default value, could be configurable
                        'timeout': 60  # Add timeout parameter
                    },
                    timeout=120  # Extended timeout for the HTTP request itself
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('status') == 'success':
                        inference_request.mark_completed(result_data.get('result', ''))
                        logger.info(f'Inference request {request_id} completed successfully')
                    else:
                        error_message = result_data.get('message', 'Unknown error')
                        inference_request.mark_failed(error_message)
                        logger.error(f'Inference request {request_id} failed: {error_message}')
                else:
                    error_message = f"Node returned status code {response.status_code}: {response.text}"
                    inference_request.mark_failed(error_message)
                    logger.error(error_message)
            
            except requests.RequestException as e:
                error_message = f"Connection error: {str(e)}"
                inference_request.mark_failed(error_message)
                logger.error(f'Inference request {request_id} failed: {error_message}')
            except Exception as e:
                error_message = str(e)
                inference_request.mark_failed(error_message)
                logger.error(f'Inference request {request_id} failed: {error_message}')
        
        else:
            # Try to find a node that can load the complete model
            active_nodes = WorkerNode.objects.filter(is_active=True)
            
            if not active_nodes.exists():
                inference_request.mark_failed("No active worker nodes available")
                logger.error(f'Inference request {request_id} failed: No active worker nodes available')
                return
            
            # For simplicity, use the first active node
            # In a real system, you would select based on load, capabilities, etc.
            node = active_nodes.first()
            
            try:
                logger.info(f'Loading model {model_name} on node {node.hostname} for request {request_id}')
                
                # First, try to load the model if not already loaded
                load_response = requests.post(
                    f"{node.get_url()}/load_model",
                    json={'model_name': model_name},
                    timeout=300  # Loading can take time
                )
                
                if load_response.status_code != 200:
                    error_message = f"Failed to load model: {load_response.text}"
                    inference_request.mark_failed(error_message)
                    logger.error(f'Inference request {request_id} failed: {error_message}')
                    return
                
                logger.info(f'Model {model_name} loaded successfully on node {node.hostname}')
                
                # Now run inference
                response = requests.post(
                    f"{node.get_url()}/inference",
                    json={
                        'model_name': model_name,
                        'prompt': prompt,
                        'max_length': 100,  # Default value, could be configurable
                        'timeout': 60  # Add timeout parameter
                    },
                    timeout=120  # Extended timeout for the HTTP request itself
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('status') == 'success':
                        inference_request.mark_completed(result_data.get('result', ''))
                        logger.info(f'Inference request {request_id} completed successfully')
                    else:
                        error_message = result_data.get('message', 'Unknown error')
                        inference_request.mark_failed(error_message)
                        logger.error(f'Inference request {request_id} failed: {error_message}')
                else:
                    error_message = f"Node returned status code {response.status_code}: {response.text}"
                    inference_request.mark_failed(error_message)
                    logger.error(error_message)
            
            except requests.RequestException as e:
                error_message = f"Connection error: {str(e)}"
                inference_request.mark_failed(error_message)
                logger.error(f'Inference request {request_id} failed: {error_message}')
            except Exception as e:
                error_message = str(e)
                inference_request.mark_failed(error_message)
                logger.error(f'Inference request {request_id} failed: {error_message}')
    except Exception as e:
        logger.critical(f'Fatal error processing inference request {request_id}: {str(e)}')
        try:
            # Try to update the request status if possible
            InferenceRequest.objects.filter(id=request_id).update(
                status='failed',
                error=f'Fatal error: {str(e)}',
                completed_at=timezone.now()
            )
        except:
            pass  # If we can't update the database, we've already logged the error
