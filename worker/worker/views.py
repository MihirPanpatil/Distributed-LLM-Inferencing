import os
import json
import time
import logging
import functools
from functools import wraps

import torch
import psutil
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/app/model_cache')
USE_GPU = os.environ.get('USE_GPU', '0') == '1'
AUTH_ENABLED = os.environ.get('AUTH_ENABLED', '0') == '1'
AUTH_KEY = os.environ.get('AUTH_KEY', '')

# Create model cache directory
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Determine device based on availability and preference
DEVICE = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'

# Global variables to track loaded models and shards
loaded_models = {}
loaded_tokenizers = {}
loaded_shards = {}

def require_auth(view_func):
    """Decorator to require authentication for endpoints."""
    @wraps(view_func)
    def decorated(request, *args, **kwargs):
        if not AUTH_ENABLED:
            return view_func(request, *args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f'Bearer {AUTH_KEY}':
            return JsonResponse({
                'status': 'error',
                'message': 'Unauthorized access'
            }, status=401)
        
        return view_func(request, *args, **kwargs)
    return decorated

@require_auth
def health_check(request):
    """Endpoint to check worker health and resource usage."""
    # Get system resource usage
    cpu_percent = psutil.cpu_percent() / 100.0
    memory = psutil.virtual_memory()
    memory_percent = memory.percent / 100.0
    
    # Check GPU usage if available
    gpu_percent = 0.0
    if torch.cuda.is_available() and USE_GPU:
        try:
            gpu_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() \
                if torch.cuda.max_memory_allocated() > 0 else 0.0
        except Exception:
            gpu_percent = 0.0
    
    # Get information about loaded shards
    shard_info = []
    for model_name, shards in loaded_shards.items():
        for shard_id, shard_data in shards.items():
            shard_info.append({
                'model_name': model_name,
                'shard_id': shard_id,
                'path': shard_data['path'],
                'metadata': shard_data['metadata']
            })
    
    return JsonResponse({
        'status': 'healthy',
        'resources': {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'gpu': gpu_percent,
            'gpu_available': torch.cuda.is_available() and USE_GPU,
            'device': DEVICE
        },
        'loaded_models': list(loaded_models.keys()),
        'loaded_tokenizers': list(loaded_tokenizers.keys()),
        'loaded_shards': shard_info
    })

@csrf_exempt
@require_POST
@require_auth
def load_model(request):
    """Endpoint to load a complete model."""
    try:
        data = json.loads(request.body)
        model_name = data.get('model_name')
        
        if not model_name:
            return JsonResponse({
                'status': 'error',
                'message': 'Model name is required'
            }, status=400)
        
        # Check if model is already loaded
        if model_name in loaded_models:
            return JsonResponse({
                'status': 'success',
                'message': f'Model {model_name} is already loaded'
            })
        
        # Load tokenizer
        if model_name not in loaded_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            loaded_tokenizers[model_name] = tokenizer
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        
        # Move model to appropriate device
        model = model.to(DEVICE)
        
        loaded_models[model_name] = model
        
        return JsonResponse({
            'status': 'success',
            'message': f'Model {model_name} loaded successfully on {DEVICE}'
        })
    
    except Exception as e:
        logger.error(f'Failed to load model: {str(e)}', exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to load model: {str(e)}'
        }, status=500)

@csrf_exempt
@require_POST
@require_auth
def load_shard(request):
    """Endpoint to load a model shard."""
    try:
        data = json.loads(request.body)
        model_name = data.get('model_name')
        shard_id = data.get('shard_id')
        shard_path = data.get('shard_path')
        
        if not all([model_name, shard_id, shard_path]):
            return JsonResponse({
                'status': 'error',
                'message': 'Model name, shard ID, and shard path are required'
            }, status=400)
        
        # Initialize the model_name entry in loaded_shards if it doesn't exist
        if model_name not in loaded_shards:
            loaded_shards[model_name] = {}
        
        # Check if shard is already loaded
        if shard_id in loaded_shards[model_name]:
            return JsonResponse({
                'status': 'success',
                'message': f'Shard {shard_id} of model {model_name} is already loaded'
            })
        
        # Load tokenizer if not already loaded
        if model_name not in loaded_tokenizers:
            tokenizer_path = os.path.join(os.path.dirname(shard_path), 'tokenizer')
            if os.path.exists(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            loaded_tokenizers[model_name] = tokenizer
        
        # Load metadata
        metadata_path = os.path.join(shard_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'model_name': model_name,
                'shard_id': shard_id,
                'num_shards': 1,
                'start_layer': 0,
                'end_layer': 0,
                'total_layers': 1
            }
        
        # Store shard information
        loaded_shards[model_name][shard_id] = {
            'path': shard_path,
            'metadata': metadata
        }
        
        return JsonResponse({
            'status': 'success',
            'message': f'Shard {shard_id} of model {model_name} loaded successfully'
        })
    
    except Exception as e:
        logger.error(f'Failed to load shard: {str(e)}', exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to load shard: {str(e)}'
        }, status=500)

@csrf_exempt
@require_POST
@require_auth
def unload_model(request):
    """Endpoint to unload a model."""
    try:
        data = json.loads(request.body)
        model_name = data.get('model_name')
        
        if not model_name:
            return JsonResponse({
                'status': 'error',
                'message': 'Model name is required'
            }, status=400)
        
        # Unload the model
        if model_name in loaded_models:
            # Delete the model to free up memory
            if DEVICE == 'cuda':
                loaded_models[model_name].cpu()
            del loaded_models[model_name]
        
        # Unload the tokenizer
        if model_name in loaded_tokenizers:
            del loaded_tokenizers[model_name]
        
        # Unload any shards
        if model_name in loaded_shards:
            del loaded_shards[model_name]
        
        # Force CUDA garbage collection
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Model {model_name} unloaded successfully'
        })
    
    except Exception as e:
        logger.error(f'Failed to unload model: {str(e)}', exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to unload model: {str(e)}'
        }, status=500)

@csrf_exempt
@require_POST
@require_auth
def run_inference(request):
    """Endpoint to run inference with a model."""
    try:
        data = json.loads(request.body)
        model_name = data.get('model_name')
        prompt = data.get('prompt')
        max_length = data.get('max_length', 100)
        shard_ids = data.get('shard_ids')
        timeout = data.get('timeout', 60)
        
        if not all([model_name, prompt]):
            return JsonResponse({
                'status': 'error',
                'message': 'Model name and prompt are required'
            }, status=400)
        
        # Set a timer to track execution time
        start_time = time.time()
        
        # Check if we're using shards
        if shard_ids:
            result = run_shard_inference(model_name, prompt, shard_ids, max_length)
        else:
            # Load the model if not already loaded
            if model_name not in loaded_models:
                # Try to load the model
                model_response = load_model_internal(model_name)
                if model_response.get('status') == 'error':
                    return JsonResponse(model_response, status=500)
            
            # Get the model and tokenizer
            model = loaded_models[model_name]
            tokenizer = loaded_tokenizers[model_name]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Check timeout
            if (time.time() - start_time) > timeout:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Inference preparation timed out'
                }, status=408)
            
            # Generate text
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.8,
            )
            
            # Decode the generated text
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check timeout again
            if (time.time() - start_time) > timeout:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Inference generation timed out'
                }, status=408)
        
        return JsonResponse({
            'status': 'success',
            'result': result,
            'execution_time': time.time() - start_time
        })
    
    except Exception as e:
        logger.error(f'Inference failed: {str(e)}', exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': f'Inference failed: {str(e)}'
        }, status=500)

def run_shard_inference(model_name, prompt, shard_ids, max_length):
    """Run inference using model shards."""
    # For simplicity, we'll just use the first shard for now
    shard_id = shard_ids[0]
    
    if shard_id not in loaded_shards.get(model_name, {}):
        raise ValueError(f"Shard {shard_id} of model {model_name} is not loaded")
    
    shard_info = loaded_shards[model_name][shard_id]
    shard_path = shard_info["path"]
    
    # Load the model for this shard if not already loaded
    shard_model_key = f"{model_name}_{shard_id}"
    if shard_model_key not in loaded_models:
        model = AutoModelForCausalLM.from_pretrained(shard_path)
        model = model.to(DEVICE)
        loaded_models[shard_model_key] = model
    
    model = loaded_models[shard_model_key]
    tokenizer = loaded_tokenizers[model_name]
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
    )
    
    # Decode the generated text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result

def load_model_internal(model_name):
    """Internal helper function to load a model."""
    try:
        # Check if model is already loaded
        if model_name in loaded_models:
            return {
                'status': 'success',
                'message': f'Model {model_name} is already loaded'
            }
        
        # Load tokenizer
        if model_name not in loaded_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            loaded_tokenizers[model_name] = tokenizer
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        
        # Move model to appropriate device
        model = model.to(DEVICE)
        
        loaded_models[model_name] = model
        
        return {
            'status': 'success',
            'message': f'Model {model_name} loaded successfully on {DEVICE}'
        }
    
    except Exception as e:
        logger.error(f'Failed to load model internally: {str(e)}', exc_info=True)
        return {
            'status': 'error',
            'message': f'Failed to load model: {str(e)}'
        }
