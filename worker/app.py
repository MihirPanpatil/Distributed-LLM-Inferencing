import os
import json
import psutil
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Global variables to track loaded models and shards
loaded_models = {}
loaded_tokenizers = {}
loaded_shards = {}

# Directory for model cache
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/app/model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check worker health and resource usage."""
    # Get system resource usage
    cpu_percent = psutil.cpu_percent() / 100.0
    memory = psutil.virtual_memory()
    memory_percent = memory.percent / 100.0
    
    # Check GPU usage if available
    gpu_percent = 0.0
    if torch.cuda.is_available():
        try:
            # This is a simplified approach - in production you'd use 
            # a proper GPU monitoring library like pynvml
            gpu_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() \
                if torch.cuda.max_memory_allocated() > 0 else 0.0
        except:
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
    
    return jsonify({
        'status': 'healthy',
        'resources': {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'gpu': gpu_percent,
            'gpu_available': torch.cuda.is_available()
        },
        'loaded_models': list(loaded_models.keys()),
        'loaded_tokenizers': list(loaded_tokenizers.keys()),
        'loaded_shards': shard_info
    })

@app.route('/load_model', methods=['POST'])
def load_model():
    """Endpoint to load a complete model."""
    data = request.json
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({
            'status': 'error',
            'message': 'Model name is required'
        }), 400
    
    try:
        # Check if model is already loaded
        if model_name in loaded_models:
            return jsonify({
                'status': 'success',
                'message': f'Model {model_name} is already loaded'
            })
        
        # Load tokenizer
        if model_name not in loaded_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            loaded_tokenizers[model_name] = tokenizer
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        loaded_models[model_name] = model
        
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} loaded successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to load model: {str(e)}'
        }), 500

@app.route('/load_shard', methods=['POST'])
def load_shard():
    """Endpoint to load a model shard."""
    data = request.json
    model_name = data.get('model_name')
    shard_id = data.get('shard_id')
    shard_path = data.get('shard_path')
    
    if not all([model_name, shard_id, shard_path]):
        return jsonify({
            'status': 'error',
            'message': 'Model name, shard ID, and shard path are required'
        }), 400
    
    try:
        # Initialize the model_name entry in loaded_shards if it doesn't exist
        if model_name not in loaded_shards:
            loaded_shards[model_name] = {}
        
        # Check if shard is already loaded
        if shard_id in loaded_shards[model_name]:
            return jsonify({
                'status': 'success',
                'message': f'Shard {shard_id} of model {model_name} is already loaded'
            })
        
        # Load tokenizer if not already loaded
        if model_name not in loaded_tokenizers:
            tokenizer_path = os.path.join(os.path.dirname(shard_path), 'tokenizer')
            if os.path.exists(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fall back to loading from Hugging Face
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
        
        return jsonify({
            'status': 'success',
            'message': f'Shard {shard_id} of model {model_name} loaded successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to load shard: {str(e)}'
        }), 500

@app.route('/unload_model', methods=['POST'])
def unload_model():
    """Endpoint to unload a model."""
    data = request.json
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({
            'status': 'error',
            'message': 'Model name is required'
        }), 400
    
    try:
        # Unload the model
        if model_name in loaded_models:
            del loaded_models[model_name]
        
        # Unload the tokenizer
        if model_name in loaded_tokenizers:
            del loaded_tokenizers[model_name]
        
        # Unload any shards
        if model_name in loaded_shards:
            del loaded_shards[model_name]
        
        # Force CUDA garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} unloaded successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to unload model: {str(e)}'
        }), 500

@app.route('/inference', methods=['POST'])
def run_inference():
    """Endpoint to run inference with a model."""
    data = request.json
    model_name = data.get('model_name')
    prompt = data.get('prompt')
    max_length = data.get('max_length', 100)
    shard_ids = data.get('shard_ids')
    
    if not all([model_name, prompt]):
        return jsonify({
            'status': 'error',
            'message': 'Model name and prompt are required'
        }), 400
    
    try:
        # Check if we're using shards
        if shard_ids:
            result = run_shard_inference(model_name, prompt, shard_ids, max_length)
        else:
            # Load the model if not already loaded
            if model_name not in loaded_models:
                # Try to load the model
                load_response = load_model()
                if load_response[1] == 500:  # Error status code
                    return load_response
            
            # Get the model and tokenizer
            model = loaded_models[model_name]
            tokenizer = loaded_tokenizers[model_name]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
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
        
        return jsonify({
            'status': 'success',
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Inference failed: {str(e)}'
        }), 500

def run_shard_inference(model_name, prompt, shard_ids, max_length):
    """Run inference using model shards."""
    # For simplicity, we'll just use the first shard for now
    # In a real implementation, you would coordinate between shards
    shard_id = shard_ids[0]
    
    if shard_id not in loaded_shards.get(model_name, {}):
        raise ValueError(f"Shard {shard_id} of model {model_name} is not loaded")
    
    shard_info = loaded_shards[model_name][shard_id]
    shard_path = shard_info["path"]
    
    # Load the model for this shard if not already loaded
    shard_model_key = f"{model_name}_{shard_id}"
    if shard_model_key not in loaded_models:
        model = AutoModelForCausalLM.from_pretrained(shard_path)
        if torch.cuda.is_available():
            model = model.to('cuda')
        loaded_models[shard_model_key] = model
    
    model = loaded_models[shard_model_key]
    tokenizer = loaded_tokenizers[model_name]
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
