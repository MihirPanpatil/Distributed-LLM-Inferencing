from main import Logger
import os
import sys
import time
import socket
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize logger
logger = Logger('utils')

def get_llm_model(model_name):
    """
    Download and load a language model from Huggingface.
    
    Args:
        model_name (str): Name of the model to download from Huggingface
    
    Returns:
        tuple: The loaded model and tokenizer
    """
    try:
        logger.info(f'Downloading model: {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f'Successfully loaded model: {model_name}')
        return model, tokenizer
    except Exception as e:
        logger.error(f'Failed to load model {model_name}')
        logger.exception(str(e))
        return None, None

def shard_model(model, num_shards=2):
    """
    Split a model into multiple shards for distributed processing.
    
    Args:
        model: The model to shard
        num_shards (int): Number of shards to create
    
    Returns:
        list: List of model shards
    """
    # Placeholder for actual sharding logic
    # In a real implementation, this would use libraries like DeepSpeed or Accelerate
    logger.info(f'Sharding model into {num_shards} parts')
    try:
        # Simulating sharding for now
        shards = [f'Shard {i+1} of {model}' for i in range(num_shards)]
        logger.info('Model sharding completed successfully')
        return shards
    except Exception as e:
        logger.error('Failed to shard model')
        logger.exception(str(e))
        return []

def run_worker(master_host, port=5555):
    """
    Run a worker node that connects to the master and processes assigned tasks.
    
    Args:
        master_host (str): Hostname or IP of the master node
        port (int): Port to connect to on the master node
    """
    logger.info(f'Starting worker node, connecting to master at {master_host}:{port}')
    
    try:
        # Set up socket connection to master
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((master_host, port))
        logger.info('Connected to master node')
        
        # Main worker loop
        while True:
            # Receive data from master
            data = client_socket.recv(4096)
            if not data:
                logger.warning('Connection to master lost')
                break
            
            # Process the received data
            logger.info(f'Received task from master: {data.decode()}')
            
            # Simulate processing
            time.sleep(2)
            
            # Send response back to master
            response = f'Processed: {data.decode()}'
            client_socket.send(response.encode())
            logger.info('Task completed and response sent to master')
            
    except socket.error as e:
        logger.error(f'Socket error: {str(e)}')
    except Exception as e:
        logger.error('Worker encountered an error')
        logger.exception(str(e))
    finally:
        logger.info('Worker shutting down')
        try:
            client_socket.close()
        except:
            pass

def distribute_model_to_workers(model, worker_ips):
    """
    Distribute model shards to worker nodes.
    
    Args:
        model: The model to distribute
        worker_ips (list): List of worker IP addresses
    
    Returns:
        bool: True if distribution succeeded, False otherwise
    """
    if not worker_ips:
        logger.warning('No worker nodes available for model distribution')
        return False
    
    try:
        # Shard the model based on number of workers
        shards = shard_model(model, len(worker_ips))
        logger.info(f'Created {len(shards)} shards for {len(worker_ips)} workers')
        
        # Distribute shards to workers (placeholder for actual distribution logic)
        for i, worker_ip in enumerate(worker_ips):
            # In a real implementation, this would send the actual model shard
            # to the worker using some protocol (e.g., gRPC, SSH, etc.)
            logger.info(f'Sending shard {i+1} to worker at {worker_ip}')
            # Simulating network latency
            time.sleep(1)
            
        logger.info('Model distribution completed successfully')
        return True
    except Exception as e:
        logger.error('Failed to distribute model to workers')
        logger.exception(str(e))
        return False

def run_inference(model, tokenizer, input_text):
    """
    Run inference on the given model and input text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        input_text (str): The input text for inference
    
    Returns:
        str: Generated text response
    """
    try:
        logger.info('Running inference on input text')
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors='pt')
        
        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7
        )
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info('Inference completed successfully')
        return response
    except Exception as e:
        logger.error('Inference failed')
        logger.exception(str(e))
        return 'Error: Failed to generate response'
