from main import Logger
import os
import sys
import time
import socket
import threading
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Tuple, Optional, Union, Any

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

def shard_model(model, num_shards=2, parallelism_type='both'):
    """
    Split a model into multiple shards for distributed processing using
    either pipeline parallelism, tensor parallelism, or both.
    
    Args:
        model: The model to shard
        num_shards (int): Number of shards to create
        parallelism_type (str): Type of parallelism ('pipeline', 'tensor', or 'both')
    
    Returns:
        list: List of model shards
    """
    logger.info(f'Sharding model into {num_shards} parts using {parallelism_type} parallelism')
    
    try:
        if parallelism_type == 'pipeline':
            return pipeline_parallel_shard(model, num_shards)
        elif parallelism_type == 'tensor':
            return tensor_parallel_shard(model, num_shards)
        else:  # 'both' or any other value
            # Combine both approaches - first split using pipeline parallelism then tensor
            pipeline_shards = pipeline_parallel_shard(model, num_shards // 2 or 1)
            final_shards = []
            
            for pipe_shard in pipeline_shards:
                tensor_shards = tensor_parallel_shard(pipe_shard, 2)
                final_shards.extend(tensor_shards)
            
            logger.info(f'Created {len(final_shards)} combined shards using both parallelism types')
            return final_shards
    except Exception as e:
        logger.error('Failed to shard model')
        logger.exception(str(e))
        return []

def pipeline_parallel_shard(model, num_stages):
    """
    Implement pipeline parallelism by dividing the model into sequential stages.
    
    Args:
        model: The model to divide into pipeline stages
        num_stages (int): Number of pipeline stages
    
    Returns:
        list: List of model components for each pipeline stage
    """
    try:
        logger.info(f'Creating {num_stages} pipeline stages')
        
        # In a real implementation, we would use deepspeed.pipe or similar
        # Here, we're creating a simplified representation of pipeline stages
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # For models like GPT-2 with transformer blocks
            layers = model.transformer.h
            num_layers = len(layers)
            
            # Determine layers per stage
            layers_per_stage = num_layers // num_stages
            if layers_per_stage == 0:
                layers_per_stage = 1
                logger.warning(f'Too many stages for model with {num_layers} layers. Using {num_layers} stages instead.')
                num_stages = num_layers
            
            # Create pipeline stages
            stages = []
            for i in range(num_stages):
                start_idx = i * layers_per_stage
                end_idx = min((i + 1) * layers_per_stage, num_layers)
                
                # In a real implementation, we would create actual model segments
                # For now, we're using dictionary representations
                stage = {
                    'type': 'pipeline_stage',
                    'stage_id': i,
                    'layer_range': (start_idx, end_idx),
                    'input_layer': i == 0,
                    'output_layer': i == num_stages - 1,
                    'requires_input_from': i - 1 if i > 0 else None,
                    'sends_output_to': i + 1 if i < num_stages - 1 else None
                }
                stages.append(stage)
                
            logger.info(f'Successfully created {len(stages)} pipeline stages')
            return stages
        else:
            # Fallback for models without transformer.h structure
            logger.warning('Model structure not supported for pipeline parallelism, using generic splits')
            stages = []
            for i in range(num_stages):
                stage = {
                    'type': 'pipeline_stage',
                    'stage_id': i,
                    'generic_split': True,
                    'split_index': i,
                    'total_splits': num_stages,
                    'input_layer': i == 0,
                    'output_layer': i == num_stages - 1,
                    'requires_input_from': i - 1 if i > 0 else None,
                    'sends_output_to': i + 1 if i < num_stages - 1 else None
                }
                stages.append(stage)
            return stages
    except Exception as e:
        logger.error('Failed to create pipeline stages')
        logger.exception(str(e))
        return []

def tensor_parallel_shard(model, num_shards):
    """
    Implement tensor parallelism by splitting matrix operations across devices.
    
    Args:
        model: The model or pipeline stage to split
        num_shards (int): Number of tensor shards to create
    
    Returns:
        list: List of model components with tensor parallelism
    """
    try:
        logger.info(f'Creating {num_shards} tensor-parallel shards')
        
        # In a real implementation, we would use deepspeed.zero or PyTorch's DistributedDataParallel
        # Here, we're creating a simplified representation
        shards = []
        
        for i in range(num_shards):
            # Represent tensor parallelism information
            shard = {
                'type': 'tensor_shard',
                'shard_id': i,
                'total_shards': num_shards,
                'handles_dims': f'{i} to {i+1}/{num_shards}',
                'original_model': model if isinstance(model, dict) else f'tensor_shard_of_{model}'
            }
            
            # In tensor parallelism, each shard handles a portion of each layer's weights
            shard['weight_partition'] = {
                'start_dim': i * (100 // num_shards),
                'end_dim': (i + 1) * (100 // num_shards)
            }
            
            shards.append(shard)
        
        logger.info(f'Successfully created {len(shards)} tensor-parallel shards')
        return shards
    except Exception as e:
        logger.error('Failed to create tensor-parallel shards')
        logger.exception(str(e))
        return []

def create_pipeline_execution_plan(worker_ips, parallelism_type='both'):
    """
    Create an execution plan for coordinating inference across worker nodes.
    
    Args:
        worker_ips (list): List of worker IP addresses
        parallelism_type (str): Type of parallelism to use
    
    Returns:
        dict: Execution plan with tasks assigned to workers
    """
    try:
        num_workers = len(worker_ips)
        logger.info(f'Creating execution plan for {num_workers} workers using {parallelism_type} parallelism')
        
        execution_plan = {
            'parallelism_type': parallelism_type,
            'num_workers': num_workers,
            'worker_assignments': {},
            'communication_pattern': {}
        }
        
        if parallelism_type == 'pipeline':
            # In pipeline parallelism, each worker processes a stage sequentially
            for i, worker_ip in enumerate(worker_ips):
                execution_plan['worker_assignments'][worker_ip] = {
                    'task_type': 'pipeline_stage',
                    'stage_id': i,
                    'receives_from': worker_ips[i-1] if i > 0 else None,
                    'sends_to': worker_ips[i+1] if i < num_workers - 1 else None
                }
                
            # Define the communication pattern (who sends to whom)
            for i in range(num_workers - 1):
                execution_plan['communication_pattern'][worker_ips[i]] = [worker_ips[i+1]]
                
        elif parallelism_type == 'tensor':
            # In tensor parallelism, workers process the same stage but different parts
            for i, worker_ip in enumerate(worker_ips):
                execution_plan['worker_assignments'][worker_ip] = {
                    'task_type': 'tensor_partition',
                    'partition_id': i,
                    'total_partitions': num_workers,
                    'all_workers': worker_ips  # All workers need to communicate
                }
                
            # All workers communicate with all others for tensor aggregation
            for worker_ip in worker_ips:
                execution_plan['communication_pattern'][worker_ip] = [
                    w for w in worker_ips if w != worker_ip
                ]
                
        else:  # 'both' or any other value
            # Combine pipeline and tensor parallelism
            pipeline_stages = max(2, num_workers // 2)
            tensor_partitions_per_stage = max(2, num_workers // pipeline_stages)
            
            worker_idx = 0
            for stage in range(pipeline_stages):
                stage_workers = []
                
                for partition in range(min(tensor_partitions_per_stage, num_workers - worker_idx)):
                    if worker_idx < num_workers:
                        worker_ip = worker_ips[worker_idx]
                        execution_plan['worker_assignments'][worker_ip] = {
                            'task_type': 'combined',
                            'pipeline_stage': stage,
                            'tensor_partition': partition,
                            'total_partitions_in_stage': tensor_partitions_per_stage
                        }
                        stage_workers.append(worker_ip)
                        worker_idx += 1
                
                # Define communication pattern for this stage
                if stage > 0:
                    previous_stage_workers = [
                        w for w, data in execution_plan['worker_assignments'].items()
                        if data.get('pipeline_stage') == stage - 1
                    ]
                    
                    # Workers in this stage get input from all workers in previous stage
                    for worker in stage_workers:
                        execution_plan['communication_pattern'].setdefault(worker, []).extend(previous_stage_workers)
                    
                    # Workers in previous stage send output to all workers in this stage
                    for worker in previous_stage_workers:
                        execution_plan['communication_pattern'].setdefault(worker, []).extend(stage_workers)
                
                # Workers within the same stage communicate for tensor aggregation
                for worker in stage_workers:
                    other_stage_workers = [w for w in stage_workers if w != worker]
                    execution_plan['communication_pattern'].setdefault(worker, []).extend(other_stage_workers)
        
        logger.info('Successfully created execution plan')
        return execution_plan
    except Exception as e:
        logger.error('Failed to create execution plan')
        logger.exception(str(e))
        return {'error': str(e)}

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
        
        # Worker state
        worker_state = {
            'model_shard': None,
            'tokenizer': None,
            'task_type': None,
            'stage_id': None,
            'partition_id': None
        }
        
        # Main worker loop
        while True:
            # Receive data from master
            data = client_socket.recv(8192)  # Increased buffer size for larger messages
            if not data:
                logger.warning('Connection to master lost')
                break
            
            try:
                # Decode and parse the message
                message = json.loads(data.decode())
                command = message.get('command')
                
                logger.info(f'Received command from master: {command}')
                
                if command == 'init_shard':
                    # Initialize a model shard
                    shard_config = message.get('shard_config', {})
                    parallelism_type = shard_config.get('type')
                    
                    logger.info(f'Initializing {parallelism_type} shard')
                    
                    # In a real implementation, we would load the actual model shard
                    # For now, we're just storing the configuration
                    worker_state['model_shard'] = shard_config
                    worker_state['task_type'] = parallelism_type
                    
                    if parallelism_type == 'pipeline_stage':
                        worker_state['stage_id'] = shard_config.get('stage_id')
                    elif parallelism_type == 'tensor_shard':
                        worker_state['partition_id'] = shard_config.get('shard_id')
                    
                    response = {
                        'status': 'success',
                        'message': f'Initialized {parallelism_type} shard'
                    }
                
                elif command == 'process_input':
                    # Process input for inference
                    input_data = message.get('input_data')
                    execution_info = message.get('execution_info', {})
                    
                    logger.info(f'Processing input for {worker_state["task_type"]}')
                    
                    # In a real implementation, we would do actual processing
                    # based on the shard type and configuration
                    if worker_state['task_type'] == 'pipeline_stage':
                        # Simulate pipeline stage processing
                        time.sleep(1)  # Simulate computation time
                        output = f'Stage {worker_state["stage_id"]} processed: {input_data}'
                        
                        # If this is the final stage, return the complete result
                        # Otherwise, the result would be passed to the next stage
                        is_final = worker_state['model_shard'].get('output_layer', False)
                        
                        response = {
                            'status': 'success',
                            'output': output,
                            'is_final': is_final,
                            'next_stage': worker_state['model_shard'].get('sends_output_to')
                        }
                    
                    elif worker_state['task_type'] == 'tensor_shard':
                        # Simulate tensor parallelism processing
                        time.sleep(0.5)  # Simulate computation time
                        partition_result = f'Partition {worker_state["partition_id"]} result for {input_data}'
                        
                        response = {
                            'status': 'success',
                            'partition_result': partition_result,
                            'partition_id': worker_state['partition_id'],
                            'requires_aggregation': True
                        }
                    
                    else:
                        response = {
                            'status': 'error',
                            'message': f'Unknown task type: {worker_state["task_type"]}'
                        }
                
                elif command == 'shutdown':
                    logger.info('Received shutdown command')
                    response = {
                        'status': 'success',
                        'message': 'Worker shutting down'
                    }
                    client_socket.send(json.dumps(response).encode())
                    break
                
                else:
                    logger.warning(f'Unknown command: {command}')
                    response = {
                        'status': 'error',
                        'message': f'Unknown command: {command}'
                    }
                
                # Send response back to master
                client_socket.send(json.dumps(response).encode())
                logger.info(f'Sent response for command: {command}')
                
            except json.JSONDecodeError:
                logger.error('Failed to decode JSON message')
                response = {
                    'status': 'error',
                    'message': 'Invalid JSON format'
                }
                client_socket.send(json.dumps(response).encode())
            
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

def distribute_model_to_workers(model, worker_ips, parallelism_type='both'):
    """
    Distribute model shards to worker nodes based on the parallelism type.
    
    Args:
        model: The model to distribute
        worker_ips (list): List of worker IP addresses
        parallelism_type (str): Type of parallelism to use
    
    Returns:
        bool: True if distribution succeeded, False otherwise
    """
    if not worker_ips:
        logger.warning('No worker nodes available for model distribution')
        return False
    
    try:
        # Create shards based on parallelism type and number of workers
        shards = shard_model(model, len(worker_ips), parallelism_type)
        logger.info(f'Created {len(shards)} shards for {len(worker_ips)} workers using {parallelism_type} parallelism')
        
        if len(shards) != len(worker_ips):
            logger.warning(f'Number of shards ({len(shards)}) does not match number of workers ({len(worker_ips)})')
            # In a real implementation, we might adjust the sharding or mapping
        
        # Create an execution plan
        execution_plan = create_pipeline_execution_plan(worker_ips, parallelism_type)
        logger.info('Created execution plan for distributed inference')
        
        # Distribute shards to workers (placeholder for actual distribution logic)
        for i, worker_ip in enumerate(worker_ips):
            if i < len(shards):
                # In a real implementation, this would send the actual model shard
                # to the worker using some protocol (e.g., gRPC, SSH, etc.)
                logger.info(f'Sending shard {i+1} to worker at {worker_ip}')
                
                # Get worker assignment from execution plan
                worker_assignment = execution_plan['worker_assignments'].get(worker_ip, {})
                
                # Prepare initialization message
                init_message = {
                    'command': 'init_shard',
                    'shard_config': shards[i],
                    'execution_info': worker_assignment
                }
                
                # Simulate sending message to worker
                # In a real implementation, we would actually send this message
                logger.info(f'Worker {worker_ip} assigned: {worker_assignment}')
                # Simulating network latency
                time.sleep(0.5)
            else:
                logger.warning(f'No shard available for worker at {worker_ip}')
        
        logger.info('Model distribution completed successfully')
        return True
    except Exception as e:
        logger.error('Failed to distribute model to workers')
        logger.exception(str(e))
        return False

def distributed_inference(model, tokenizer, input_text, worker_connections, execution_plan):
    """
    Run distributed inference across multiple worker nodes based on the execution plan.
    
    Args:
        model: The language model (master copy)
        tokenizer: The tokenizer for the model
        input_text (str): The input text for inference
        worker_connections (dict): Dictionary of worker connections (socket objects)
        execution_plan (dict): Execution plan detailing how to distribute the work
    
    Returns:
        str: Generated text response
    """
    try:
        logger.info('Starting distributed inference')
        parallelism_type = execution_plan.get('parallelism_type', 'both')
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors='pt')
        logger.info('Input tokenized successfully')
        
        if parallelism_type == 'pipeline':
            # For pipeline parallelism, send the input to the first stage
            first_stage_workers = [
                worker for worker, config in execution_plan['worker_assignments'].items()
                if config.get('stage_id') == 0 or config.get('pipeline_stage') == 0
            ]
            
            if not first_stage_workers:
                logger.error('No workers found for the first pipeline stage')
                return 'Error: Pipeline configuration error'
            
            # Send to the first worker in the pipeline
            first_worker = first_stage_workers[0]
            first_worker_socket = worker_connections.get(first_worker)
            
            if not first_worker_socket:
                logger.error(f'No connection found for worker {first_worker}')
                return 'Error: Worker connection not found'
            
            # Start the pipeline
            message = {
                'command': 'process_input',
                'input_data': input_text,
                'execution_info': {
                    'parallelism_type': 'pipeline'
                }
            }
            
            first_worker_socket.send(json.dumps(message).encode())
            logger.info(f'Sent input to first pipeline stage worker: {first_worker}')
            
            # Wait for the result from the final stage
            # In a real implementation, we would need to handle pipeline coordination
            # Here we're simplifying and assuming the result comes back to the master
            
            # Simulating result collection from the final stage
            time.sleep(2)
            result = f'Pipeline processed result for: {input_text}'
        
        elif parallelism_type == 'tensor':
            # For tensor parallelism, distribute the same input to all workers
            for worker_ip, worker_socket in worker_connections.items():
                message = {
                    'command': 'process_input',
                    'input_data': input_text,
                    'execution_info': {
                        'parallelism_type': 'tensor'
                    }
                }
                
                worker_socket.send(json.dumps(message).encode())
                logger.info(f'Sent input to tensor parallel worker: {worker_ip}')
            
            # In a real implementation, we would gather and aggregate results from all workers
            # Here we're simulating result aggregation
            time.sleep(1)
            result = f'Tensor-parallel aggregated result for: {input_text}'
        
        else:  # 'both' or any other value
            # Combine pipeline and tensor parallelism
            # Similar to pipeline parallelism, but with tensor parallelism at each stage
            
            # Find workers for the first stage
            first_stage_workers = [
                worker for worker, config in execution_plan['worker_assignments'].items()
                if config.get('pipeline_stage') == 0
            ]
            
            if not first_stage_workers:
                logger.error('No workers found for the first combined stage')
                return 'Error: Combined parallelism configuration error'
            
            # Send to all workers in the first stage
            for worker_ip in first_stage_workers:
                worker_socket = worker_connections.get(worker_ip)
                
                if worker_socket:
                    message = {
                        'command': 'process_input',
                        'input_data': input_text,
                        'execution_info': {
                            'parallelism_type': 'combined'
                        }
                    }
                    
                    worker_socket.send(json.dumps(message).encode())
                    logger.info(f'Sent input to combined parallelism worker: {worker_ip}')
            
            # Simulating complex coordination between stages
            time.sleep(3)
            result = f'Combined pipeline and tensor parallel result for: {input_text}'
        
        logger.info('Distributed inference completed successfully')
        return result
    except Exception as e:
        logger.error('Distributed inference failed')
        logger.exception(str(e))
        return f'Error: {str(e)}'

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
