from main import Logger
import os
import sys
import time
import socket
import threading
import json
import torch
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

def shard_model(model, num_shards=2, parallelism_type='both'):
    """
    Split a model into multiple shards for distributed processing using
    pipeline parallelism, tensor parallelism, or both.
    
    Args:
        model: The model to shard
        num_shards (int): Number of shards to create
        parallelism_type (str): Type of parallelism to use ('pipeline', 'tensor', or 'both')
    
    Returns:
        list: List of model shards with parallelism information
    """
    logger.info(f'Sharding model into {num_shards} parts using {parallelism_type} parallelism')
    
    try:
        if parallelism_type == 'pipeline':
            return pipeline_parallel_shard(model, num_shards)
        elif parallelism_type == 'tensor':
            return tensor_parallel_shard(model, num_shards)
        else:  # 'both' or default
            # For combined parallelism, determine the division between pipeline and tensor
            pipeline_stages = max(2, num_shards // 2)
            tensor_partitions_per_stage = max(2, num_shards // pipeline_stages)
            
            logger.info(f'Creating {pipeline_stages} pipeline stages with {tensor_partitions_per_stage} tensor partitions each')
            
            # First, create pipeline stages
            pipeline_shards = pipeline_parallel_shard(model, pipeline_stages)
            
            # Then, for each pipeline stage, apply tensor parallelism
            all_shards = []
            for i, pipe_shard in enumerate(pipeline_shards):
                tensor_shards = tensor_parallel_shard(pipe_shard, tensor_partitions_per_stage)
                
                # Add pipeline stage information to each tensor shard
                for j, tensor_shard in enumerate(tensor_shards):
                    tensor_shard['pipeline_stage'] = i
                    tensor_shard['pipeline_total_stages'] = pipeline_stages
                    tensor_shard['combined_parallelism'] = True
                    tensor_shard['combined_id'] = f'p{i}t{j}'
                
                all_shards.extend(tensor_shards)
            
            logger.info(f'Created {len(all_shards)} combined shards')
            return all_shards
    except Exception as e:
        logger.error('Failed to shard model')
        logger.exception(str(e))
        return []

def pipeline_parallel_shard(model, num_stages):
    """
    Implement pipeline parallelism by dividing the model into sequential stages.
    Each stage processes a different part of the model in sequence.
    
    Args:
        model: The model to divide into sequential stages
        num_stages (int): Number of pipeline stages to create
    
    Returns:
        list: List of model shards for pipeline parallelism
    """
    try:
        logger.info(f'Creating {num_stages} pipeline stages')
        
        # For models with transformer architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            num_layers = len(layers)
            
            # Ensure we don't create more stages than layers
            if num_stages > num_layers:
                logger.warning(f'Reducing pipeline stages from {num_stages} to {num_layers} (number of layers)')
                num_stages = num_layers
            
            # Distribute layers across stages
            layers_per_stage = num_layers // num_stages
            extra_layers = num_layers % num_stages
            
            stages = []
            start_idx = 0
            
            for stage_id in range(num_stages):
                # Calculate how many layers this stage gets
                stage_layers = layers_per_stage + (1 if stage_id < extra_layers else 0)
                end_idx = start_idx + stage_layers
                
                # Create stage configuration
                stage = {
                    'type': 'pipeline_stage',
                    'stage_id': stage_id,
                    'total_stages': num_stages,
                    'layer_range': (start_idx, end_idx),
                    'num_layers': stage_layers,
                    'is_first': stage_id == 0,
                    'is_last': stage_id == num_stages - 1,
                    'requires_input_from': stage_id - 1 if stage_id > 0 else None,
                    'sends_output_to': stage_id + 1 if stage_id < num_stages - 1 else None
                }
                
                stages.append(stage)
                start_idx = end_idx
            
            logger.info(f'Created {len(stages)} pipeline stages with layer distribution: {[s["num_layers"] for s in stages]}')
            return stages
            
        # For models with a different structure
        else:
            logger.warning('Model structure not recognized for optimal pipeline splitting. Using equal division.')
            
            # Create generic pipeline stages
            stages = []
            for i in range(num_stages):
                stage = {
                    'type': 'pipeline_stage',
                    'stage_id': i,
                    'total_stages': num_stages,
                    'generic_division': True,
                    'division_fraction': (i / num_stages, (i + 1) / num_stages),
                    'is_first': i == 0,
                    'is_last': i == num_stages - 1,
                    'requires_input_from': i - 1 if i > 0 else None,
                    'sends_output_to': i + 1 if i < num_stages - 1 else None
                }
                stages.append(stage)
            
            logger.info(f'Created {len(stages)} generic pipeline stages')
            return stages
    except Exception as e:
        logger.error('Failed to create pipeline stages')
        logger.exception(str(e))
        return []

def tensor_parallel_shard(model, num_partitions):
    """
    Implement tensor parallelism by splitting matrix operations across devices.
    Each partition handles a subset of the weights for parallel computation.
    
    Args:
        model: The model or pipeline stage to split for tensor parallelism
        num_partitions (int): Number of tensor partitions to create
    
    Returns:
        list: List of model partitions for tensor parallelism
    """
    try:
        logger.info(f'Creating {num_partitions} tensor-parallel partitions')
        
        # Check if we're dealing with a pipeline stage or full model
        is_pipeline_stage = isinstance(model, dict) and model.get('type') == 'pipeline_stage'
        
        partitions = []
        for i in range(num_partitions):
            # Create tensor partition configuration
            partition = {
                'type': 'tensor_partition',
                'partition_id': i,
                'total_partitions': num_partitions,
                'dimension_range': (i / num_partitions, (i + 1) / num_partitions),
                'original_source': model
            }
            
            # If the input is a pipeline stage, preserve its information
            if is_pipeline_stage:
                partition['pipeline_stage_id'] = model['stage_id']
                partition['pipeline_total_stages'] = model['total_stages']
                partition['layer_range'] = model['layer_range']
                partition['is_first_stage'] = model['is_first']
                partition['is_last_stage'] = model['is_last']
                partition['requires_input_from'] = model['requires_input_from']
                partition['sends_output_to'] = model['sends_output_to']
            
            # For tensor parallelism, we need to define which part of each weight matrix this partition handles
            partition['weight_ranges'] = {
                'attention': {
                    'q_proj': (i / num_partitions, (i + 1) / num_partitions),
                    'k_proj': (i / num_partitions, (i + 1) / num_partitions),
                    'v_proj': (i / num_partitions, (i + 1) / num_partitions),
                    'out_proj': (i / num_partitions, (i + 1) / num_partitions)
                },
                'mlp': {
                    'fc_in': (i / num_partitions, (i + 1) / num_partitions),
                    'fc_out': (i / num_partitions, (i + 1) / num_partitions)
                }
            }
            
            partitions.append(partition)
        
        logger.info(f'Created {len(partitions)} tensor-parallel partitions')
        return partitions
    except Exception as e:
        logger.error('Failed to create tensor-parallel partitions')
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
        
        # Worker state
        worker_state = {
            'model_shard': None,
            'tokenizer': None,
            'parallelism_type': None,
            'pipeline_stage': None,
            'tensor_partition': None,
            'is_combined': False,
            'processing_queue': [],
            'peers': []
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
                    # Initialize the worker with a model shard
                    shard_config = message.get('shard_config', {})
                    shard_type = shard_config.get('type')
                    
                    logger.info(f'Initializing worker with {shard_type} shard')
                    
                    worker_state['model_shard'] = shard_config
                    worker_state['parallelism_type'] = shard_type
                    
                    # Extract relevant configuration based on shard type
                    if shard_type == 'pipeline_stage':
                        worker_state['pipeline_stage'] = shard_config.get('stage_id')
                        worker_state['is_first_stage'] = shard_config.get('is_first', False)
                        worker_state['is_last_stage'] = shard_config.get('is_last', False)
                    
                    elif shard_type == 'tensor_partition':
                        worker_state['tensor_partition'] = shard_config.get('partition_id')
                        worker_state['total_partitions'] = shard_config.get('total_partitions')
                    
                    else:  # Combined
                        worker_state['is_combined'] = True
                        worker_state['pipeline_stage'] = shard_config.get('pipeline_stage')
                        worker_state['tensor_partition'] = shard_config.get('partition_id')
                        worker_state['combined_id'] = shard_config.get('combined_id')
                    
                    # Store peer information if available
                    if 'peers' in message:
                        worker_state['peers'] = message['peers']
                    
                    response = {
                        'status': 'success',
                        'message': f'Initialized with {shard_type} shard'
                    }
                
                elif command == 'process_input':
                    # Process input for the assigned shard
                    input_data = message.get('input_data')
                    execution_info = message.get('execution_info', {})
                    
                    logger.info(f'Processing input with {worker_state["parallelism_type"]} parallelism')
                    
                    if worker_state['parallelism_type'] == 'pipeline_stage':
                        # For pipeline parallelism, process the assigned stage
                        stage_id = worker_state['pipeline_stage']
                        is_first = worker_state['is_first_stage']
                        is_last = worker_state['is_last_stage']
                        
                        # Simulate processing time for this pipeline stage
                        time.sleep(1)
                        processed_output = f'Pipeline stage {stage_id} processed: {input_data}'
                        
                        response = {
                            'status': 'success',
                            'stage_id': stage_id,
                            'output': processed_output,
                            'is_final': is_last
                        }
                        
                        # If not the last stage, this output needs to go to the next stage
                        if not is_last:
                            response['next_stage'] = stage_id + 1
                    
                    elif worker_state['parallelism_type'] == 'tensor_partition':
                        # For tensor parallelism, process the assigned partition
                        partition_id = worker_state['tensor_partition']
                        total_parts = worker_state['total_partitions']
                        
                        # Simulate processing time for tensor partition
                        time.sleep(0.5)
                        partial_output = f'Tensor partition {partition_id}/{total_parts} result for: {input_data}'
                        
                        response = {
                            'status': 'success',
                            'partition_id': partition_id,
                            'partial_output': partial_output,
                            'requires_aggregation': True
                        }
                    
                    else:  # Combined parallelism
                        # For combined parallelism, handle both pipeline and tensor aspects
                        pipeline_stage = worker_state['pipeline_stage']
                        tensor_part = worker_state['tensor_partition']
                        combined_id = worker_state['combined_id']
                        
                        # Simulate combined processing
                        time.sleep(1.5)
                        combined_output = f'Combined {combined_id} (p{pipeline_stage}-t{tensor_part}) processed: {input_data}'
                        
                        response = {
                            'status': 'success',
                            'combined_id': combined_id,
                            'pipeline_stage': pipeline_stage,
                            'tensor_partition': tensor_part,
                            'output': combined_output,
                            'requires_pipeline_forward': pipeline_stage < worker_state.get('pipeline_total_stages', 1) - 1,
                            'requires_tensor_aggregation': True
                        }
                
                elif command == 'aggregate_tensor_results':
                    # Aggregate partial results from tensor parallelism
                    partial_results = message.get('partial_results', [])
                    
                    logger.info(f'Aggregating {len(partial_results)} tensor partial results')
                    
                    # Simulate aggregation time
                    time.sleep(0.5)
                    aggregated_result = f'Aggregated result from {len(partial_results)} tensor partitions'
                    
                    response = {
                        'status': 'success',
                        'aggregated_result': aggregated_result
                    }
                
                elif command == 'shutdown':
                    # Handle shutdown command
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
        logger.info(f'Created {len(shards)} shards for {len(worker_ips)} workers')
        
        if len(shards) != len(worker_ips):
            logger.warning(f'Number of shards ({len(shards)}) does not match number of workers ({len(worker_ips)})')
            # Adjust by taking the first N shards where N is the number of workers
            if len(shards) > len(worker_ips):
                shards = shards[:len(worker_ips)]
                logger.info(f'Using only the first {len(worker_ips)} shards')
            # If fewer shards than workers, some workers will remain unused
            elif len(shards) < len(worker_ips):
                logger.warning(f'Only {len(shards)} workers will be used out of {len(worker_ips)}')
                worker_ips = worker_ips[:len(shards)]
        
        # Create a map of peers for each worker based on the parallelism type
        peer_map = {}
        
        if parallelism_type == 'pipeline':
            # For pipeline parallelism, each worker talks to previous and next
            for i, worker_ip in enumerate(worker_ips):
                peers = []
                if i > 0:  # Has previous stage
                    peers.append({'ip': worker_ips[i-1], 'role': 'previous_stage'})
                if i < len(worker_ips) - 1:  # Has next stage
                    peers.append({'ip': worker_ips[i+1], 'role': 'next_stage'})
                peer_map[worker_ip] = peers
        
        elif parallelism_type == 'tensor':
            # For tensor parallelism, all workers need to communicate with each other
            for worker_ip in worker_ips:
                peers = [{'ip': ip, 'role': 'tensor_peer'} for ip in worker_ips if ip != worker_ip]
                peer_map[worker_ip] = peers
        
        else:  # 'both'
            # For combined parallelism, organize by pipeline stages and tensor partitions
            # This is a simplified version - a real implementation would be more complex
            worker_assignments = {}
            num_workers = len(worker_ips)
            pipeline_stages = max(2, num_workers // 2)
            tensor_parts_per_stage = max(2, num_workers // pipeline_stages)
            
            # Assign workers to pipeline stages and tensor partitions
            worker_idx = 0
            for stage in range(pipeline_stages):
                stage_workers = []
                for part in range(min(tensor_parts_per_stage, num_workers - worker_idx)):
                    if worker_idx < num_workers:
                        worker_assignments[worker_ips[worker_idx]] = {
                            'pipeline_stage': stage,
                            'tensor_partition': part
                        }
                        stage_workers.append(worker_ips[worker_idx])
                        worker_idx += 1
                
                # For each worker in this stage, set up peers
                for worker_ip in stage_workers:
                    peers = []
                    
                    # Add tensor peers (other workers in same stage)
                    tensor_peers = [ip for ip in stage_workers if ip != worker_ip]
                    for peer_ip in tensor_peers:
                        peers.append({'ip': peer_ip, 'role': 'tensor_peer'})
                    
                    # Add pipeline peers (workers in adjacent stages)
                    if stage > 0:  # Has previous stage
                        prev_stage_workers = [
                            ip for ip, data in worker_assignments.items()
                            if data.get('pipeline_stage') == stage - 1
                        ]
                        for peer_ip in prev_stage_workers:
                            peers.append({'ip': peer_ip, 'role': 'previous_stage'})
                    
                    if stage < pipeline_stages - 1:  # Has next stage
                        # Next stage workers might not exist yet, so calculate them
                        next_stage_start = worker_idx
                        next_stage_end = min(next_stage_start + tensor_parts_per_stage, num_workers)
                        next_stage_workers = worker_ips[next_stage_start:next_stage_end]
                        for peer_ip in next_stage_workers:
                            peers.append({'ip': peer_ip, 'role': 'next_stage'})
                    
                    peer_map[worker_ip] = peers
        
        # Distribute shards to workers
        for i, (worker_ip, shard) in enumerate(zip(worker_ips, shards)):
            logger.info(f'Sending shard {i+1} to worker at {worker_ip}')
            
            # In a real implementation, this would send the actual shard to the worker
            # Here we're just simulating the distribution
            
            # Simulating network latency
            time.sleep(0.5)
        
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

def distributed_inference(model, tokenizer, input_text, worker_connections, execution_plan):
    """
    Run distributed inference across multiple worker nodes based on the execution plan.
    
    Args:
        model: The language model (master copy)
        tokenizer: The tokenizer for the model
        input_text (str): The input text for inference
        worker_connections (dict): Dictionary of worker connections (socket objects)
        execution_plan (dict): Execution plan for distributed inference
    
    Returns:
        str: Generated text response
    """
    try:
        logger.info('Starting distributed inference')
        parallelism_type = execution_plan.get('parallelism_type', 'both')
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs.input_ids.tolist()
        logger.info(f'Input tokenized with {len(input_ids[0])} tokens')
        
        # Process based on parallelism type
        if parallelism_type == 'pipeline':
            # For pipeline parallelism, find the first stage workers
            first_stage_workers = [
                worker_ip for worker_ip, assignment in execution_plan['worker_assignments'].items()
                if assignment.get('stage_id') == 0 or assignment.get('pipeline_stage') == 0
            ]
            
            if not first_stage_workers:
                logger.error('No workers found for the first pipeline stage')
                return 'Error: Pipeline configuration error'
            
            # Send input to the first stage worker
            first_worker = first_stage_workers[0]
            worker_socket = worker_connections.get(first_worker)
            
            if not worker_socket:
                logger.error(f'No connection found for worker {first_worker}')
                return 'Error: Worker connection lost'
            
            # Send inference request to the first stage
            request = {
                'command': 'process_input',
                'input_data': {
                    'text': input_text,
                    'tokens': input_ids
                },
                'execution_info': {
                    'parallelism_type': 'pipeline',
                    'is_first_stage': True
                }
            }
            
            worker_socket.send(json.dumps(request).encode())
            logger.info(f'Sent input to first pipeline stage worker: {first_worker}')
            
            # In a real implementation, we would track the pipeline execution
            # and collect the final result from the last stage
            # For now, we simulate with a delay and dummy result
            time.sleep(2 * len(first_stage_workers))  # Simulate pipeline processing time
            
            # Dummy pipeline result
            result = f'Pipeline processed: {input_text[:50]}...'
            
        elif parallelism_type == 'tensor':
            # For tensor parallelism, send the same input to all workers
            for worker_ip, worker_socket in worker_connections.items():
                request = {
                    'command': 'process_input',
                    'input_data': {
                        'text': input_text,
                        'tokens': input_ids
                    },
                    'execution_info': {
                        'parallelism_type': 'tensor'
                    }
                }
                
                worker_socket.send(json.dumps(request).encode())
                logger.info(f'Sent input to tensor parallel worker: {worker_ip}')
            
            # In a real implementation, we would collect and aggregate results
            # For now, simulate with a delay and dummy result
            time.sleep(1.5)  # Simulate tensor parallel processing and aggregation
            
            # Dummy tensor result
            result = f'Tensor-parallel processed: {input_text[:50]}...'
            
        else:  # 'both' or any other value
            # For combined parallelism, find first stage workers
            first_stage_workers = [
                worker_ip for worker_ip, assignment in execution_plan['worker_assignments'].items()
                if assignment.get('pipeline_stage') == 0
            ]
            
            if not first_stage_workers:
                logger.error('No workers found for the first combined stage')
                return 'Error: Combined parallelism configuration error'
            
            # Send to all workers in the first stage (for tensor parallelism)
            for worker_ip in first_stage_workers:
                worker_socket = worker_connections.get(worker_ip)
                
                if worker_socket:
                    request = {
                        'command': 'process_input',
                        'input_data': {
                            'text': input_text,
                            'tokens': input_ids
                        },
                        'execution_info': {
                            'parallelism_type': 'combined',
                            'is_first_stage': True
                        }
                    }
                    
                    worker_socket.send(json.dumps(request).encode())
                    logger.info(f'Sent input to combined parallelism worker: {worker_ip}')
            
            # Simulate complex processing across pipeline stages and tensor partitions
            time.sleep(3)  # Longer time for combined parallelism
            
            # Dummy combined result
            result = f'Combined parallel processed: {input_text[:50]}...'
        
        logger.info('Distributed inference completed successfully')
        return result
    except Exception as e:
        logger.error('Distributed inference failed')
        logger.exception(str(e))
        return f'Error: {str(e)}'
