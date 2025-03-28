import os
import sys
import logging
import json
from logging.handlers import RotatingFileHandler
import argparse
import time
import threading
import socket
from utils import get_llm_model, shard_model, distribute_model_to_workers, run_worker, distributed_inference

class Logger:
    """
    Logger class for handling logging operations consistently throughout the application.
    Supports logging to both console and file with configurable log levels.
    """
    def __init__(self, name, log_file='inferencing.log', level=logging.INFO):
        """
        Initialize the logger with console and file handlers.
        
        Args:
            name (str): Name of the logger
            log_file (str): Path to the log file
            level (int): Logging level
        """
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers to avoid duplicates
        
        # Create formatters for console and file
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        """Log an info level message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning level message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error level message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical level message."""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log an exception with traceback."""
        self.logger.exception(message)

def start_master_server(port=5555):
    """
    Start a server socket on the master node to accept worker connections.
    
    Args:
        port (int): Port to listen on
        
    Returns:
        socket.socket: The server socket
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(10)
    return server_socket

def handle_worker_connections(server_socket, worker_connections, logger):
    """
    Accept and handle incoming worker connections.
    
    Args:
        server_socket (socket.socket): The server socket
        worker_connections (dict): Dictionary to store worker connections
        logger (Logger): Logger instance
    """
    while True:
        try:
            client_socket, address = server_socket.accept()
            worker_ip = address[0]
            logger.info(f'Worker connected from {worker_ip}')
            worker_connections[worker_ip] = client_socket
        except Exception as e:
            logger.error(f'Error accepting worker connection: {str(e)}')
            time.sleep(1)

def create_pipeline_execution_plan(worker_ips, parallelism_type):
    """
    Create a pipeline execution plan for distributed inference with enhanced
    support for pipeline and tensor parallelism.
    
    Args:
        worker_ips (list): List of worker IP addresses
        parallelism_type (str): Type of parallelism ('pipeline', 'tensor', or 'both')
        
    Returns:
        dict: Execution plan for distributed inference
    """
    num_workers = len(worker_ips)
    execution_plan = {
        'parallelism_type': parallelism_type,
        'num_workers': num_workers,
        'worker_assignments': {},
        'communication_pattern': {},
        'execution_sequence': []
    }
    
    if parallelism_type == 'pipeline':
        # In pipeline parallelism, each worker handles a sequential part of the model
        for i, worker_ip in enumerate(worker_ips):
            execution_plan['worker_assignments'][worker_ip] = {
                'task_type': 'pipeline_stage',
                'stage_id': i,
                'receives_from': worker_ips[i-1] if i > 0 else None,
                'sends_to': worker_ips[i+1] if i < num_workers - 1 else None
            }
            
        # Define the sequential flow of data
        for i in range(num_workers - 1):
            execution_plan['communication_pattern'][worker_ips[i]] = [worker_ips[i+1]]
            
        # Define execution sequence (pipeline order)
        execution_plan['execution_sequence'] = worker_ips
            
    elif parallelism_type == 'tensor':
        # In tensor parallelism, each worker handles part of each layer's computation
        for i, worker_ip in enumerate(worker_ips):
            execution_plan['worker_assignments'][worker_ip] = {
                'task_type': 'tensor_partition',
                'partition_id': i,
                'total_partitions': num_workers,
                'all_workers': worker_ips,
                'weight_range': {
                    'start': i / num_workers,
                    'end': (i + 1) / num_workers
                }
            }
            
        # All workers need to communicate with each other for tensor ops
        for worker_ip in worker_ips:
            execution_plan['communication_pattern'][worker_ip] = [
                w for w in worker_ips if w != worker_ip
            ]
            
        # In tensor parallelism, execution is parallel
        execution_plan['execution_sequence'] = [worker_ips]
            
    else:  # 'both'
        # Combine pipeline and tensor parallelism
        pipeline_stages = max(2, num_workers // 2)
        tensor_partitions_per_stage = max(2, num_workers // pipeline_stages)
        
        worker_idx = 0
        stage_workers_list = []
        
        for stage in range(pipeline_stages):
            stage_workers = []
            
            for partition in range(min(tensor_partitions_per_stage, num_workers - worker_idx)):
                if worker_idx < num_workers:
                    worker_ip = worker_ips[worker_idx]
                    execution_plan['worker_assignments'][worker_ip] = {
                        'task_type': 'combined',
                        'pipeline_stage': stage,
                        'tensor_partition': partition,
                        'total_partitions_in_stage': min(tensor_partitions_per_stage, num_workers - (stage * tensor_partitions_per_stage)),
                        'weight_range': {
                            'start': partition / tensor_partitions_per_stage,
                            'end': (partition + 1) / tensor_partitions_per_stage
                        }
                    }
                    stage_workers.append(worker_ip)
                    worker_idx += 1
            
            stage_workers_list.append(stage_workers)
            
            # Define communication patterns for pipeline and tensor parallelism
            if stage > 0:
                previous_stage_workers = stage_workers_list[stage - 1]
                
                # Pipeline communication between stages
                for worker in stage_workers:
                    execution_plan['communication_pattern'].setdefault(worker, []).extend(previous_stage_workers)
                
                for worker in previous_stage_workers:
                    execution_plan['communication_pattern'].setdefault(worker, []).extend(stage_workers)
            
            # Tensor communication within stages
            for worker in stage_workers:
                other_stage_workers = [w for w in stage_workers if w != worker]
                execution_plan['communication_pattern'].setdefault(worker, []).extend(other_stage_workers)
        
        # Define execution sequence for combined approach (stage by stage)
        execution_plan['execution_sequence'] = stage_workers_list
    
    return execution_plan

def process_inference_request(input_text, model, tokenizer, worker_connections, execution_plan, logger):
    """
    Process an inference request using distributed inference with enhanced support for 
    both pipeline and tensor parallelism.
    
    Args:
        input_text (str): Input text for inference
        model: The model object (may be None if using only workers)
        tokenizer: The tokenizer for the model
        worker_connections (dict): Dictionary of worker connections
        execution_plan (dict): Execution plan for distributed inference
        logger (Logger): Logger instance
        
    Returns:
        str: Generated response
    """
    logger.info('Processing inference request through distributed system')
    
    # For true distributed inference, use the distributed_inference function
    return distributed_inference(model, tokenizer, input_text, worker_connections, execution_plan)

def main():
    """
    Main function to parse arguments and execute the appropriate mode (master or worker).
    """
    # Create a logger instance
    logger = Logger('main')
    
    parser = argparse.ArgumentParser(description='Local Distributed Parallel LLM Inferencing')
    parser.add_argument('--mode', type=str, choices=['master', 'worker'], 
                      help='Run in master or worker mode')
    parser.add_argument('--model', type=str, default='gpt2',
                      help='Model name from Huggingface (default: gpt2)')
    parser.add_argument('--master_host', type=str, 
                      help='Master node hostname (required for worker mode)')
    parser.add_argument('--workers', type=str, nargs='+',
                      help='List of worker node IPs/hostnames (required for master mode)')
    parser.add_argument('--port', type=int, default=5555,
                      help='Port for communication (default: 5555)')
    parser.add_argument('--parallelism_type', type=str, choices=['pipeline', 'tensor', 'both'], 
                      default='both', help='Type of parallelism to use (default: both)')
    parser.add_argument('--input_text', type=str, 
                      help='Input text for inference testing (master mode only)')
    parser.add_argument('--num_layers_per_stage', type=int, default=2,
                      help='Number of transformer layers per pipeline stage (default: 2)')
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                      help='Size of tensor parallelism dimension (default: 2)')
    
    args = parser.parse_args()
    
    # Check if mode is provided
    if not args.mode:
        logger.error("Mode (--mode) must be specified as 'master' or 'worker'")
        parser.print_help()
        sys.exit(1)
    
    # Check if master_host is provided for worker mode
    if args.mode == 'worker' and not args.master_host:
        logger.error("Master hostname (--master_host) must be specified for worker mode")
        parser.print_help()
        sys.exit(1)
    
    # Check if workers are provided for master mode
    if args.mode == 'master' and not args.workers:
        logger.warning("No worker nodes specified. Running in single-node mode.")
    
    try:
        if args.mode == 'master':
            logger.info(f"Starting in master mode with model: {args.model}")
            logger.info(f"Using parallelism type: {args.parallelism_type}")
            
            # Load the model and tokenizer
            model, tokenizer = get_llm_model(args.model)
            if model is None or tokenizer is None:
                logger.error("Failed to load model. Exiting.")
                sys.exit(1)
            
            # Start a server to accept worker connections
            server_socket = start_master_server(args.port)
            logger.info(f"Master server started, listening on port {args.port}")
            
            # Dictionary to store worker connections
            worker_connections = {}
            
            # Start a thread to handle incoming worker connections
            worker_handler = threading.Thread(
                target=handle_worker_connections, 
                args=(server_socket, worker_connections, logger)
            )
            worker_handler.daemon = True
            worker_handler.start()
            
            # Wait for workers to connect
            if args.workers:
                logger.info(f"Waiting for {len(args.workers)} workers to connect...")
                timeout = 30  # seconds
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    connected_workers = set(worker_connections.keys())
                    expected_workers = set(args.workers)
                    
                    if connected_workers.issuperset(expected_workers):
                        logger.info("All workers connected successfully")
                        break
                    
                    logger.info(f"Connected workers: {len(connected_workers)}/{len(expected_workers)}")
                    time.sleep(2)
                
                # Check if all workers connected
                if not set(worker_connections.keys()).issuperset(set(args.workers)):
                    logger.warning("Not all workers connected within the timeout period")
            
            # Create model shards based on parallelism type
            if worker_connections:
                logger.info(f"Creating model shards for {len(worker_connections)} workers")
                
                # Additional sharding parameters
                sharding_params = {
                    'num_layers_per_stage': args.num_layers_per_stage,
                    'tensor_parallel_size': args.tensor_parallel_size
                }
                
                # Initialize shards with enhanced sharding logic
                shards = shard_model(model, len(worker_connections), args.parallelism_type, **sharding_params)
                logger.info(f"Created {len(shards)} shards using {args.parallelism_type} parallelism")
                
                # Create enhanced execution plan for distributed inference
                execution_plan = create_pipeline_execution_plan(
                    list(worker_connections.keys()),
                    args.parallelism_type
                )
                logger.info("Created pipeline execution plan with improved coordination")
                
                # Distribute model shards to workers with enhanced distribution
                success = distribute_model_to_workers(
                    model, 
                    list(worker_connections.keys()),
                    args.parallelism_type,
                    shards=shards,
                    execution_plan=execution_plan
                )
                
                if success:
                    logger.info("Model shards distributed successfully across workers")
                    
                    # Process an inference request if input_text is provided
                    if args.input_text:
                        logger.info(f"Processing test inference request: {args.input_text}")
                        result = process_inference_request(
                            args.input_text,
                            model,
                            tokenizer,
                            worker_connections,
                            execution_plan,
                            logger
                        )
                        logger.info(f"Distributed inference result: {result}")
                else:
                    logger.error("Failed to distribute model shards to workers")
            
            # Keep the master node running to accept commands
            logger.info("Master node is running. Press Ctrl+C to exit.")
            while True:
                time.sleep(10)
        
        elif args.mode == 'worker':
            logger.info(f"Starting in worker mode, connecting to master: {args.master_host}:{args.port}")
            worker_thread = threading.Thread(target=run_worker, args=(args.master_host, args.port, args.parallelism_type))
            worker_thread.daemon = True
            worker_thread.start()
            
            # Keep the main thread running
            logger.info("Worker node is running. Press Ctrl+C to exit.")
            while True:
                time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if args.mode == 'master' and 'server_socket' in locals():
            server_socket.close()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
