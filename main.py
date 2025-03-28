import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from utils import get_llm_model, run_worker, distribute_model_to_workers, create_pipeline_execution_plan
import argparse
import time
import threading
import socket

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

def main():
    """
    Main function to parse arguments and execute the appropriate mode (master or worker).
    """
    # Create a logger instance
    logger = Logger("main")
    
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
    parser.add_argument('--parallelism_type', type=str, choices=['pipeline', 'tensor', 'both'], default='both',
                      help='Type of parallelism to use (default: both)')
    
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
            
            # Load the model
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
            
            # Distribute model to workers if we have any connected
            if worker_connections:
                logger.info(f"Distributing model to {len(worker_connections)} workers")
                success = distribute_model_to_workers(model, list(worker_connections.keys()), 
                                                    parallelism_type=args.parallelism_type)
                if success:
                    logger.info("Model distributed successfully")
                    
                    # Create a pipeline execution plan for coordinating inference
                    execution_plan = create_pipeline_execution_plan(
                        list(worker_connections.keys()), 
                        args.parallelism_type
                    )
                    logger.info(f"Created execution plan: {execution_plan}")
                else:
                    logger.error("Failed to distribute model")
            
            # For testing, keep the master node running
            logger.info("Master node is running. Press Ctrl+C to exit.")
            while True:
                time.sleep(10)
        
        elif args.mode == 'worker':
            logger.info(f"Starting in worker mode, connecting to master: {args.master_host}:{args.port}")
            worker_thread = threading.Thread(target=run_worker, args=(args.master_host, args.port))
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
