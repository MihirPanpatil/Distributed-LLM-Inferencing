import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from utils import get_llm_model, run_worker
import argparse
import time
import threading

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
    
    try:
        if args.mode == 'master':
            logger.info(f"Starting in master mode with model: {args.model}")
            model = get_llm_model(args.model)
            # Start master node functionality here
            # For now, just keep the script running
            while True:
                time.sleep(10)
        
        elif args.mode == 'worker':
            logger.info(f"Starting in worker mode, connecting to master: {args.master_host}")
            worker_thread = threading.Thread(target=run_worker, args=(args.master_host,))
            worker_thread.daemon = True
            worker_thread.start()
            
            # Keep the main thread running
            while True:
                time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
