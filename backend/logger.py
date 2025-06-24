# logger.py
import os
import logging

# Read and normalize the DEBUG env variable
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Create a logger instance
logger = logging.getLogger('app_logger')

# Set the logging level
logger.setLevel(logging.DEBUG if DEBUG else logging.CRITICAL)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG if DEBUG else logging.CRITICAL)

# Define log message format
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger (if not already added)
if not logger.handlers:
    logger.addHandler(ch)
