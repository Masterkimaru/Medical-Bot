import multiprocessing
import os

# Single worker for memory efficiency
workers = 1
worker_class = 'sync'
worker_connections = 1
timeout = 30
keepalive = 2
max_requests = 50
max_requests_jitter = 10

# Memory optimization
preload_app = False
graceful_timeout = 10

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
