"""
Circuit breaker pattern (simplified)
"""
import time
from collections import defaultdict

_circuit_breakers = {}

class CircuitBreaker:
    def __init__(self, name, failure_threshold=5, recovery_timeout=60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

def get_circuit_breaker(name, failure_threshold=5, recovery_timeout=60):
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, failure_threshold, recovery_timeout)
    return _circuit_breakers[name]

def retry_on_error(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        raise
                    time.sleep(2 ** i)
            return None
        return wrapper
    return decorator
