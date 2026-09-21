"""Bounded, durable 503 retries and a shared outage cooldown."""
import json
import logging
import random
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class ServiceRetryExhausted(RuntimeError):
    pass


class ServiceBackoff:
    delays = (15, 45, 120)

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition()
        self.consecutive = 0
        self.blocked_until = 0.0
        self.probing = False

    def state(self, key):
        path = self.directory / (key + '.json')
        return json.loads(path.read_text()) if path.exists() else {'failures': 0}

    def before(self, key):
        with self.condition:
            while True:
                state = self.state(key)
                if state.get('exhausted'):
                    raise ServiceRetryExhausted(f'503 retry budget exhausted; request_key={key}; retained for review')
                delay = max(state.get('retry_at', 0) - time.time(), self.blocked_until - time.monotonic())
                if delay > 0:
                    self.condition.wait(min(delay, 30))
                elif self.blocked_until:
                    if self.probing:
                        self.condition.wait(30)
                    else:
                        self.probing = True
                        return
                else:
                    return

    def result(self, key, unavailable):
        with self.condition:
            if not unavailable:
                self.consecutive = 0
                self.blocked_until = 0.0
                self.probing = False
                self.condition.notify_all()
                return
            state = self.state(key)
            failures = state['failures'] + 1
            exhausted = failures >= 4
            delay = self.delays[failures - 1] * random.uniform(.9, 1.1) if not exhausted else 0
            value = dict(failures=failures, exhausted=exhausted, retry_at=time.time()+delay,
                         updated_at=time.time(), policy='503-backoff-v1')
            path = self.directory / (key + '.json')
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value)+'\n')
            temporary.replace(path)
            self.consecutive += 1
            if self.probing or self.consecutive >= 5:
                self.blocked_until = time.monotonic() + 180
                self.probing = False
                log.warning('503 service cooldown: pause new requests for 180s, then one probe')
            self.condition.notify_all()
            log.warning('503 request %s failure %s/4; delay %.1fs; exhausted=%s', key, failures, delay, exhausted)
            if exhausted:
                raise ServiceRetryExhausted(f'503 retry budget exhausted; request_key={key}; retained for review')
