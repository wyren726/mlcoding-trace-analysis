from types import SimpleNamespace
import pytest
from trace_analysis.model_api.service_backoff import ServiceBackoff, ServiceRetryExhausted
from trace_analysis.model_api.client import OpenAICompatibleClient


def test_schedule_and_durable_budget(tmp_path, monkeypatch):
    monkeypatch.setattr('trace_analysis.model_api.service_backoff.random.uniform', lambda a,b: 1)
    monkeypatch.setattr('trace_analysis.model_api.service_backoff.time.time', lambda: 1000)
    guard = ServiceBackoff(tmp_path)
    for delay in (15,45,120):
        guard.result('request', True)
        assert guard.state('request')['retry_at'] == 1000+delay
    with pytest.raises(ServiceRetryExhausted):
        guard.result('request', True)
    with pytest.raises(ServiceRetryExhausted):
        ServiceBackoff(tmp_path).before('request')


def test_cooldown_single_probe_and_recovery(tmp_path, monkeypatch):
    now = [1000]
    monkeypatch.setattr('trace_analysis.model_api.service_backoff.time.monotonic', lambda: now[0])
    guard = ServiceBackoff(tmp_path)
    for i in range(5):
        guard.result(str(i), True)
    assert guard.blocked_until == 1180
    now[0] = 1181
    guard.before('new-request')
    assert guard.probing
    guard.result('new-request', True)
    assert guard.blocked_until == 1361 and not guard.probing
    guard.result('successful-request', False)
    assert not guard.blocked_until and guard.consecutive == 0


def test_503_does_not_stack_transport_retries(tmp_path, monkeypatch):
    class Unavailable(Exception):
        status_code = 503
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise Unavailable()
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.model = 'test'
    client.cache_dir = tmp_path
    client.max_retries = 9
    client.service_backoff = ServiceBackoff(tmp_path/'retries')
    monkeypatch.setattr(client.service_backoff, 'delays', (0,0,0))
    client._wait_for_request_slot = lambda: None
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ServiceRetryExhausted):
        client.complete_json('system','user')
    assert len(calls) == 4
    with pytest.raises(ServiceRetryExhausted):
        client.complete_json('system','user')
    assert len(calls) == 4
