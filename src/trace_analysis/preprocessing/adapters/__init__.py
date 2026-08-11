from .collector import CollectorEventsAdapter
from .sls_proxy import SLSProxyAdapter
from .session_jsonl import SessionJSONLAdapter

ADAPTER_TYPES = (CollectorEventsAdapter, SLSProxyAdapter, SessionJSONLAdapter)


def choose_adapter(sample: dict, requested: str | None = None):
    adapters = [adapter_type() for adapter_type in ADAPTER_TYPES]
    if requested:
        for adapter in adapters:
            if adapter.name == requested:
                if not adapter.detect(sample):
                    raise ValueError(f"Input does not match requested adapter: {requested}")
                return adapter
        raise ValueError(f"Unknown adapter: {requested}")
    matches = [adapter for adapter in adapters if adapter.detect(sample)]
    if len(matches) != 1:
        names = ", ".join(adapter.name for adapter in matches) or "none"
        raise ValueError(f"Could not uniquely detect adapter; matches: {names}")
    return matches[0]
