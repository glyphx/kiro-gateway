# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py                    # localhost:8000
python main.py --port 9000 --host 0.0.0.0

# Tests
pytest                            # all tests
pytest tests/unit/ -v            # unit only
pytest tests/integration/ -v     # integration only
pytest tests/unit/test_auth.py   # single file
pytest -x                        # stop on first failure
pytest --cov=kiro --cov-report=html

# Docker
docker-compose up -d
```

## Architecture

**Entry point:** `main.py` → FastAPI app with two route groups: OpenAI-compatible (`routes_openai.py`) and Anthropic-compatible (`routes_anthropic.py`).

**Request flow:**
1. Route handler authenticates via `auth.py` (KiroAuthManager)
2. **Bedrock bypass check**: If model matches `BEDROCK_MODELS` config, route directly to AWS Bedrock (native Anthropic Messages API format, no Kiro conversion)
3. Request converted to Kiro format: `converters_openai.py` or `converters_anthropic.py` (shared logic in `converters_core.py`)
4. Sent to Kiro API via `http_client.py`
5. Response streamed back via `streaming_openai.py` or `streaming_anthropic.py` (shared in `streaming_core.py`)

**Key modules:**
- `auth.py` — token lifecycle; supports Kiro Desktop, AWS SSO OIDC, env var refresh token
- `model_resolver.py` — 4-layer pipeline: normalize → cache → hidden models → passthrough
- `bedrock_client.py` — boto3 wrapper for Bedrock invoke_model_with_response_stream
- `bedrock_streaming.py` — Bedrock EventStream to Anthropic SSE adapter
- `parsers.py` — AWS EventStream parser + tool call parsing
- `thinking_parser.py` — FSM for extended thinking blocks
- `truncation_recovery.py` / `truncation_state.py` — synthetic message generation for truncation recovery
- `cache.py` — thread-safe model metadata cache with TTL
- `debug_logger.py` / `debug_middleware.py` — configurable request/response logging

**Bedrock bypass:**
Models listed in `BEDROCK_MODELS` (config.py) skip the Kiro API entirely. Requests are sent as native Anthropic Messages API calls to Bedrock via boto3. Extended thinking is supported natively (no fake reasoning tag injection). AWS credentials come from the default credential chain.

Env vars:
- `BEDROCK_REGION` — AWS region for Bedrock calls (default: `us-east-1`)

To add more Bedrock-routed models, edit the `BEDROCK_MODELS` dict in `kiro/config.py`.

**Tests:** All tests in `tests/unit/` and `tests/integration/`. A global `block_all_network_calls` fixture in `conftest.py` prevents real network calls — all external calls must be mocked.
