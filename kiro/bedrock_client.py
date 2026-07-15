"""
AWS Bedrock client for direct model invocation.

Bypasses the Kiro API entirely for models configured in BEDROCK_MODELS.
Resolves credentials via the AWS CLI (which handles SSO token refresh),
caches them, and auto-refreshes on expiry.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from typing import Any, AsyncGenerator, Dict, Optional

import boto3
from botocore.credentials import Credentials
from loguru import logger

from kiro.config import BEDROCK_REGION

REFRESH_BUFFER_SECONDS = 120


class BedrockClient:
    """Bedrock runtime wrapper with CLI-delegated credential refresh."""

    def __init__(self, region: str = BEDROCK_REGION):
        self._region = region
        self._client = None
        self._creds_expires_at: float = 0
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> None:
        """Refresh the boto3 client if credentials are stale."""
        if self._client and time.time() < self._creds_expires_at:
            return

        async with self._lock:
            if self._client and time.time() < self._creds_expires_at:
                return

            creds = await self._resolve_credentials()
            if creds:
                session = boto3.Session(
                    aws_access_key_id=creds["AccessKeyId"],
                    aws_secret_access_key=creds["SecretAccessKey"],
                    aws_session_token=creds.get("SessionToken"),
                    region_name=self._region,
                )
                self._client = session.client("bedrock-runtime")
                self._creds_expires_at = creds.get("_expires_epoch", time.time() + 3600) - REFRESH_BUFFER_SECONDS
                logger.debug("Bedrock credentials refreshed via AWS CLI")
            else:
                logger.warning("CLI credential export failed, falling back to default chain")
                self._client = boto3.client("bedrock-runtime", region_name=self._region)
                self._creds_expires_at = time.time() + 300

    async def _resolve_credentials(self) -> Optional[Dict[str, Any]]:
        """Shell out to AWS CLI to get fresh credentials (CLI handles SSO refresh)."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["aws", "configure", "export-credentials", "--format", "env-no-export"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(f"aws configure export-credentials failed: {result.stderr.strip()}")
                return None

            creds = {}
            for line in result.stdout.strip().splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    creds[key.strip()] = value.strip()

            if "AWS_ACCESS_KEY_ID" not in creds:
                return None

            return {
                "AccessKeyId": creds["AWS_ACCESS_KEY_ID"],
                "SecretAccessKey": creds["AWS_SECRET_ACCESS_KEY"],
                "SessionToken": creds.get("AWS_SESSION_TOKEN"),
                "_expires_epoch": time.time() + 3600,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Credential resolution via CLI failed: {e}")
            return None

    async def invoke_stream(
        self, model_id: str, body: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Call invoke_model_with_response_stream and yield parsed JSON chunks.

        Each yielded dict is an Anthropic SSE event (message_start,
        content_block_delta, etc.) extracted from Bedrock's EventStream frames.
        """
        await self._ensure_client()

        body_bytes = json.dumps(body).encode("utf-8")

        response = await asyncio.to_thread(
            self._client.invoke_model_with_response_stream,
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body_bytes,
        )

        event_stream = response["body"]

        for event in await asyncio.to_thread(list, event_stream):
            chunk_bytes = event.get("chunk", {}).get("bytes")
            if not chunk_bytes:
                continue
            try:
                yield json.loads(chunk_bytes)
            except json.JSONDecodeError as e:
                logger.warning(f"Bedrock: failed to parse chunk: {e}")
                continue

    async def invoke_collect(
        self, model_id: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming invocation. Returns the full response dict."""
        await self._ensure_client()

        body_bytes = json.dumps(body).encode("utf-8")

        response = await asyncio.to_thread(
            self._client.invoke_model,
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body_bytes,
        )

        response_body = json.loads(response["body"].read())
        return response_body
