"""Mutation tasks for pydantic-ai evaluation.

Each task introduces a single-line bug into the pydantic-ai codebase,
provides a test command that detects the bug, and includes both exact
(developer) and vague (user-reported) descriptions.

These mutations target core logic in pydantic-ai's source code (usage limits,
SSRF protection, exception handling, settings merge, token counting) — all
backed by the project's own test suite.

Repo: https://github.com/pydantic/pydantic-ai
"""

from __future__ import annotations

# Path prefix: relative to pydantic-ai repo root
SRC = "pydantic_ai_slim/pydantic_ai"

MUTATIONS = [
    # --- Usage / Token Limits ---
    {
        "task_id": "usage-total-tokens-math",
        "file": f"{SRC}/usage.py",
        "line_num": 66,
        "original": "return self.input_tokens + self.output_tokens",
        "mutated": "return self.input_tokens - self.output_tokens",
        "test_cmd": "python -m pytest tests/test_usage_limits.py::test_total_token_limit -x",
        "description": "UsageBase.total_tokens subtracts output tokens instead of adding them, making the total_tokens property return wrong values.",
        "vague_description": "Token usage tracking seems off. The total token count is always lower than expected, even lower than the input tokens alone.",
    },
    {
        "task_id": "usage-request-limit-off-by-one",
        "file": f"{SRC}/usage.py",
        "line_num": 369,
        "original": "if request_limit is not None and usage.requests >= request_limit:",
        "mutated": "if request_limit is not None and usage.requests > request_limit:",
        "test_cmd": "python -m pytest tests/test_usage_limits.py::test_retry_limit -x",
        "description": "UsageLimits.check_before_request uses > instead of >= for the request limit check, allowing one extra request beyond the limit.",
        "vague_description": "Setting request_limit=1 still allows 2 requests before stopping. The limit is off by one.",
    },
    {
        "task_id": "usage-check-tokens-output-inverted",
        "file": f"{SRC}/usage.py",
        "line_num": 391,
        "original": "if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:",
        "mutated": "if self.output_tokens_limit is None and output_tokens > self.output_tokens_limit:",
        "test_cmd": "python -m pytest tests/test_usage_limits.py::test_response_token_limit -x",
        "description": "check_tokens uses 'is None' instead of 'is not None' for output_tokens_limit, so the limit is never enforced.",
        "vague_description": "Output token limits are completely ignored. The agent generates unlimited output tokens even with output_tokens_limit set.",
    },
    {
        "task_id": "usage-incr-tokens-wrong-op",
        "file": f"{SRC}/usage.py",
        "line_num": 231,
        "original": "slf.input_tokens += incr_usage.input_tokens",
        "mutated": "slf.input_tokens -= incr_usage.input_tokens",
        "test_cmd": "python -m pytest tests/test_usage_limits.py::test_add_usages -x",
        "description": "_incr_usage_tokens subtracts input_tokens instead of adding, causing RunUsage.incr() to produce negative token counts.",
        "vague_description": "After multiple agent runs, the cumulative usage stats show negative input token counts. Something is wrong with the usage accumulation.",
    },
    {
        "task_id": "usage-tool-calls-limit-comparison",
        "file": f"{SRC}/usage.py",
        "line_num": 404,
        "original": "if tool_calls_limit is not None and tool_calls > tool_calls_limit:",
        "mutated": "if tool_calls_limit is not None and tool_calls < tool_calls_limit:",
        "test_cmd": "python -m pytest tests/test_usage_limits.py::test_tool_call_limit -x",
        "description": "check_before_tool_call raises UsageLimitExceeded when tool calls are BELOW the limit instead of above it, blocking tools prematurely.",
        "vague_description": "Tool calls are being rejected immediately even though the tool call limit hasn't been reached. The limit enforcement seems inverted.",
    },
    # --- SSRF Protection ---
    {
        "task_id": "ssrf-cloud-metadata-inverted",
        "file": f"{SRC}/_ssrf.py",
        "line_num": 79,
        "original": "return ip_str in _CLOUD_METADATA_IPS",
        "mutated": "return ip_str not in _CLOUD_METADATA_IPS",
        "test_cmd": "python -m pytest tests/test_ssrf.py::TestIsCloudMetadataIp -x",
        "description": "is_cloud_metadata_ip returns True for non-metadata IPs and False for actual cloud metadata endpoints — the check is inverted.",
        "vague_description": "URL downloads to public websites are being blocked as 'cloud metadata', but requests to 169.254.169.254 go through unchecked.",
    },
    {
        "task_id": "ssrf-protocol-validation",
        "file": f"{SRC}/_ssrf.py",
        "line_num": 145,
        "original": "if scheme not in ('http', 'https'):",
        "mutated": "if scheme not in ('http',):",
        "test_cmd": "python -m pytest tests/test_ssrf.py::TestValidateUrlProtocol -x",
        "description": "validate_url_protocol rejects HTTPS URLs, only allowing plain HTTP — breaking all secure downloads.",
        "vague_description": "All HTTPS URL downloads fail with a protocol error. Only plain HTTP URLs work, which defeats the security purpose.",
    },
    {
        "task_id": "ssrf-default-port-swap",
        "file": f"{SRC}/_ssrf.py",
        "line_num": 169,
        "original": "default_port = 443 if is_https else 80",
        "mutated": "default_port = 80 if is_https else 443",
        "test_cmd": "python -m pytest tests/test_ssrf.py::TestExtractHostAndPort -x",
        "description": "extract_host_and_port swaps default ports — HTTPS gets port 80 and HTTP gets port 443, breaking URL resolution.",
        "vague_description": "URL downloads fail silently. HTTPS connections seem to be going to the wrong port.",
    },
    {
        "task_id": "ssrf-private-ip-allow-local-inverted",
        "file": f"{SRC}/_ssrf.py",
        "line_num": 241,
        "original": "if not allow_local and is_private_ip(ip):",
        "mutated": "if allow_local and is_private_ip(ip):",
        "test_cmd": "python -m pytest tests/test_ssrf.py::TestValidateAndResolveUrl::test_private_ip_blocked_by_default -x",
        "description": "The allow_local guard is inverted — private IPs are blocked when allow_local=True and allowed when allow_local=False.",
        "vague_description": "Setting allow_local=True blocks local network requests, and removing it allows them. The flag seems to do the opposite of what it should.",
    },
    # --- Exception Handling ---
    {
        "task_id": "tool-retry-isinstance-inverted",
        "file": f"{SRC}/exceptions.py",
        "line_num": 205,
        "original": "if isinstance(tool_retry.content, str)",
        "mutated": "if not isinstance(tool_retry.content, str)",
        "test_cmd": "python -m pytest tests/test_exceptions.py::test_tool_retry_error_str_with_string_content -x",
        "description": "ToolRetryError._format_error_details inverts the isinstance check, passing string content to dict-processing code and crashing.",
        "vague_description": "Tool retry errors crash with a TypeError when the tool returns a simple string error message. Only structured errors work.",
    },
    {
        "task_id": "tool-retry-error-format",
        "file": f"{SRC}/exceptions.py",
        "line_num": 221,
        "original": 'f\'{error_count} validation error{"" if error_count == 1 else "s"}{f" for {tool_name!r}" if tool_name else ""}\'',
        "mutated": 'f\'{error_count} validation error{"s" if error_count == 1 else ""}{f" for {tool_name!r}" if tool_name else ""}\'',
        "test_cmd": "python -m pytest tests/test_exceptions.py::test_tool_retry_error_str_with_error_details -x",
        "description": "ToolRetryError pluralization is inverted — says 'errors' for 1 error and 'error' for multiple.",
        "vague_description": "Tool retry error messages have wrong grammar. A single validation error says 'errors' plural.",
    },
    # --- Settings Merge ---
    {
        "task_id": "settings-merge-priority",
        "file": f"{SRC}/settings.py",
        "line_num": 192,
        "original": "return base | overrides",
        "mutated": "return overrides | base",
        "test_cmd": "python -m pytest tests/test_agent.py::test_model_settings_override -x",
        "description": "merge_model_settings gives base settings priority over overrides instead of the other way around, ignoring per-run settings.",
        "vague_description": "Model settings passed to agent.run() are silently ignored. The agent always uses the settings from initialization.",
    },
    # --- Private IP detection ---
    {
        "task_id": "ssrf-ipv4-mapped-ipv6-skip",
        "file": f"{SRC}/_ssrf.py",
        "line_num": 91,
        "original": "if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:",
        "mutated": "if isinstance(ip, ipaddress.IPv6Address) and not ip.ipv4_mapped:",
        "test_cmd": "python -m pytest tests/test_ssrf.py::TestIsPrivateIp::test_ipv4_mapped_ipv6_private -x",
        "description": "is_private_ip skips IPv4-mapped IPv6 unwrapping for actual mapped addresses, failing to detect private IPs in IPv6 form.",
        "vague_description": "Private IP addresses wrapped in IPv6 notation bypass the SSRF protection. For example ::ffff:192.168.1.1 is not blocked.",
    },
    # --- Redirect handling ---
    {
        "task_id": "ssrf-max-redirects-zero",
        "file": f"{SRC}/_ssrf.py",
        "line_num": 50,
        "original": "_MAX_REDIRECTS = 10",
        "mutated": "_MAX_REDIRECTS = 0",
        "test_cmd": "python -m pytest tests/test_ssrf.py::TestSafeDownload::test_redirect_followed -x",
        "description": "_MAX_REDIRECTS is set to 0, preventing any HTTP redirects from being followed during safe downloads.",
        "vague_description": "URL downloads that involve redirects always fail. The downloader seems unable to follow even a single redirect.",
    },
]
