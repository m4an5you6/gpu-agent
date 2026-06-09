# Langfuse Observability Plugin

This plugin ships bundled with GPUCLOUD but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
gpucloud tools  # → Langfuse Observability

# Manual
pip install langfuse
gpucloud plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.gpucloud/.env` (or via `gpucloud tools`):

```bash
GPUCLOUD_LANGFUSE_PUBLIC_KEY=pk-lf-...
GPUCLOUD_LANGFUSE_SECRET_KEY=sk-lf-...
GPUCLOUD_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

## Verify

```bash
gpucloud plugins list                 # observability/langfuse should show "enabled"
gpucloud chat -q "hello"              # then check Langfuse for a "GPUCLOUD turn" trace
```

## Optional tuning

```bash
GPUCLOUD_LANGFUSE_ENV=production       # environment tag
GPUCLOUD_LANGFUSE_RELEASE=v1.0.0       # release tag
GPUCLOUD_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
GPUCLOUD_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
GPUCLOUD_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Disable

```bash
gpucloud plugins disable observability/langfuse
```
