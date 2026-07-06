# Security Policy

## Supported Versions

| Version | Supported |
|---------|:---------:|
| latest  | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue.**
2. Use the [GitHub Security Advisory](https://github.com/transcend-0/VibeQuant/security/advisories/new) to report privately.
3. Include steps to reproduce, potential impact, and any suggested fixes.

We will acknowledge your report within **5 business days** and work with you to resolve the issue.

## Scope

This policy applies to the [transcend-0/VibeQuant](https://github.com/transcend-0/VibeQuant) repository.

## Data fetching

Market data loaders (`src/data_sources/`) call free, keyless public endpoints
(eastmoney, tencent, yahoo, OKX, Binance, etc.) over plain HTTPS/HTTP and
cache responses locally under `data/raw/` and `workspace/data_cache/`. Treat
cached data as untrusted input to parsing code the same way you would any
third-party API response.

## Sensitive configuration

- `config/llm.yaml` holds your LLM provider API key. It is gitignored by
  default — do not commit it, and do not paste it into issues or PRs.
- Signal deployment (`vq deploy`) sends email via SMTP credentials, which
  would live in a similarly gitignored `config/email.yaml`. Treat these the
  same as any other credential: local-only, never committed, never logged.

## Disclosure

- Please do not publicly disclose the vulnerability until we have released a fix.
- We will credit reporters in the release notes (unless you prefer anonymity).
