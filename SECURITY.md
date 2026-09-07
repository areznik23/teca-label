# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub's **Security → Report a
vulnerability** on this repository (private vulnerability reporting). Do not open a
public issue for security reports. You should receive a response within a few days.

## Supported versions

Pre-1.0: only the latest release receives security fixes.

## Security posture (what to audit)

Teca Label is a BYO-credentials library. The properties a review should verify, and
that the test suite guards:

- **No artifact ever contains a secret.** Codebook files, logs, and pending files
  hold questions, category definitions, model *names*, and revision history — never keys,
  DSNs, or tokens. The artifact schema is a closed set (see
  `tests/test_credential_hygiene.py`).
- **Credentials are read, never stored.** Model keys come from the process
  environment (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, read by the vendor SDK) or an
  injected `client=`, which also carries any custom endpoint URL; the database DSN
  is a call argument passed verbatim to `psycopg2.connect` and is never formatted
  into any log, error message, or file.
- **The library never loads `.env`, writes config, or phones home.** There is no
  telemetry, no account, and no network egress except the model API you selected (or
  the endpoint of the client you inject) and the database you point it at.
