# Security and privacy

This repository is a research snapshot, not a hardened service. The ZeroMQ and
WebSocket transports do not authenticate clients, so both bind to loopback by
default. Do not expose either port directly to the internet. Use an authenticated
SSH tunnel for an authorized remote GPU worker.

## Reporting a problem

Report security issues privately through GitHub's **Security → Report a
vulnerability** flow when it is available. Do not post credentials, private
infrastructure addresses, facial media, embeddings, or other biometric data in a
public issue.

Include a minimal reproduction made with synthetic or non-identifying data. The
maintainer will acknowledge a usable report when available, assess its impact,
and coordinate a fix before public disclosure where appropriate.

## Operator responsibilities

- Keep source photographs, captured frames, model weights, and output outside Git.
- Confirm consent and the applicable model/data licenses before processing media.
- Restrict file permissions on identity material and define a deletion date.
- Treat logs and benchmark output as potentially sensitive.
- Add authentication, payload limits, origin checks, rate limits, and session
  isolation before considering any multi-user deployment.

See [Responsible use](docs/RESPONSIBLE_USE.md) for the project's consent and
disclosure requirements.
