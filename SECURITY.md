# Security Policy

crabcrop fetches, processes, and serves images from URLs it is asked to, and it
makes outbound requests on behalf of its clients. We take reports about
server-side request forgery, access control, resource exhaustion, and
signed-URL integrity seriously. Thank you for helping keep crabcrop and the
people who run it safe.

## Supported versions

crabcrop is pre-1.0 and under active development. Security fixes are applied to
the latest `main` and included in the next release. Older commits and tags are
not backported yet. This policy will be revised once a stable 1.0 line exists.

| Version          | Supported          |
| ---------------- | ------------------ |
| latest `main`    | :white_check_mark: |
| latest release   | :white_check_mark: |
| anything older   | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.** Public disclosure before a fix is available
puts every operator at risk.

Report privately through GitHub's private vulnerability reporting:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability** to start a private advisory.

If private reporting is not available to you, contact the maintainer through
their GitHub profile and ask for a private channel before sharing any details.

Please include:

- the affected version or commit hash;
- the type of issue and its impact — what an attacker could achieve;
- steps to reproduce, including the relevant configuration and request shape;
- a proof of concept, if you have one;
- any suggested fix or mitigation.

## What to expect

- We aim to acknowledge a report within three business days.
- We confirm the issue, identify the affected versions, and prepare a fix.
- We practice coordinated disclosure: we agree a timeline with you and publish
  an advisory once a fix is available.
- We are glad to credit you in the advisory. Let us know your preference.

## Scope

In scope: the crabcrop server and the signing and SDK code in this repository —
source-URL handling and access controls, signed-URL verification, and the
resource limits that protect the service.

Out of scope: vulnerabilities in third-party dependencies (please report those
upstream, though a heads-up is welcome), and issues that require a
misconfiguration the documentation explicitly warns against.
