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

## Child processes

The server does some work in child processes: the SVG and AVIF worker, and,
when they are configured, the video encoder, the video frame decoder, and the
PDF tools. A hostile source that exploits a bug in one of their decoders takes
over the child, not the server. Each child:

- starts with an empty environment (the worker keeps only the font variables),
  so it is not handed the server's secrets;
- runs in its own process group, under a wall-clock limit and, on Linux, kernel
  limits on address space and CPU time.

On Linux the server marks itself non-dumpable at startup (issue #605). A child
then cannot read the server's environment or memory through `/proc`, or attach
to it. This holds for a child without capabilities, as the images run the
server: as `nobody`, with none. A privileged process, such as one with
`CAP_SYS_PTRACE`, can still read them. The mark has two side effects: the
server leaves no core dump, and unprivileged tools that run as the server's
user (a debugger, a profiler, a process exporter) can no longer read its
`/proc` files. If the kernel refuses the mark, the server says so on stderr and
starts anyway.

The children still run as the server's user, with no sandbox. A compromised
child can signal the server and read or write any file the server can: the
presets and source roots files, the C2PA signing key (`CRABCROP_C2PA_KEY`), and
the disk result tier (`CRABCROP_RESULT_STORAGE_DIR`). Running children under a
separate user or in a sandbox (seccomp or Landlock) is not decided yet; issue
#605 tracks it. Until then, configure only the tools you need, and give the
server's user no files it does not need.
