On macOS, source fetches and bucket requests no longer take a proxy from the
system's network settings (issue #742). The binary compiled a second HTTP
client, reqwest 0.12, for the C2PA SDK, and its system proxy support was
switched on for every client in the binary: when `HTTPS_PROXY` and `HTTP_PROXY`
were unset, an HTTP or HTTPS proxy set in the macOS network settings carried
every fetch that honours a proxy. Only those variables and `ALL_PROXY` are
documented, and the boot warning about pinned fetches counts only them. They
are now the only proxy settings on every platform, as
`docs/configuration.md` says. Linux builds read only the variables before as
well.
