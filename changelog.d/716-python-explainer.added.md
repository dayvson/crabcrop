The Python SDK gains the TypeScript SDK's explainer and URL verifier (issue
#716). `explain_url(input, now_seconds=None)` parses a crabcrop URL or bare
path back into its signature, its segments (each with its kind, title, and a
detail or an error), its source, the remainder the signature covers, and
every error, as `explainUrl` does, message for message: it reads each number
as the server's types read it, reports the reserved, purge and upload
remainders and the 100-operation and 16-URL caps, and reads at most 64 KiB.
The answer is an `Explanation`, a frozen dataclass, with `ExplainedSignature`
and `ExplainedSegment` beside it, and `FILTER_DESCRIPTIONS` is the filter
table it explains with. `verify_url_signature(secret_or_keyring, input)`
answers as `verifyUrlSignature` does: `None` for an unsigned, sealed,
malformed or longer than 64 KiB URL and `False` for a reserved remainder.
With one secret it ignores the URL's key ID; with a keyring (a mapping of key
ID to secret), a keyed URL checks its named key only and an unkeyed one every
key. Tags compare with `hmac.compare_digest`. A string that holds
surrogate code points reads as JavaScript reads the same UTF-16, so neither
function raises for a URL or a secret of text. Class names, in the explainer
and in the typed builders, take Unicode 17.0's letters and digits, as the
server reads them, whatever the Unicode version of the interpreter's
`unicodedata`. The package's suite runs `spec/explain-verdicts.json` and
`spec/operation-vectors.json` through `explain_url`, and runs the two
explainers and the two verifiers against each other on 6,000 generated
inputs. The package stays standard-library only, and its suite passes on
Python 3.9 to 3.14.
