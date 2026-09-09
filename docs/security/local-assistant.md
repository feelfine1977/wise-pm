# The local assistant: threat model and what it does not promise

*Status: experimental, opt-in, added on the extension branch. Nothing in this
document is switched on by default, and `import wise` neither imports
`wise.llm` nor contacts anything.*

This page states what `wise.llm` defends against, how, and — the more useful
half — what it does **not** defend against and what the deploying organisation
still has to do.

## The one-sentence version

A model on this path can ask for one of six named read-only tools and propose
an order for facts that already exist. It cannot compute a number, reach data
the access policy does not allow, name a tool that is not in the frozen set,
cause an import, an `eval`, a shell or a path to be used, follow a redirect off
the loopback interface, approve anything, or make a report drop a mandatory
qualification.

## The trust boundary

```text
    application                        library                    local server
 ┌────────────────┐            ┌──────────────────────┐        ┌──────────────┐
 │ authenticates  │  policy →  │ AccessPolicy         │        │ Ollama       │
 │ the principal  │            │  ├ scopes            │        │ already      │
 │ owns prompts,  │  packet →  │  ├ row mask  ────────┼─ before│ running,     │
 │ audit, approval│            │  └ column/doc filter │  every │ model        │
 └────────────────┘            │ ToolGateway (6 names)│  aggregate already    │
                               │ StrictLocalTransport ├───────►│ installed    │
        untrusted ───────────► │ retrieval (data only)│  loopback only        │
        documents, labels,     │ NormDraft validation │        └──────────────┘
        model replies          └──────────────────────┘
```

Everything crossing into the library from the right or the bottom is **data**.
Everything that grants authority comes from the left, from the application, as
an argument.

## What is enforced, and where

| Threat | Guard | Where |
|---|---|---|
| A reply names a tool that does not exist | frozen `APPROVED_TOOLS`, checked before argument validation and before dispatch | `llm/gateway.py` |
| A reply reaches a callable by name | dispatch table written out literally; no `getattr`, `eval`, dynamic import or shell anywhere on the path | `llm/gateway.py` |
| A reply smuggles a path or URL into an argument | every string argument screened for scheme, control characters, traversal and absolute/home prefixes; no tool accepts a path at all | `llm/gateway.py` |
| A reply asks for rows, columns, runs, views or documents it may not see | `AccessPolicy` fails closed; the row mask runs **before** any aggregate | `llm/policy.py` |
| A restricted reader is shown a company-wide comparator | `authorised_baseline` recomputes in the authorised population and relabels the comparator | `llm/policy.py` |
| A retrieved document issues instructions | permissions and callables come from the application, never from content; suspicious spans are flagged and delimited | `llm/retrieval.py` |
| Two approved policies disagree | both cited, a `ReviewQuestion` raised, neither chosen | `llm/retrieval.py` |
| Vague language becomes an invented threshold | `clarification_questions` and `check_threshold_support` mark it unresolved | `llm/drafts.py` |
| A reply is unparseable, oversize, looping or slow | typed `ProviderStatus`; the deterministic report still renders | `llm/provider.py`, `llm/ollama.py` |
| A reply cites facts or evidence that do not exist | `check_references` against the authorised packet, after schema validation | `llm/assistant.py` |
| A reply drops a mandatory qualification | the renderer prints every `packet.limitations` entry regardless | `llm/assistant.py` |
| A draft evaluates an expression | `kind="eval"`, unknown kinds, unknown evaluators, unbounded regexes and out-of-range shapes refused **before** any derivation | `llm/drafts.py` |
| A draft approves itself | `review_status` is a constant; an unknown envelope key is refused, not dropped | `llm/drafts.py` |
| A draft patches a norm it was not written against | stale parent fingerprint raises `DraftConflict` | `llm/drafts.py` |
| A preview mutates the approved configuration | an isolated log copy, fresh caches; `applied` and `activated` have no code path that sets them | `llm/drafts.py` |
| A "local" request leaves the machine | loopback-literal default validated at configuration time, route allowlist, redirects refused, final host compared, inherited proxy dropped | `llm/transport.py` |
| A missing server or model triggers a fallback | there is no second endpoint in the code; absence is `ProviderStatus.UNAVAILABLE` | `llm/ollama.py` |

## What this does **not** promise

**Loopback binding is not confidentiality.** `127.0.0.1` means other hosts
cannot reach the port. It says nothing about other users or processes on the
same machine, about what the model server writes to its own logs, about how
long it keeps prompts, or about what the server itself may contact. Treat the
model server as a separate system with its own security posture.

**The library does not configure your server.** It never sets an environment
variable, edits a service file, opens or closes a firewall port, pulls a
model, or starts or stops anything. The deployment decisions below are yours,
and this library will not make them for you:

- run the server with cloud routing disabled — for Ollama, `OLLAMA_NO_CLOUD=1`
  in the service environment — so an unrecognised model name cannot become a
  remote call;
- keep models in a local cache with a known provenance, and record the digest
  your deployment approved;
- bind the server to loopback and add host-level egress restrictions if the
  machine can reach the internet at all;
- decide the server's logging and retention: prompts on this path can contain
  authorised evidence and approved policy text;
- if you must run non-loopback, `TransportConfig(allow_non_loopback=True,
  reviewed_hosts=(...))` is deliberately awkward, and it is not enough on its
  own: that deployment needs its own transport security, authentication and
  network controls.

**A passing test suite is not a safe deployment.** Every test in this
repository runs against a fake provider. They establish that the guards above
behave as described; they establish nothing about a particular model, a
particular prompt, or the quality of anything a model writes.

**Structure is not truth.** A draft that validates against its schema and
cites real identifiers can still be wrong. That is why hypotheses are printed
in their own section, labelled unverified, under a sentence saying that valid
identifiers show what was looked at and not that the claim follows.

**The library is not an approval service.** It stores and validates a
proposal. Authorisation, the actor who gave it, and the audit trail belong to
the application.

## Checking that the boundary holds, offline

`wise.evaluation.llm` is a harness for the guarantees above. A scenario is
recorded material — a reply a model once produced, a document somebody wrote, a
stale fingerprint — plus the outcome this document declares for it. The runner
puts each one through the real gateway, the real access policy, the real draft
reader and the real evidence packet, and reports what was asked, what happened,
and whether that is the declared behaviour:

```python
from wise.evaluation.llm import load_tasks, run_tasks

report = run_tasks(load_tasks("tests/fixtures/llm_tasks"))
print(report.render())
assert report.ok
```

The shipped scenarios cover an unauthorised population, a re-derived
comparator, an invented fact id, an unknown tool, a poisoned document, an
oversize reply and a stale fingerprint in both places one exists — plus a
**negative control** that is served whole, because a harness that refused
everything would pass every refusal test.

What it measures is the boundary, not a model, and the difference is the point:
the answer does not change when the model does. It says nothing about quality,
latency or whether a reviewer is better off. A scenario that would need a model
declares `requires_env` and is skipped with that variable named; running it
needs both the variable and a `live_provider=` the caller builds, because this
module constructs no provider that can reach a network.

## The opt-in live path

`tests/extensions/test_live_ollama.py` is the only file that can contact a
server, and it is skipped unless `WISE_OLLAMA_LIVE=1` and `WISE_OLLAMA_MODEL`
are set. It checks the *shape* of the exchange. It measures no quality and no
latency, and a passing run establishes neither.
