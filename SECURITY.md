# Security policy

## Supported versions

Local AI Doctor is currently pre-release software. Security fixes are applied to the current `dev` line; no older release line is maintained yet.

## Reporting a vulnerability

Do not open a public issue for a vulnerability, credential exposure, arbitrary file access, model-code execution, authentication bypass, or unsafe container behavior. Use the repository host's private security-advisory facility when available, or contact the repository owner through an established private channel. Include:

- affected revision and platform;
- configuration needed to reproduce, with secrets and personal paths removed;
- minimal reproduction and observed impact;
- whether a malicious model, upload, browser origin, or authenticated user is required;
- any evidence that private data or credentials were exposed.

Do not include real prompts, uploads, model weights, tokens, database contents, access tokens, or private filesystem paths. Allow the maintainer time to reproduce and coordinate a fix before public disclosure.

## Intended threat model

The default service is a single-user application bound to loopback. It assumes the operating-system account and local application data directory are trusted. It does not provide multi-tenant isolation.

The following are untrusted inputs:

- model directories, SafeTensors metadata, configuration JSON, tokenizer files, templates, and processors;
- upload names, declared MIME types, bytes, and media decoders;
- chat content, model output, and API values;
- browser origins and WebSocket clients.

The spawned worker provides crash and memory-failure isolation from the web process. It is not a security sandbox. A user who points the application at malicious local files should not assume OS-level containment.

## Security properties

- The default listener is `127.0.0.1`.
- Non-loopback binding is rejected unless `allow_external_access=true` and an authentication token is configured.
- HTTP bearer comparisons and WebSocket token comparisons use constant-time matching.
- State-changing browser requests and WebSockets accept the effective same origin or a configured allowed origin and reject other supplied origins.
- Responses set a restrictive Content Security Policy, `nosniff`, no-referrer, and disabled camera/microphone/geolocation permissions.
- Transformers and Hugging Face operate offline in the worker with `local_files_only=true` and `trust_remote_code=false`.
- Discovery performs bounded metadata/header reads and does not load weights.
- Uploads are streamed under a byte limit, classified by signature, hashed, and stored under a content-derived name. Resolution rejects invalid names, traversal, and symlinks.
- Portable chat imports use a strict versioned schema, reject unknown fields and invalid/cyclic identifier graphs, remap IDs, and do not accept application filesystem-reference fields.
- SQLite queries parameterize user values; schema mutations are checked-in migrations.
- Public application errors and configuration views redact arbitrary paths and secrets. Prompts and output are prohibited from normal logging.
- Containers run as a non-root UID, drop all capabilities, use `no-new-privileges`, mount the root filesystem and model root read-only, and keep writable state in explicit volumes.

These controls reduce risk; they are not a guarantee that every upstream tokenizer, model parser, image/video decoder, GPU driver, or numerical library is vulnerability-free.

## Data at rest

The SQLite database may contain chats, complete or partial generated output, rendered prompts, model fingerprints, settings, timing, probability telemetry, and short embedding input previews. The upload directory contains original upload bytes under content-addressed filenames. Optional embedding-vector persistence stores vectors in SQLite.

Cache, export, backup, and container volume directories can also contain sensitive derived state. Protect them with operating-system permissions and storage encryption appropriate to the data. Backups are not encrypted by the application. Deleting a chat cascades its database records, but does not currently garbage-collect unreferenced content-addressed upload files or copies outside application storage.

Portable workspace and run exports contain prompt/output and potentially detailed token telemetry; treat downloaded copies as sensitive. The confirmed telemetry-retention endpoint deletes old terminal run records and their dependent telemetry, but deliberately preserves chat messages, attachments, upload files, backups, and external exports. Delete the owning chat and separately manage retained files/backups when the goal is content erasure rather than telemetry reduction.

Normal operation does not intentionally call an external inference or model API. Dependency installation, image builds, and any administrator-configured proxy are separate network activities.

## Authentication and remote exposure

Bearer authentication is a single shared token, not an identity system. There are no roles, sessions, rate limits, lockouts, token rotation, or audit administration. The application does not terminate TLS.

Keep the default loopback bind whenever possible. If remote access is required:

1. generate and store a strong token outside Git and command history;
2. use an exact origin allowlist;
3. place a reviewed TLS/authenticating reverse proxy in front;
4. restrict network reachability with host controls;
5. configure the proxy not to log WebSocket subprotocol headers, because browser WebSockets carry the tab-scoped bearer credential in a `lad.auth.*` protocol value;
6. review data retention, backups, and upload exposure;
7. test unauthorized HTTP, WebSocket, and cross-origin requests.

Do not put credentials into `.env.example`, Compose build arguments, image layers, URLs committed to source, or support bundles.

## Model and dependency safety

Use SafeTensors checkpoints from a trusted provenance and verify them with a provenance-appropriate full-file digest when integrity matters. The application's default quick fingerprint hashes SafeTensors headers and sizes, not every weight byte. Do not add pickle-based model loading as a fallback. Do not globally enable repository custom code. If an architecture eventually requires local bundled code, it must be audited, fingerprint-constrained, documented, and run only in the isolated worker.

Pin and review dependency changes, particularly PyTorch, Transformers, SentenceTransformers, media decoders, Uvicorn, and browser dependencies. CPU and CUDA wheels come from different indexes; verify the installed build before running a model. Container base images and package indexes are supply-chain inputs even though runtime inference is offline.

## Upload safety

The current store does not extract archives or fetch URLs. PDF extraction is deliberately unavailable. Image files are decoded and verified through Pillow; audio/video signatures are gated by model capability, and a downstream adapter may still reject a media path it does not implement.

Upload limits reduce resource abuse but do not replace decoder hardening. Run untrusted-media workflows on a host with appropriate OS isolation, keep dependencies patched, and avoid external exposure. A MIME type claimed by a client is not trusted.

## Operational response

If compromise or accidental exposure is suspected:

1. stop the application without deleting volumes;
2. revoke/replace any configured access token at its secret source;
3. preserve relevant logs and a copy of application state with access restricted;
4. record the exact revision, model fingerprints, and dependency versions;
5. inspect configured model roots and uploads without executing them;
6. restore only from a known-good integrity-checked backup after the cause is understood.

Never publish a database or support artifact without checking it for prompts, model output, private paths, and tokens.
