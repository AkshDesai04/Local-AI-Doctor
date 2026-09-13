# Contributing

Local AI Doctor is developed on the `dev` branch. Changes should preserve its local-first, evidence-based design: do not claim a model capability that discovery and the executable adapter can both demonstrate.

## Before contributing

No license has been selected for this repository. Copyright is reserved by default, so public reuse and redistribution are not currently authorized by an open-source license. Coordinate with the owner before making a contribution that depends on particular licensing terms. Do not add or change a license as part of an unrelated pull request.

Never commit model weights, user prompts or outputs, uploads, databases, local configuration, credentials, private paths, virtual environments, caches, build output, raw telemetry, or unreviewed benchmark artifacts.

Use the repository's existing Git identity and contribution workflow. Do not add generated-by or automated co-author trailers.

## Development environment

Python 3.12 is the validated target. From PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock -r requirements-ml.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --editable .

Push-Location frontend
npm ci
Pop-Location
```

The normal test suite uses local tiny fixtures. You do not need the supplied multi-gigabyte checkpoints, CUDA, WSL, or Docker for routine contributions.

Copy `config/local.example.yaml` to the ignored `config/local.yaml` only when a local run needs custom paths. Keep the default listener on loopback.

## Quality checks

Run the backend checks from the repository root:

```powershell
.\.venv\Scripts\ruff.exe format backend tests
.\.venv\Scripts\ruff.exe check backend tests
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\pytest.exe -m "not real_model and not gpu and not performance"
```

Run frontend checks from `frontend/`:

```powershell
npm run lint
npm run test:run
npm run build
npx playwright test
```

Run targeted tests while iterating, but run the complete relevant set before requesting review. A change to API normalization or WebSocket behavior normally needs backend integration coverage and frontend tests. A layout change needs desktop and narrow-width browser inspection.

The checked-in [CPU CI workflow](.github/workflows/ci.yml) runs the backend checks with Python 3.12 and the frontend checks with Node.js 22, excluding tests marked `real_model` or `gpu`. Keep routine fixture coverage independent of private checkpoints and accelerators; do not treat CI as a source of benchmark results.

Docker must be invoked through the selected WSL2 distribution, never through Windows Docker commands. Follow [docs/deployment.md](docs/deployment.md) for Compose validation and image checks.

## Desktop release versions

`VERSION` is the release-version source of truth. Keep it synchronized with the Python, frontend, and desktop manifests through the checked-in helper:

```powershell
node scripts/release-version.mjs check
node scripts/release-version.mjs patch
```

Use a patch bump for an ordinary completed feature or fix. Use `minor` or `major` in place of `patch` only when the project owner explicitly requests that release level. Commit the version bump with the feature; do not edit one manifest independently.

After backend, frontend, and desktop validation succeeds, every `dev` push produces a GitHub prerelease tagged `vX.Y.Z-beta.<workflow-run>`. Rerunning that workflow repairs or confirms the same release instead of creating another one. A `main` push produces the immutable stable `vX.Y.Z` release, so merging the tested `dev` revision promotes the same base version. The GitHub build deliberately bundles CPU PyTorch because a CUDA runtime exceeds GitHub's 2 GiB per-asset limit; users can still run the repository's CUDA Docker profile. The workflow uploads exactly one custom release asset: `Local-AI-Doctor-<version>.exe`. GitHub additionally displays its unavoidable auto-generated source-code links.

## Design rules

- Keep configuration in `AppSettings`; do not read new `LAD_` variables from business logic.
- Keep model/device objects inside the spawned worker and send only bounded serializable messages.
- Treat model metadata, tokenizer templates, filenames, uploads, and output as untrusted.
- Preserve local-only loading and `trust_remote_code=false` unless a narrowly fingerprinted custom-code path has been audited and approved.
- Keep discovery evidence separate from runtime support. Every non-full capability needs a reason.
- Never classify a dense gated MLP as MoE routing.
- Keep raw and post-sampler distributions distinct. Do not normalize displayed Top-K values and label them exact.
- Preserve token ID/piece/byte/display boundaries; tokenizer pieces do not necessarily map one-to-one to visible text.
- Keep high-volume persistence off the model worker's decode path and preserve event ordering.
- Use structured, redacted errors at transport boundaries.
- Add migrations rather than modifying an applied schema in place after a release.
- Avoid architecture/model-name conditionals when stable metadata or a narrow adapter probe can express the rule.

## Adding models or metrics

Read [docs/adapters.md](docs/adapters.md) before changing discovery or execution. New support needs negative tests, a complete capability matrix, CPU fixture coverage, and opt-in real-model evidence where available.

Read [docs/metrics.md](docs/metrics.md) before adding a metric. Define the distribution, units, clock, inclusion/exclusion rules, and whether the value is exact, sampled, estimated, or unavailable. Update API types and UI labels in the same change.

Performance claims require the protocol in [docs/benchmarking.md](docs/benchmarking.md), including raw run IDs and environment identity. Keep large or private results out of Git.

## Tests that use local models

The read-only discovery test is opt-in:

```powershell
$env:LAD_REAL_MODEL_ROOT = 'X:\path\to\models'
.\.venv\Scripts\pytest.exe -m real_model
Remove-Item Env:LAD_REAL_MODEL_ROOT
```

Never download a large substitute automatically or mutate the configured root. Mark GPU tests so CPU CI excludes them, and mark performance tests so benchmark-specific invocations can select them deliberately. State clearly whether evidence came from a fixture, metadata/header inspection, or a real forward pass.

## Pull request checklist

- The change has one coherent purpose and no unrelated formatting churn.
- Boundary objects, errors, and public interfaces are typed.
- Relevant Ruff, mypy, pytest, ESLint, Vitest, build, and browser checks pass.
- Failure, cancellation, restart, and security paths were considered.
- Documentation, API examples, capabilities, and changelog are updated where needed.
- `git diff --check` is clean.
- The diff contains no secret, personal path, model weight, database, upload, or generated output.
- No unsupported feature is described as implemented or tested.

Report suspected vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in a public issue.
