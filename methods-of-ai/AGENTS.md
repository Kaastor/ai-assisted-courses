# AGENTS.md

> **Purpose:** This file guides **Codex** when working with code in this repository. It summarizes how to set up, test, and contribute, plus the standards and design principles to follow.

---

## Setup

- **Language:** Python **3.11**
- **Env:** Use a local virtualenv at `.venv`
- **Dependency & project config:** `pyproject.toml` (Poetry)

### Commands

- Install dependencies:  
  `poetry install`
- Activate virtualenv (optional):  
  `poetry shell`

> **Tip:** Prefer running commands via `poetry run …` to ensure tools use the project’s virtualenv.

---

## Test & Verify

- **All tests:**  
  `poetry run python -m pytest`
- **Single test example:**  
  `poetry run python -m pytest app/tests/test_app.py::test_hello_name -v`

**If configured** (run when available in `pyproject.toml`):
- **Type check:** `poetry run mypy .`
- **Lint:** `poetry run ruff check .`  _(or `flake8 .` if used)_
- **Format:** `poetry run black . --check`  _(auto-format with `--diff` or without `--check`)_

**Before proposing changes:**
1. Run tests and fix failures.
2. Ensure type checks, linters, and formatters are clean.
3. Add/adjust tests for your changes.

---

## Project Layout

- **Source:** Typically under `app/`
- **Tests:** `app/tests/`
- **Config:** `pyproject.toml`

> **Discover structure quickly:**  
> `poetry run python -c "import pathlib; print('\\n'.join(str(p) for p in pathlib.Path('.').glob('**/*') if p.is_dir() and not any(s in p.parts for s in ('.git','.venv','__pycache__'))))"`  
> or  
> `tree -a -I '.git|.venv|__pycache__' -L 2`

---

## Code Style

- **PEP 8** for general style.
- **Type hints:** Use modern/natural types (e.g., `list[str]`, `dict[str, Any]`).
- **Docstrings:** Google-style for modules, classes, and functions.
- **Naming:** `snake_case` for functions/variables; `PascalCase` for classes.
- **Function size:** Aim for < 30 lines and single responsibility.
- **Project layout:** Standard Python package layout.

---

## Python Best Practices

- Prefer `pathlib.Path` over `os.path`.
- Use the `logging` module instead of `print` for diagnostics.
- Raise specific exceptions with context; log appropriately.
- Use comprehensions for clear, concise transformations.
- **Never** use mutable default arguments.
- Use `@dataclass` for data containers.
- Load configuration via environment variables (e.g., `python-dotenv`).

---

## Object-Oriented Design (Python)

- **SOLID** principles:
  - **SRP:** One reason to change per class.
  - **OCP:** Extend via composition/inheritance; don’t modify core behavior.
  - **LSP:** Subclasses must be substitutable.
  - **ISP:** Prefer smaller, focused interfaces (use `typing.Protocol` when helpful).
  - **DIP:** Depend on abstractions; inject dependencies.
- Favor **composition** over inheritance; use mixins sparingly.
- Encapsulate with private attrs; expose via `@property` when appropriate.
- Define interfaces with `abc.ABC`/`@abstractmethod`; pair with type hints.
- Embrace duck typing; add `Protocol` for static checks when useful.
- Keep classes small (< ~200 lines); refactor early.
- Start with functions; introduce classes when state/polymorphism is needed.
- Test via public interfaces; mock dependencies with `unittest.mock`.

---

## Microservices (If Applicable)

- **Boundaries:** DDD-aligned; each service owns its data and logic.
- **Communication:** HTTP/REST for simple sync; messaging (e.g., RabbitMQ/Celery) for async events.
- **Resilience:** Circuit breakers, retries, and timeouts (e.g., `tenacity`).
- **Config/Discovery:** Env vars; service discovery as needed (Consul/etcd).
- **Containerization:** Docker with multi-stage builds; include `/health` endpoint (e.g., FastAPI).
- **Orchestration:** Target Kubernetes; readiness/liveness probes.
- **Testing:** Contract tests (e.g., pact); integration with testcontainers.

---

## Development Patterns & Practices

- Prefer the simplest solution that meets requirements.
- **DRY:** Reuse; avoid duplication.
- Keep changes **focused**; preserve existing patterns when fixing bugs.
- Keep files under ~300 lines where feasible; refactor when exceeding.
- Write comprehensive unit/integration tests; use `pytest` parametrize for table-driven cases.
- Mock external deps; avoid testing implementation details.
- Build modular, reusable components.
- Logging with appropriate levels (`DEBUG`, `INFO`, `ERROR`).
- Robust error handling for production reliability.
- Validate inputs; protect data (security first).
- Optimize only critical paths; measure before/after.
- Plan for scalability (e.g., sharding, decomposing to services).
- Prototype for complex designs before full build.

---

## Architectural Patterns

- **Modular/Layered:** Separate presentation, application, domain, infrastructure; use dependency injection for testability.
- **Patterns:** MVC for APIs; Factories/Singleton for shared resources; Observer/Command for events.
- **API-first:** Define APIs up front; document with OpenAPI/Swagger.

---

## Scalability & Performance

- **Horizontally scale** stateless components; use load balancers and containers.
- **Caching:** Use Redis (or similar) for hot paths.
- **Async & queues:** Use `asyncio`/`httpx` or Celery/RQ for I/O-bound workloads.
- **Observability:** Metrics (Prometheus/Grafana), structured logs, traces.
- **ML apps:** Use MLflow/Kubeflow for pipelines; handle large data with Dask/Spark.

---

## ML-Specific (If Repo Involves AI/ML)

- Build end-to-end pipelines (ingest → preprocess → train → deploy).
- Version data/models; detect drift; automate retraining triggers.
- Ethics: bias checks, explainability (e.g., SHAP), privacy (e.g., federated learning).
- For LLM apps, consider **RAG**; serve via FastAPI or gRPC for low latency.

---

## Workflow for Codex

1. **Understand the task:** Read this file and relevant code; locate impacted modules/tests.
2. **Propose a minimal plan:** Aim for small, reviewable diffs.
3. **Implement:**
   - Follow style and design rules above.
   - Keep functions focused; prefer composition.
   - Update/add tests for new behavior.
4. **Verify locally:**
   - `poetry run python -m pytest`
   - (if configured) `poetry run mypy .`, `poetry run ruff check .`, `poetry run black . --check`
5. **Final checks:**
   - Ensure logging is appropriate; remove debug-only statements.
   - Confirm error handling and input validation.
   - Keep file sizes reasonable; refactor if growing too large.
6. **Commit/PR guidance:**
   - Title format: `[module] short, descriptive title`
   - Include a brief rationale and test notes in the description.

---

## Dependencies

See `pyproject.toml` for production vs. dev dependencies. Keep them separated and minimal.

---

## Notes for New Contributors & Agents

- Respect existing patterns; when in doubt, mirror adjacent code.
- Prefer incremental, tested changes over broad refactors.
- When touching public APIs, update docstrings and tests accordingly.
