## Repo-Local Job Specs

Repo-local job specs live in the project tree at `docs/Job<id>/job_<id>.json`.

The layout is standardized so watchdog and packet builders can treat every repo the same:

- Document path: `<repo>/docs/Job<id>/job_<id>.json`
- Shared schema: `common/docs/job_specs/job_spec_schema.json`
- Shared validator: `python3 common/scripts/validate_job_spec.py <repo>/docs/Job<id>/job_<id>.json`
- Scaffold helper: `python3 common/scripts/scaffold_job_spec.py --job <id> --repo <repo-root> --output <path>`

Notes:

- Repo-local `job_schema.json` files are no longer required for packet validation.
- Repo-local `validate_job.py` files are optional compatibility wrappers around the shared validator.
- Prefer repo-relative `primary_files` entries. Legacy `<repo-name>/...` prefixes are still accepted by the packet builder.
