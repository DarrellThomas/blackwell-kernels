## Watchdog Job Packets

`job_packet.json` is the common structured handoff that watchdog generates for each worker slot.

It is meant to scale the `docs/Job22/job_22.json` pattern without forcing every repo to store the same data twice:

- The packet is generated from the factory DB, open messages, experiment summary, repo/file resolution, and worker protocol.
- If a repo-local structured spec exists at `docs/Job<id>/job_<id>.json`, the packet attaches it under `repo_local_spec` and records its validation status.
- Packets live under `/data/src/bwk/data/watchdog-worktrees/<worker>/job_packet.json`, outside the git worktree, so watchdog guidance does not dirty `git status`.

Worker flow:

1. Watchdog creates or reuses a dedicated worktree for the worker.
2. Watchdog generates a validated packet for the active assignment.
3. The worker reads the packet first, then refreshes with the DB commands embedded in `protocol.refresh_commands`.
4. Handoff is reported through the packet's `done`, `check_my_work`, or `problem` protocol entries.

Validation:

- Schema: `common/docs/job_packets/job_packet_schema.json`
- Validator: `python3 common/scripts/validate_job_packet.py <packet.json>`
- Builder: `python3 common/scripts/build_job_packet.py --worker <cxN>`
