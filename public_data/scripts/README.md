# VapeGuard Data Scripts

The executable pipeline commands are packaged under `src/leukerbad/data/`:

- `uv run vapeguard-inspect`
- `uv run vapeguard-download --source-id <id>`
- `uv run vapeguard-inventory`

The `dataset_sources.json` registry is the single source of truth for configured source IDs,
URLs, raw destinations, and known metadata. Raw files must not be edited after download.
