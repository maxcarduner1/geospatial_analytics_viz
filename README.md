# Geospatial analytics & visualization (Chandler Fashion Center)

Databricks Asset Bundle demo that loads T-Mobile-style cell signal readings and building footprints around **Chandler Fashion Center**, runs **geospatial SQL** (`ST_*`) to intersect points with polygons, optionally uses **H3 spatial indexing** for a scalable alternative join path, then publishes **Unity Catalog tables**, a **Dash Databricks App** (plus optional agent app artifacts), and optional **Genie** space instructions.

## What runs in the job

The serverless job `geospatial_analysis_job` runs **five** tasks with this dependency graph:

1. **`setup_tables`** runs first.
2. **`run_h3_analysis`** and **`run_analysis`** both depend only on `setup_tables`, so they may run **in parallel** after load.
3. **`build_app`** and **`create_genie_space`** depend on **`run_analysis`** (classic ST intersection path). They may run in parallel with each other once analysis finishes.

| Task | Notebook | Purpose |
|------|----------|---------|
| `setup_tables` | `notebooks/01_setup_tables.py` | Read raw CSV + shapefile from a UC Volume; create `signal_points` and `building_polygons` Delta tables. |
| `run_h3_analysis` | `notebooks/02a_h3_analysis.py` | H3-indexed spatial join: tessellate polygons, coarse join on H3 cell, refine with `ST_Contains`; writes tables such as `intersected_signal_points_h3` (parameter `h3_resolution`, default `9`). |
| `run_analysis` | `notebooks/02_analysis.py` | Spatial join / intersection; write `intersected_signal_points` and summaries (warehouse-backed SQL). |
| `build_app` | `notebooks/03_build_app.py` | Create UC SQL helper functions, generate Dash app source under workspace paths, create/deploy the primary app and agent app assets. |
| `create_genie_space` | `notebooks/04_create_genie_space.py` | Create or update a Genie space wired to the geospatial tables. |

Python dependencies (for example `geopandas`) are declared in `databricks.yml` under the serverless environment — not via `%pip` in notebooks.

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) with a profile that can deploy to your workspace.
- Unity Catalog **catalog** and **schema**, and a **Volume** at  
  `/Volumes/<catalog>/<schema>/raw_data` containing the expected CSV and shapefile assets (see `01_setup_tables.py` for filenames).
- A SQL **warehouse** ID for notebooks that issue SQL against the warehouse (`run_analysis`, `build_app`, `create_genie_space`).
- For `build_app`: workspace directories for generated app files (`app_dir`, `agent_app_dir`), and permissions to create/update Apps and UC objects.

## Configure the bundle

Edit variables in `databricks.yml` (or override at deploy time) for your environment:

| Variable | Typical use |
|----------|-------------|
| `catalog`, `schema` | UC destination for tables and functions. |
| `warehouse_id` | Serverless SQL warehouse for warehouse-backed queries. |
| `app_name`, `app_dir` | Primary Databricks App name and workspace folder for generated source. |
| `agent_app_name`, `agent_app_dir` | Agent-oriented app name and workspace folder (passed through `03_build_app.py`). |
| `genie_space_title`, `genie_space_description`, `genie_parent_path` | Genie space metadata. |
| `genie_space_id` | Leave empty to **create** a new space; set to an existing space ID to **update** it. |

Replace workspace-specific defaults (host, profile, paths under `/Workspace/Users/...`) if you are not using the original demo workspace.

## Deploy and run

```bash
# From this repository root
databricks bundle deploy --profile <your-profile>

# Run the full pipeline (or start the job from the Jobs UI)
databricks bundle run geospatial_analysis_job --profile <your-profile>
```

For a failed multi-task run, prefer [repairing from the failed task](https://docs.databricks.com/jobs.html) rather than re-running the whole job.

## Repository layout

```
databricks.yml           # Bundle definition, variables, serverless job
notebooks/
  01_setup_tables.py   # Volume → Delta tables
  02_analysis.py       # ST_* analysis → intersected_signal_points
  02a_h3_analysis.py   # H3-indexed join → intersected_signal_points_h3 (+ related)
  03_build_app.py      # UC functions + Dash app generation & deploy
  04_create_genie_space.py
```

## License

Use and adapt for internal demos and customer workshops as appropriate for your organization.
