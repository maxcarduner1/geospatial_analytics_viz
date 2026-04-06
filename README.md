# Geospatial analytics & visualization (Chandler Fashion Center)

Databricks Asset Bundle demo that loads T-Mobile-style cell signal readings and building footprints around **Chandler Fashion Center**, runs **geospatial SQL** (`ST_*`) to intersect points with polygons, then publishes **Unity Catalog tables**, a **Dash Databricks App**, and optional **Genie** space instructions.

## What runs in the job

The serverless job `geospatial_analysis_job` runs four tasks in order:

| Task | Notebook | Purpose |
|------|----------|---------|
| `setup_tables` | `notebooks/01_setup_tables.py` | Read raw CSV + shapefile from a UC Volume; create `signal_points` and `building_polygons` Delta tables. |
| `run_analysis` | `notebooks/02_analysis.py` | Spatial join / intersection; write `intersected_signal_points` and summaries. |
| `build_app` | `notebooks/03_build_app.py` | Create UC SQL helper functions, generate Dash app source under a workspace path, create/deploy the Databricks App. |
| `create_genie_space` | `notebooks/04_create_genie_space.py` | Create or update a Genie space wired to the geospatial tables (runs in parallel with app build after analysis). |

Python dependencies (for example `geopandas`) are declared in `databricks.yml` under the serverless environment — not via `%pip` in notebooks.

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) with a profile that can deploy to your workspace.
- Unity Catalog **catalog** and **schema**, and a **Volume** at  
  `/Volumes/<catalog>/<schema>/raw_data` containing the expected CSV and shapefile assets (see `01_setup_tables.py` for filenames).
- A SQL **warehouse** ID for notebooks that issue SQL against the warehouse.
- For `build_app`: a workspace directory for generated app files (`app_dir`), and permissions to create/update Apps and UC objects.

## Configure the bundle

Edit variables in `databricks.yml` (or override at deploy time) for your environment:

| Variable | Typical use |
|----------|-------------|
| `catalog`, `schema` | UC destination for tables and functions. |
| `warehouse_id` | Serverless SQL warehouse for warehouse-backed queries. |
| `app_name`, `app_dir` | Databricks App name and workspace folder for generated source. |
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
databricks.yml          # Bundle definition, variables, serverless job
notebooks/
  01_setup_tables.py    # Volume → Delta tables
  02_analysis.py        # ST_* analysis → intersected_signal_points
  03_build_app.py       # UC functions + Dash app generation & deploy
  04_create_genie_space.py
```

## License

Use and adapt for internal demos and customer workshops as appropriate for your organization.
