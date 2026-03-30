# Databricks notebook source
# MAGIC %md
# MAGIC ## Create or Update Genie Space
# MAGIC
# MAGIC Creates a Genie AI/BI space (or updates an existing one) with the three geospatial
# MAGIC tables and full ST function instructions.
# MAGIC
# MAGIC - Leave `genie_space_id` empty to **create** a new space.
# MAGIC - Provide an existing `genie_space_id` to **update** title/description/tables/instructions.

# COMMAND ----------

dbutils.widgets.text("catalog",         "cmegdemos_catalog")
dbutils.widgets.text("schema",          "geospatial_analytics")
dbutils.widgets.text("warehouse_id",    "9cd919d96b11bf1c")
dbutils.widgets.text("space_title",     "Geospatial — Analysis")
dbutils.widgets.text("space_description",
    "Live spatial queries on T-Mobile cell signal readings at Chandler Fashion Center "
    "using Databricks ST_* functions. Analyze RSRP signal quality inside vs outside "
    "building polygons with ST_Contains, ST_Point, ST_GeomFromWKT.")
dbutils.widgets.text("parent_path",     "/Shared")
# Leave blank to create a new space; set to an existing space_id to update it
dbutils.widgets.text("genie_space_id",  "")

CATALOG         = dbutils.widgets.get("catalog")
SCHEMA          = dbutils.widgets.get("schema")
WAREHOUSE_ID    = dbutils.widgets.get("warehouse_id")
SPACE_TITLE     = dbutils.widgets.get("space_title")
SPACE_DESC      = dbutils.widgets.get("space_description")
PARENT_PATH     = dbutils.widgets.get("parent_path")
EXISTING_ID     = dbutils.widgets.get("genie_space_id").strip()

print(f"Catalog/Schema : {CATALOG}.{SCHEMA}")
print(f"Warehouse      : {WAREHOUSE_ID}")
print(f"Space title    : {SPACE_TITLE}")
print(f"Parent path    : {PARENT_PATH}")
print(f"Existing ID    : {EXISTING_ID or '(none — will create new)'}")

# COMMAND ----------

# MAGIC %md ### 1. Build the serialized_space payload

# COMMAND ----------

import json

# ── Instructions ─────────────────────────────────────────────────────────────
instructions_text = (
    f"You are a Geospatial Cell Signal Intelligence assistant analyzing T-Mobile (TMO) "
    f"signal quality at Chandler Fashion Center, Chandler AZ. "
    f"Three tables are available in {CATALOG}.{SCHEMA}:\n\n"
    f"(1) intersected_signal_points — Echolocate readings that fall INSIDE mall building polygons "
    f"(ST_Contains join already applied). Use for all inside-mall analysis. "
    f"Short aliases: rsrp, wifi_rssi, network_type, connectivity_type, cell_type, "
    f"building_name, building_type, building_osm_id, EVENTDATE.\n\n"
    f"(2) signal_points — Raw Echolocate readings within 0.5 km of the mall. "
    f"RSRP column: ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP. "
    f"WiFi RSSI column: ED_ENVIRONMENT_NET_CONNECTEDWIFISTATUS_RSSILEVEL. "
    f"Network type: ED_ENVIRONMENT_TELEPHONY_NETWORKTYPE. "
    f"Use ST_Point(LONGITUDE, LATITUDE, 4326) to build a geometry.\n\n"
    f"(3) building_polygons — Chandler Fashion Center building footprints (OpenStreetMap). "
    f"wkt column holds WKT polygon geometry in WGS-84 (EPSG:4326). "
    f"Use ST_GeomFromWKT(wkt, 4326) for spatial joins.\n\n"
    f"SIGNAL QUALITY THRESHOLDS (3GPP LTE RSRP, dBm): "
    f"Excellent >= -80 | Good -90 to -81 | Fair -100 to -91 | Poor < -100.\n\n"
    f"LIVE ST JOIN — inside mall: "
    f"SELECT /*+ BROADCAST(poly) */ s.*, poly.name "
    f"FROM {CATALOG}.{SCHEMA}.signal_points s "
    f"JOIN {CATALOG}.{SCHEMA}.building_polygons poly "
    f"ON ST_Contains(ST_GeomFromWKT(poly.wkt, 4326), ST_Point(s.LONGITUDE, s.LATITUDE, 4326)).\n\n"
    f"OUTSIDE MALL: "
    f"SELECT * FROM {CATALOG}.{SCHEMA}.signal_points s "
    f"WHERE NOT EXISTS ("
    f"SELECT 1 FROM {CATALOG}.{SCHEMA}.building_polygons p "
    f"WHERE ST_Contains(ST_GeomFromWKT(p.wkt, 4326), ST_Point(s.LONGITUDE, s.LATITUDE, 4326))).\n\n"
    f"Use REGEXP_REPLACE(network_type, 'NETWORK_TYPE_', '') to clean network_type values for display. "
    f"Available ST functions: ST_Contains, ST_GeomFromWKT, ST_Point, ST_Distance, ST_Within, ST_Intersects."
)

# ── Column configs (must be sorted alphabetically per table) ──────────────────
def col_cfg(names, entity_match_cols=()):
    return sorted(
        [
            {"column_name": c, "enable_format_assistance": True,
             **({"enable_entity_matching": True} if c in entity_match_cols else {})}
            for c in names
        ],
        key=lambda x: x["column_name"],
    )

intersected_cols = col_cfg(
    ["ED_TIMESTAMP", "EVENTDATE", "LATITUDE", "LONGITUDE",
     "building_class", "building_name", "building_osm_id", "building_type",
     "cell_type", "connectivity_type", "network_name", "network_type",
     "rsrp", "service_state", "wifi_rssi"],
    entity_match_cols={"building_name", "building_type", "cell_type",
                       "connectivity_type", "network_name", "network_type", "service_state"},
)

signal_points_cols = col_cfg(
    ["ED_ENVIRONMENT_NET_CONNECTEDWIFISTATUS_RSSILEVEL",
     "ED_ENVIRONMENT_NET_CONNECTIVITYTYPE",
     "ED_ENVIRONMENT_TELEPHONY_NETWORKTYPE",
     "ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP",
     "ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLTYPE",
     "ED_ENVIRONMENT_TELEPHONY_SERVICESTATE",
     "ED_TIMESTAMP", "EVENTDATE", "LATITUDE", "LONGITUDE"],
    entity_match_cols={"ED_ENVIRONMENT_TELEPHONY_NETWORKTYPE",
                       "ED_ENVIRONMENT_NET_CONNECTIVITYTYPE",
                       "ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLTYPE",
                       "ED_ENVIRONMENT_TELEPHONY_SERVICESTATE"},
)

building_polygon_cols = col_cfg(["fclass", "name", "osm_id", "type", "wkt"],
                                 entity_match_cols={"name", "type"})

serialized_space = {
    "version": 2,
    "data_sources": {
        "tables": [
            # Tables are alphabetically ordered by identifier
            {
                "identifier": f"{CATALOG}.{SCHEMA}.building_polygons",
                "column_configs": building_polygon_cols,
            },
            {
                "identifier": f"{CATALOG}.{SCHEMA}.intersected_signal_points",
                "column_configs": intersected_cols,
            },
            {
                "identifier": f"{CATALOG}.{SCHEMA}.signal_points",
                "column_configs": signal_points_cols,
            },
        ]
    },
    "instructions": {
        # Note: sql_instructions are NOT supported by the API — add curated SQL
        # queries manually in the Genie UI after creation.
        "text_instructions": [
            {"id": "1", "content": [instructions_text]}
        ]
    },
}

print("serialized_space built.")
print(f"  Tables : {[t['identifier'].split('.')[-1] for t in serialized_space['data_sources']['tables']]}")

# COMMAND ----------

# MAGIC %md ### 2. Create or update the Genie space via REST API

# COMMAND ----------

import urllib.request, urllib.error

# Get a fresh OAuth token via the Databricks SDK
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
host  = w.config.host.rstrip("/")
token = w.config.authenticate()["Authorization"].split(" ", 1)[1]

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def api_call(method, path, body=None):
    url  = f"{host}{path}"
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        raise RuntimeError(f"HTTP {e.code} — {err.get('message', err)}") from None


common_payload = {
    "title":            SPACE_TITLE,
    "description":      SPACE_DESC,
    "warehouse_id":     WAREHOUSE_ID,
    "serialized_space": json.dumps(serialized_space),
}

if EXISTING_ID:
    # ── Update existing space ─────────────────────────────────────────────────
    print(f"Updating existing space {EXISTING_ID} ...")
    result = api_call("PATCH", f"/api/2.0/genie/spaces/{EXISTING_ID}", common_payload)
    space_id = result["space_id"]
    print("Space updated.")
else:
    # ── Create new space ──────────────────────────────────────────────────────
    print("Creating new Genie space ...")
    create_payload = {**common_payload, "parent_path": PARENT_PATH}
    result = api_call("POST", "/api/2.0/genie/spaces", create_payload)
    space_id = result["space_id"]
    print("Space created.")

space_url = f"{host}/genie/rooms/{space_id}"
print(f"\nSpace ID  : {space_id}")
print(f"Space URL : {space_url}")

# Surface the space ID as a job output value so downstream tasks can reference it
dbutils.jobs.taskValues.set(key="genie_space_id", value=space_id)
