# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup: Persist Raw Signal Points & Building Polygons
# MAGIC
# MAGIC Creates managed Delta tables in Unity Catalog from the raw Volume files:
# MAGIC - `signal_points`      — raw Echolocate CSV readings (lat/lon + cell metrics)
# MAGIC - `building_polygons`  — building footprint shapefile stored as WKT geometry column
# MAGIC
# MAGIC These tables enable Genie to run live ST_* spatial queries without re-reading files.

# COMMAND ----------

# Parameters (injected by DAB; fall back to notebook defaults for interactive runs)
dbutils.widgets.text("catalog", "cmegdemos_catalog")
dbutils.widgets.text("schema",  "geospatial_analytics")

CATALOG     = dbutils.widgets.get("catalog")
SCHEMA      = dbutils.widgets.get("schema")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"
CSV_FILE    = "Echolocate points around Chandler Fashion Center Area (0.5km).csv"
SHP_FILE    = "Chandler Fashion Center Area (0.5km).shp"

print(f"Catalog : {CATALOG}")
print(f"Schema  : {SCHEMA}")
print(f"Volume  : {VOLUME_PATH}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

# COMMAND ----------

# MAGIC %md ### 1. Signal Points Table

# COMMAND ----------

csv_path = f"{VOLUME_PATH}/{CSV_FILE}"

points_df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .option("escape", '"')
    .load(csv_path)
)

print(f"Raw CSV rows: {points_df.count():,}")
points_df.printSchema()

# COMMAND ----------

# Persist as a managed Delta table — overwrite for idempotent re-runs
(
    points_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.signal_points")
)

spark.sql(f"""
    COMMENT ON TABLE `{CATALOG}`.`{SCHEMA}`.signal_points IS
    'Raw Echolocate cell-signal readings (lat/lon + RSRP/RSSI + network metadata) around Chandler Fashion Center (0.5 km radius). Use ST_Point(LONGITUDE, LATITUDE, 4326) to build a geometry.'
""")

print(f"Saved signal_points: {spark.table(f'`{CATALOG}`.`{SCHEMA}`.signal_points').count():,} rows")

# COMMAND ----------

# MAGIC %md ### 2. Building Polygons Table

# COMMAND ----------

# %pip install geopandas --quiet   # already installed on LTS clusters; uncomment if needed

import geopandas as gpd

shp_path = f"{VOLUME_PATH}/{SHP_FILE}"
gdf = gpd.read_file(shp_path)

print(f"Shapefile records : {len(gdf)}")
print(f"CRS               : {gdf.crs}")
print(gdf.head())

# Flatten to WKT — ST_GeomFromWKT is available in Databricks SQL
gdf["wkt"] = gdf.geometry.apply(lambda g: g.wkt)
cols = [c for c in gdf.columns if c != "geometry"]

polygons_sdf = spark.createDataFrame(gdf[cols])

(
    polygons_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.building_polygons")
)

spark.sql(f"""
    COMMENT ON TABLE `{CATALOG}`.`{SCHEMA}`.building_polygons IS
    'Building footprint polygons for Chandler Fashion Center (from OpenStreetMap shapefile). The wkt column holds POLYGON geometry in WGS-84 (EPSG:4326). Use ST_GeomFromWKT(wkt, 4326) in spatial joins.'
""")

print(f"Saved building_polygons: {spark.table(f'`{CATALOG}`.`{SCHEMA}`.building_polygons').count():,} rows")

# COMMAND ----------

# MAGIC %md ### Verify: Quick ST join sanity-check
# MAGIC
# MAGIC Confirms ST functions resolve and the tables are queryable together.

# COMMAND ----------

spark.sql(f"""
SELECT
    p.name           AS building_name,
    COUNT(*)         AS point_count
FROM `{CATALOG}`.`{SCHEMA}`.signal_points   s
JOIN `{CATALOG}`.`{SCHEMA}`.building_polygons p
  ON ST_Contains(
       ST_GeomFromWKT(p.wkt, 4326),
       ST_Point(s.LONGITUDE, s.LATITUDE, 4326)
     )
GROUP BY p.name
ORDER BY point_count DESC
""").show()
