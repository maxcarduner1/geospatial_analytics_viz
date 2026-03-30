# Databricks notebook source
# MAGIC %md
# MAGIC ## Spatial Signal Strength Analysis — Chandler Fashion Center
# MAGIC
# MAGIC Runs a full ST intersection between signal readings and building polygons,
# MAGIC persists results to `intersected_signal_points`, and generates summary statistics.
# MAGIC
# MAGIC **Depends on:** `01_setup_tables` (signal_points + building_polygons tables must exist)

# COMMAND ----------

dbutils.widgets.text("catalog",      "cmegdemos_catalog")
dbutils.widgets.text("schema",       "geospatial_analytics")
dbutils.widgets.text("warehouse_id", "9cd919d96b11bf1c")

CATALOG      = dbutils.widgets.get("catalog")
SCHEMA       = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")

TARGET_TABLE = f"`{CATALOG}`.`{SCHEMA}`.intersected_signal_points"

print(f"Target table : {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md ### 1. ST Intersection — signal points inside building polygons

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {TARGET_TABLE} AS
SELECT /*+ BROADCAST(poly) */
    s.LONGITUDE,
    s.LATITUDE,
    s.EVENTDATE,
    s.ED_TIMESTAMP,
    s.ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP   AS rsrp,
    s.ED_ENVIRONMENT_NET_CONNECTEDWIFISTATUS_RSSILEVEL               AS wifi_rssi,
    s.ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLIDENTITY_NETWORKNAME  AS network_name,
    s.ED_ENVIRONMENT_TELEPHONY_NETWORKTYPE                           AS network_type,
    s.ED_ENVIRONMENT_NET_CONNECTIVITYTYPE                            AS connectivity_type,
    s.ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLTYPE                  AS cell_type,
    s.ED_ENVIRONMENT_TELEPHONY_SERVICESTATE                          AS service_state,
    poly.osm_id   AS building_osm_id,
    poly.fclass   AS building_class,
    poly.name     AS building_name,
    poly.type     AS building_type
FROM `{CATALOG}`.`{SCHEMA}`.signal_points   s
JOIN `{CATALOG}`.`{SCHEMA}`.building_polygons poly
  ON ST_Contains(
       ST_GeomFromWKT(poly.wkt, 4326),
       ST_Point(s.LONGITUDE, s.LATITUDE, 4326)
     )
""")

count = spark.sql(f"SELECT COUNT(*) AS n FROM {TARGET_TABLE}").collect()[0]["n"]
print(f"Intersected rows: {count:,}")

spark.sql(f"""
    COMMENT ON TABLE {TARGET_TABLE} IS
    'Signal readings that fall inside a Chandler Fashion Center building polygon (ST_Contains join). Includes RSRP, WiFi RSSI, network metadata, and building attributes.'
""")

# COMMAND ----------

# MAGIC %md ### 2. Signal Quality by Building

# COMMAND ----------

spark.sql(f"""
SELECT
    building_name,
    building_type,
    COUNT(*)                                                              AS point_count,
    ROUND(AVG(rsrp), 1)                                                  AS avg_rsrp,
    MIN(rsrp)                                                             AS min_rsrp,
    MAX(rsrp)                                                             AS max_rsrp,
    ROUND(AVG(wifi_rssi), 1)                                             AS avg_wifi_rssi,
    ROUND(100.0 * SUM(CASE WHEN rsrp >= -80  THEN 1 ELSE 0 END) / COUNT(*), 1)              AS pct_excellent,
    ROUND(100.0 * SUM(CASE WHEN rsrp BETWEEN -90 AND -81 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_good,
    ROUND(100.0 * SUM(CASE WHEN rsrp BETWEEN -100 AND -91 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_fair,
    ROUND(100.0 * SUM(CASE WHEN rsrp < -100 THEN 1 ELSE 0 END) / COUNT(*), 1)              AS pct_poor
FROM {TARGET_TABLE}
WHERE rsrp IS NOT NULL
GROUP BY building_name, building_type
ORDER BY point_count DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ### 3. Inside vs Outside Polygon — full dataset comparison

# COMMAND ----------

spark.sql(f"""
WITH inside AS (
    SELECT 'Inside Mall' AS location, rsrp, wifi_rssi
    FROM   {TARGET_TABLE}
    WHERE  rsrp IS NOT NULL
),
outside AS (
    SELECT 'Outside Mall' AS location,
           s.ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP AS rsrp,
           s.ED_ENVIRONMENT_NET_CONNECTEDWIFISTATUS_RSSILEVEL              AS wifi_rssi
    FROM   `{CATALOG}`.`{SCHEMA}`.signal_points  s
    WHERE  s.ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP IS NOT NULL
      AND NOT EXISTS (
            SELECT 1
            FROM `{CATALOG}`.`{SCHEMA}`.building_polygons poly
            WHERE ST_Contains(
                    ST_GeomFromWKT(poly.wkt, 4326),
                    ST_Point(s.LONGITUDE, s.LATITUDE, 4326)
                  )
          )
)
SELECT
    location,
    COUNT(*)                          AS readings,
    ROUND(AVG(rsrp), 1)              AS avg_rsrp,
    ROUND(STDDEV(rsrp), 1)           AS stddev_rsrp,
    ROUND(AVG(wifi_rssi), 1)         AS avg_wifi_rssi,
    ROUND(100.0 * SUM(CASE WHEN rsrp >= -80  THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_excellent,
    ROUND(100.0 * SUM(CASE WHEN rsrp < -100  THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_poor
FROM (SELECT * FROM inside UNION ALL SELECT * FROM outside)
GROUP BY location
ORDER BY location
""").show()

# COMMAND ----------

# MAGIC %md ### 4. Network Type Distribution (inside only)

# COMMAND ----------

spark.sql(f"""
SELECT
    REGEXP_REPLACE(network_type, 'NETWORK_TYPE_', '') AS network,
    connectivity_type,
    cell_type,
    COUNT(*)             AS readings,
    ROUND(AVG(rsrp), 1) AS avg_rsrp
FROM {TARGET_TABLE}
WHERE rsrp IS NOT NULL
GROUP BY network_type, connectivity_type, cell_type
ORDER BY readings DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ### 5. Daily Signal Trend (inside mall)

# COMMAND ----------

spark.sql(f"""
SELECT
    EVENTDATE,
    COUNT(*)             AS readings,
    ROUND(AVG(rsrp), 1) AS avg_rsrp,
    MIN(rsrp)            AS min_rsrp,
    MAX(rsrp)            AS max_rsrp
FROM {TARGET_TABLE}
WHERE rsrp IS NOT NULL AND EVENTDATE IS NOT NULL
GROUP BY EVENTDATE
ORDER BY EVENTDATE
""").show(50, truncate=False)
