# Databricks notebook source
# MAGIC %md
# MAGIC ## Build & Deploy — Signal Quality Dash App
# MAGIC
# MAGIC Generates the Dash app source files, creates the Databricks App (if needed),
# MAGIC grants the app service principal the required permissions, then deploys.
# MAGIC
# MAGIC **Prerequisites:** `intersected_signal_points` table must exist (run `02_analysis` first).

# COMMAND ----------

dbutils.widgets.text("catalog",      "cmegdemos_catalog")
dbutils.widgets.text("schema",       "geospatial_analytics")
dbutils.widgets.text("warehouse_id", "9cd919d96b11bf1c")
dbutils.widgets.text("app_name",     "geospatial-signal-quality")
# Workspace path where app source files will be written
dbutils.widgets.text("app_dir",      "/Workspace/Users/max.carduner@databricks.com/geospatial-signal-quality-app")

CATALOG      = dbutils.widgets.get("catalog")
SCHEMA       = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
APP_NAME     = dbutils.widgets.get("app_name")
APP_DIR      = dbutils.widgets.get("app_dir")
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.intersected_signal_points"

print(f"Target table : {TARGET_TABLE}")
print(f"App name     : {APP_NAME}")
print(f"App dir      : {APP_DIR}")
print(f"Warehouse    : {WAREHOUSE_ID}")

# COMMAND ----------

# MAGIC %md ### 1. Generate app source files

# COMMAND ----------

import os
os.makedirs(APP_DIR, exist_ok=True)

# ── app.py ───────────────────────────────────────────────────────────────────
app_py = f'''import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks import sql

TABLE        = "{TARGET_TABLE}"
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "{WAREHOUSE_ID}")
MAP_CENTER   = {{"lat": 33.3013, "lon": -111.8986}}
DEFAULT_ZOOM = 15

w = WorkspaceClient()


def get_connection():
    hostname = w.config.host.replace("https://", "").rstrip("/")
    return sql.connect(
        server_hostname=hostname,
        http_path=f"/sql/1.0/warehouses/{{WAREHOUSE_ID}}",
        credentials_provider=lambda: w.config.authenticate,
    )


def load_data() -> pd.DataFrame:
    query = f"""
        SELECT LONGITUDE, LATITUDE, EVENTDATE, rsrp, wifi_rssi,
               network_name, network_type, connectivity_type,
               cell_type, service_state, building_osm_id, building_name, building_type
        FROM {{TABLE}}
        WHERE rsrp IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    df["network_type"] = df["network_type"].str.replace("NETWORK_TYPE_", "", regex=False)
    df["signal_quality"] = pd.cut(
        df["rsrp"],
        bins=[float("-inf"), -100, -90, -80, float("inf")],
        labels=["Poor", "Fair", "Good", "Excellent"],
        ordered=True,
    )
    df["building_label"] = df["building_name"].fillna("ID:" + df["building_osm_id"].astype(str))
    return df


print("Loading data from Delta table...")
df = load_data()
print(f"Loaded {{len(df):,}} signal readings.")


def opts(series):
    return [{{"label": v, "value": v}} for v in sorted(series.dropna().unique())]


METRIC_OPTIONS = [
    {{"label": "RSRP Signal Strength", "value": "rsrp"}},
    {{"label": "WiFi RSSI Level",       "value": "wifi_rssi"}},
]
VIEW_OPTIONS = [
    {{"label": "Heatmap",   "value": "heatmap"}},
    {{"label": "Point Map", "value": "scatter"}},
]
COLOR_BY_OPTIONS = [
    {{"label": "Signal Quality",    "value": "signal_quality"}},
    {{"label": "Network Type",      "value": "network_type"}},
    {{"label": "Connectivity Type", "value": "connectivity_type"}},
    {{"label": "Cell Type",         "value": "cell_type"}},
    {{"label": "Building",          "value": "building_label"}},
]
QUALITY_COLORS = {{"Excellent": "#2ecc71", "Good": "#3498db", "Fair": "#f39c12", "Poor": "#e74c3c"}}

app = dash.Dash(__name__, title="Signal Quality")

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            .Select-menu-outer, .Select-menu-outer * {{ color: #111 !important; }}
            .VirtualizedSelectOption {{ color: #111 !important; background-color: #fff !important; }}
            .VirtualizedSelectFocusedOption {{ background-color: #deebff !important; }}
            .Select-option {{ color: #111 !important; }}
            .Select-option.is-focused {{ background-color: #deebff !important; }}
            .Select-value-label {{ color: #111 !important; }}
            .Select--multi .Select-value {{ background-color: #e8e8e8 !important; color: #111 !important; }}
            .DateInput_input {{ color: #111 !important; }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>"""

SIDEBAR = {{"width": "310px", "minWidth": "310px", "padding": "20px",
            "backgroundColor": "#1a1a2e", "color": "#eee",
            "borderRadius": "8px", "overflowY": "auto", "maxHeight": "85vh"}}
LBL = {{"fontWeight": "600", "marginTop": "14px", "marginBottom": "4px", "fontSize": "13px"}}
DD  = {{"marginBottom": "8px"}}

app.layout = html.Div([
    html.Div([
        html.H2("Cell Signal Quality \u2014 Chandler Fashion Center",
                style={{"margin": "0", "color": "#1a1a2e"}}),
        html.P("Interactive spatial heatmap of RSRP signal strength inside building polygons",
               style={{"margin": "4px 0 0", "color": "#666", "fontSize": "14px"}}),
    ], style={{"padding": "16px 24px", "borderBottom": "2px solid #e0e0e0"}}),
    html.Div([
        html.Div([
            html.H4("Controls", style={{"color": "#e94560", "marginTop": "0"}}),
            html.Label("View Mode", style=LBL),
            dcc.RadioItems(id="view-mode", options=VIEW_OPTIONS, value="heatmap",
                           labelStyle={{"display": "block", "marginBottom": "4px"}},
                           style={{"marginBottom": "8px"}}),
            html.Label("Heatmap Metric", style=LBL),
            dcc.Dropdown(id="metric", options=METRIC_OPTIONS, value="rsrp", clearable=False, style=DD),
            html.Div(id="color-by-container", children=[
                html.Label("Color By (Point Map)", style=LBL),
                dcc.Dropdown(id="color-by", options=COLOR_BY_OPTIONS,
                             value="signal_quality", clearable=False, style=DD),
            ], style={{"display": "none"}}),
            html.Hr(style={{"borderColor": "#333"}}),
            html.H4("Filters", style={{"color": "#e94560"}}),
            html.Label("Network Type",   style=LBL),
            dcc.Dropdown(id="f-network",     options=opts(df["network_type"]),     multi=True, placeholder="All", style=DD),
            html.Label("Connectivity",   style=LBL),
            dcc.Dropdown(id="f-connectivity",options=opts(df["connectivity_type"]),multi=True, placeholder="All", style=DD),
            html.Label("Cell Type",      style=LBL),
            dcc.Dropdown(id="f-cell",       options=opts(df["cell_type"]),         multi=True, placeholder="All", style=DD),
            html.Label("Building",       style=LBL),
            dcc.Dropdown(id="f-building",   options=opts(df["building_label"]),    multi=True, placeholder="All", style=DD),
            html.Label("Signal Quality", style=LBL),
            dcc.Dropdown(id="f-quality",
                         options=[{{"label": q, "value": q}} for q in ["Excellent", "Good", "Fair", "Poor"]],
                         multi=True, placeholder="All", style=DD),
            html.Label("Date Range", style=LBL),
            dcc.DatePickerRange(id="f-dates",
                                min_date_allowed=df["EVENTDATE"].min(),
                                max_date_allowed=df["EVENTDATE"].max(),
                                start_date=df["EVENTDATE"].min(),
                                end_date=df["EVENTDATE"].max(),
                                style={{"marginBottom": "14px"}}),
            html.Hr(style={{"borderColor": "#333"}}),
            html.Div(id="stats-panel"),
        ], style=SIDEBAR),
        html.Div([dcc.Graph(id="map-graph", style={{"height": "82vh"}})],
                 style={{"flex": "1", "paddingLeft": "16px"}}),
    ], style={{"display": "flex", "padding": "16px"}}),
], style={{"fontFamily": "'Segoe UI', Arial, sans-serif", "backgroundColor": "#f5f5f5", "minHeight": "100vh"}})


@app.callback(Output("color-by-container", "style"), Input("view-mode", "value"))
def toggle_color_by(view):
    return {{"display": "block"}} if view == "scatter" else {{"display": "none"}}


@app.callback(
    [Output("map-graph", "figure"), Output("stats-panel", "children")],
    [Input("view-mode", "value"), Input("metric", "value"), Input("color-by", "value"),
     Input("f-network", "value"), Input("f-connectivity", "value"), Input("f-cell", "value"),
     Input("f-building", "value"), Input("f-quality", "value"),
     Input("f-dates", "start_date"), Input("f-dates", "end_date")],
)
def update_map(view, metric, color_by, networks, connectivities, cells,
               buildings, qualities, start_date, end_date):
    filtered = df.copy()
    if networks:       filtered = filtered[filtered["network_type"].isin(networks)]
    if connectivities: filtered = filtered[filtered["connectivity_type"].isin(connectivities)]
    if cells:          filtered = filtered[filtered["cell_type"].isin(cells)]
    if buildings:      filtered = filtered[filtered["building_label"].isin(buildings)]
    if qualities:      filtered = filtered[filtered["signal_quality"].isin(qualities)]
    if start_date:     filtered = filtered[filtered["EVENTDATE"] >= pd.to_datetime(start_date).date()]
    if end_date:       filtered = filtered[filtered["EVENTDATE"] <= pd.to_datetime(end_date).date()]

    center = ({{"lat": filtered["LATITUDE"].mean(), "lon": filtered["LONGITUDE"].mean()}}
              if len(filtered) > 0 else MAP_CENTER)

    if len(filtered) == 0:
        fig = px.scatter_mapbox(
            pd.DataFrame({{"lat": [MAP_CENTER["lat"]], "lon": [MAP_CENTER["lon"]]}}),
            lat="lat", lon="lon", zoom=DEFAULT_ZOOM, mapbox_style="carto-darkmatter")
        fig.update_layout(title="No data matches the current filters")
    elif view == "heatmap":
        plot_df = filtered.dropna(subset=[metric]).copy()
        plot_df["_z"] = ((plot_df["rsrp"] + 140) / 96 if metric == "rsrp"
                         else (plot_df["wifi_rssi"] + 100) / 60)
        plot_df["_z"] = plot_df["_z"].clip(0, 1)
        fig = px.density_mapbox(
            plot_df, lat="LATITUDE", lon="LONGITUDE", z="_z",
            radius=14, center=center, zoom=DEFAULT_ZOOM, mapbox_style="carto-darkmatter",
            color_continuous_scale=["#0d0887","#46039f","#7201a8","#9c179e",
                                    "#bd3786","#d8576b","#ed7953","#fb9f3a","#fdca26","#f0f921"])
        fig.update_layout(coloraxis_colorbar_title="RSRP (dBm)" if metric == "rsrp" else "WiFi RSSI (dBm)")
    else:
        sample = filtered.sample(n=min(8000, len(filtered)), random_state=42)
        fig = px.scatter_mapbox(
            sample, lat="LATITUDE", lon="LONGITUDE", color=color_by,
            hover_data=["rsrp", "wifi_rssi", "network_type", "building_label"],
            center=center, zoom=DEFAULT_ZOOM, mapbox_style="carto-darkmatter",
            color_discrete_map=(QUALITY_COLORS if color_by == "signal_quality" else None),
            category_orders={{"signal_quality": ["Excellent", "Good", "Fair", "Poor"]}},
            opacity=0.7)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

    n = len(filtered)
    if n > 0:
        avg_rsrp = filtered["rsrp"].mean()
        qc = filtered["signal_quality"].value_counts()
        stats = [
            html.H4("Summary", style={{"color": "#e94560", "marginTop": "0"}}),
            html.P(f"Points: {{n:,}}",               style={{"margin": "4px 0"}}),
            html.P(f"Avg RSRP: {{avg_rsrp:.1f}} dBm", style={{"margin": "4px 0"}}),
            html.Hr(style={{"borderColor": "#333"}}),
            html.P(f"Excellent: {{qc.get('Excellent',0):,}} ({{qc.get('Excellent',0)/n*100:.1f}}%)", style={{"margin":"2px 0","color":"#2ecc71"}}),
            html.P(f"Good:      {{qc.get('Good',0):,}} ({{qc.get('Good',0)/n*100:.1f}}%)",      style={{"margin":"2px 0","color":"#3498db"}}),
            html.P(f"Fair:      {{qc.get('Fair',0):,}} ({{qc.get('Fair',0)/n*100:.1f}}%)",      style={{"margin":"2px 0","color":"#f39c12"}}),
            html.P(f"Poor:      {{qc.get('Poor',0):,}} ({{qc.get('Poor',0)/n*100:.1f}}%)",      style={{"margin":"2px 0","color":"#e74c3c"}}),
        ]
    else:
        stats = [html.P("No data", style={{"color": "#888"}})]

    return fig, html.Div(stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
'''

with open(os.path.join(APP_DIR, "app.py"), "w") as f:
    f.write(app_py)
print(f"Wrote app.py ({len(app_py):,} chars)")

# ── requirements.txt ─────────────────────────────────────────────────────────
with open(os.path.join(APP_DIR, "requirements.txt"), "w") as f:
    f.write(
        "dash>=2.14.0\n"
        "plotly>=5.18.0\n"
        "pandas>=2.0.0\n"
        "databricks-sql-connector>=3.0.0\n"
        "databricks-sdk>=0.20.0\n"
    )
print("Wrote requirements.txt")

# ── app.yaml ─────────────────────────────────────────────────────────────────
with open(os.path.join(APP_DIR, "app.yaml"), "w") as f:
    f.write(
        f"command:\n"
        f'  - "python"\n'
        f'  - "app.py"\n\n'
        f"env:\n"
        f"  - name: DATABRICKS_WAREHOUSE_ID\n"
        f"    description: SQL warehouse ID used to query the Delta table\n"
        f"    value: \"{WAREHOUSE_ID}\"\n"
    )
print("Wrote app.yaml")
print(f"\nAll files written to {APP_DIR}")

# COMMAND ----------

# MAGIC %md ### 2. Create or retrieve the Databricks App

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App
import time

w = WorkspaceClient()

try:
    app_info = w.apps.get(APP_NAME)
    print(f"App '{APP_NAME}' already exists — skipping creation")
except Exception:
    print(f"Creating app '{APP_NAME}'...")
    app_info = w.apps.create_and_wait(
        app=App(
            name=APP_NAME,
            description="Interactive cell signal quality heatmap for Chandler Fashion Center",
        ),
    )
    print("App created.")

print(f"App URL : {app_info.url}")
sp_id   = app_info.service_principal_id
sp_name = app_info.service_principal_name
print(f"SP      : {sp_name} (ID: {sp_id})")

# COMMAND ----------

# MAGIC %md ### 3. Grant app service principal permissions

# COMMAND ----------

from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

sp_details = w.service_principals.get(sp_id)
sp_app_id  = sp_details.application_id
print(f"SP application_id : {sp_app_id}")

time.sleep(5)  # let SP propagate

# CAN_USE on the SQL warehouse
w.permissions.update(
    request_object_type="sql/warehouses",
    request_object_id=WAREHOUSE_ID,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=sp_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)
print(f"Granted CAN_USE on warehouse {WAREHOUSE_ID}")

# SELECT on the intersected table
spark.sql(f"GRANT SELECT ON TABLE {TARGET_TABLE} TO `{sp_app_id}`")
print(f"Granted SELECT on {TARGET_TABLE}")

# USAGE on catalog + schema
for stmt, label in [
    (f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{sp_app_id}`",             f"USE CATALOG {CATALOG}"),
    (f"GRANT USE SCHEMA ON SCHEMA `{CATALOG}`.`{SCHEMA}` TO `{sp_app_id}`",    f"USE SCHEMA {SCHEMA}"),
]:
    try:
        spark.sql(stmt)
        print(f"Granted {label}")
    except Exception as e:
        print(f"Note ({label}): {e}")

# COMMAND ----------

# MAGIC %md ### 4. Deploy the app

# COMMAND ----------

from databricks.sdk.service.apps import AppDeployment

print(f"Deploying '{APP_NAME}' from {APP_DIR} ...")
result = w.apps.deploy(
    app_name=APP_NAME,
    app_deployment=AppDeployment(source_code_path=APP_DIR),
).result()

print(f"Deployment ID : {result.deployment_id}")
print(f"Status        : {result.status}")
print(f"\nApp live at   : {app_info.url}")
