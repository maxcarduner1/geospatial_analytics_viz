# Databricks notebook source
# MAGIC %md
# MAGIC ## Build & Deploy — Signal Quality Dash App + Agent
# MAGIC
# MAGIC Generates two Databricks Apps:
# MAGIC - **Agent app** (`geospatial-signal-quality-agent`): FastAPI server wrapping the LangGraph agent with MLflow tracing
# MAGIC - **Dash app** (`geospatial-signal-quality`): Interactive map UI that calls the agent app's `/invocations` endpoint
# MAGIC
# MAGIC **Prerequisites:** `intersected_signal_points` table must exist (run `02_analysis` first).

# COMMAND ----------

dbutils.widgets.text("catalog",          "cmegdemos_catalog")
dbutils.widgets.text("schema",           "geospatial_analytics")
dbutils.widgets.text("warehouse_id",     "9cd919d96b11bf1c")
dbutils.widgets.text("app_name",         "geospatial-signal-quality")
dbutils.widgets.text("app_dir",          "/Workspace/Users/max.carduner@databricks.com/geospatial-signal-quality-app")
dbutils.widgets.text("agent_app_name",   "signal-quality-agent")
dbutils.widgets.text("agent_app_dir",    "/Workspace/Users/max.carduner@databricks.com/signal-quality-agent-app")

CATALOG        = dbutils.widgets.get("catalog")
SCHEMA         = dbutils.widgets.get("schema")
WAREHOUSE_ID   = dbutils.widgets.get("warehouse_id")
APP_NAME       = dbutils.widgets.get("app_name")
APP_DIR        = dbutils.widgets.get("app_dir")
AGENT_APP_NAME = dbutils.widgets.get("agent_app_name")
AGENT_APP_DIR  = dbutils.widgets.get("agent_app_dir")
TARGET_TABLE   = f"{CATALOG}.{SCHEMA}.intersected_signal_points"

print(f"Target table   : {TARGET_TABLE}")
print(f"Dash app name  : {APP_NAME}")
print(f"Dash app dir   : {APP_DIR}")
print(f"Agent app name : {AGENT_APP_NAME}")
print(f"Agent app dir  : {AGENT_APP_DIR}")
print(f"Warehouse      : {WAREHOUSE_ID}")

# COMMAND ----------

# MAGIC %md ### 1. Create UC functions for agent tools

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{CATALOG}`.`{SCHEMA}`.get_signal_stats_for_buildings(
    building_osm_ids STRING COMMENT 'Comma-separated building OSM IDs to look up'
)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Returns JSON signal quality stats for the specified buildings'
RETURN (
    SELECT to_json(collect_list(row_data))
    FROM (
        SELECT named_struct(
            'building_name',     COALESCE(building_name, CAST(building_osm_id AS STRING)),
            'building_osm_id',   CAST(building_osm_id AS STRING),
            'reading_count',     CAST(COUNT(*) AS BIGINT),
            'avg_rsrp_dbm',      ROUND(AVG(rsrp), 2),
            'min_rsrp_dbm',      ROUND(MIN(rsrp), 2),
            'max_rsrp_dbm',      ROUND(MAX(rsrp), 2),
            'excellent_pct',     ROUND(100.0 * COUNT(CASE WHEN rsrp > -80 THEN 1 END) / COUNT(*), 1),
            'good_pct',          ROUND(100.0 * COUNT(CASE WHEN rsrp BETWEEN -90 AND -80 THEN 1 END) / COUNT(*), 1),
            'fair_pct',          ROUND(100.0 * COUNT(CASE WHEN rsrp BETWEEN -100 AND -90 THEN 1 END) / COUNT(*), 1),
            'poor_pct',          ROUND(100.0 * COUNT(CASE WHEN rsrp < -100 THEN 1 END) / COUNT(*), 1),
            'avg_wifi_rssi_dbm', ROUND(AVG(wifi_rssi), 2),
            'network_types',     CAST(COLLECT_SET(COALESCE(network_type, 'unknown')) AS STRING)
        ) AS row_data
        FROM `{CATALOG}`.`{SCHEMA}`.intersected_signal_points
        WHERE array_contains(split(building_osm_ids, ','), CAST(building_osm_id AS STRING))
        GROUP BY building_name, building_osm_id
        ORDER BY COUNT(*) DESC
    ) t
)""")
print("Created get_signal_stats_for_buildings")

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{CATALOG}`.`{SCHEMA}`.get_buildings_list()
RETURNS STRING
LANGUAGE SQL
COMMENT 'Returns JSON list of all buildings with OSM IDs and reading counts'
RETURN (
    SELECT to_json(collect_list(row_data))
    FROM (
        SELECT named_struct(
            'building_name',   COALESCE(building_name, 'Unknown'),
            'building_osm_id', CAST(building_osm_id AS STRING),
            'reading_count',   CAST(COUNT(*) AS BIGINT)
        ) AS row_data
        FROM `{CATALOG}`.`{SCHEMA}`.intersected_signal_points
        GROUP BY building_name, building_osm_id
        ORDER BY COUNT(*) DESC
    ) t
)""")
print("Created get_buildings_list")

# COMMAND ----------

# MAGIC %md ### 2. Generate agent app source files
# MAGIC
# MAGIC The agent is a standalone FastAPI app that:
# MAGIC - Exposes `POST /invocations` with SSE streaming
# MAGIC - Wraps the LangGraph agent (same graph as before)
# MAGIC - Enables MLflow tracing via `mlflow.langchain.autolog()`

# COMMAND ----------

import os
os.makedirs(AGENT_APP_DIR, exist_ok=True)

# ── agent_app.py ──────────────────────────────────────────────────────────────
agent_app_py = f'''import os, json, logging, threading
import mlflow
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
import uvicorn

# ── MLflow tracing ────────────────────────────────────────────────────────────
mlflow.langchain.autolog()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATALOG    = os.getenv("CATALOG",    "{CATALOG}")
SCHEMA     = os.getenv("SCHEMA",     "{SCHEMA}")
UC_TOOLS   = [
    f"{{CATALOG}}.{{SCHEMA}}.get_signal_stats_for_buildings",
    f"{{CATALOG}}.{{SCHEMA}}.get_buildings_list",
]
SYSTEM_PROMPT = (
    "You are a signal quality analyst for Chandler Fashion Center. "
    "You have access to tools to query T-Mobile cell signal data measured inside mall buildings.\\n"
    "Key metrics:\\n"
    "- RSRP: Excellent >-80 dBm | Good -80 to -90 | Fair -90 to -100 | Poor <-100 dBm\\n"
    "- WiFi RSSI: WiFi signal level in dBm (higher is better)\\n"
    "When asked about buildings or a map selection, use get_signal_stats_for_buildings with "
    "the building OSM IDs. Use get_buildings_list if you need to find building IDs by name."
)

_agent = None
_lock  = threading.Lock()


def get_agent():
    global _agent
    with _lock:
        if _agent is not None:
            return _agent
        llm   = ChatDatabricks(endpoint="databricks-claude-opus-4-6", temperature=0.1)
        tools = UCFunctionToolkit(function_names=UC_TOOLS).tools
        llm_with_tools = llm.bind_tools(tools)
        tool_node      = ToolNode(tools)

        def call_model(state):
            return {{"messages": [llm_with_tools.invoke(state["messages"])]}}

        def should_continue(state):
            return "tools" if state["messages"][-1].tool_calls else END

        g = StateGraph(MessagesState)
        g.add_node("agent", call_model)
        g.add_node("tools", tool_node)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", should_continue)
        g.add_edge("tools", "agent")
        _agent = g.compile()
        logger.info("Agent graph compiled.")
        return _agent


def _to_lc_messages(msgs):
    result = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in msgs:
        role    = m.role
        content = m.content
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


async def _stream_agent(msgs):
    agent   = get_agent()
    lc_msgs = _to_lc_messages(msgs)
    async for mode, data in agent.astream(
        {{"messages": lc_msgs}}, stream_mode=["updates", "messages"]
    ):
        if mode != "messages":
            continue
        for msg in data:
            if not isinstance(msg, AIMessageChunk) or msg.tool_calls:
                continue
            text = ""
            if isinstance(msg.content, str):
                text = msg.content
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
            if text:
                yield text


app = FastAPI(title="Signal Quality Agent")


class Message(BaseModel):
    role: str
    content: str


class InvocationsRequest(BaseModel):
    input: List[Message]
    stream: bool = True


@app.post("/invocations")
async def invocations(req: InvocationsRequest):
    async def generate():
        async for chunk in _stream_agent(req.input):
            yield f"data: {{json.dumps({{'type': 'text_delta', 'text': chunk}})}}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {{"status": "ok"}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

with open(os.path.join(AGENT_APP_DIR, "agent_app.py"), "w") as f:
    f.write(agent_app_py)
print(f"Wrote agent_app.py ({len(agent_app_py):,} chars)")

# ── requirements.txt ──────────────────────────────────────────────────────────
with open(os.path.join(AGENT_APP_DIR, "requirements.txt"), "w") as f:
    f.write(
        "fastapi>=0.100.0\n"
        "uvicorn>=0.22.0\n"
        "databricks-sdk>=0.20.0\n"
        "databricks-langchain>=0.4.0\n"
        "langgraph>=0.2.0\n"
        "langchain-core>=0.3.0\n"
        "mlflow>=2.21.0\n"
    )
print("Wrote agent requirements.txt")

# ── app.yaml ──────────────────────────────────────────────────────────────────
with open(os.path.join(AGENT_APP_DIR, "app.yaml"), "w") as f:
    f.write(
        f'command:\n'
        f'  - "python"\n'
        f'  - "agent_app.py"\n\n'
        f'env:\n'
        f'  - name: MLFLOW_TRACKING_URI\n'
        f'    value: "databricks"\n'
        f'  - name: CATALOG\n'
        f'    value: "{CATALOG}"\n'
        f'  - name: SCHEMA\n'
        f'    value: "{SCHEMA}"\n'
    )
print("Wrote agent app.yaml")
print(f"\nAgent app files written to {AGENT_APP_DIR}")

# COMMAND ----------

# MAGIC %md ### 3. Create or retrieve the agent Databricks App

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App
import time

w = WorkspaceClient()

try:
    agent_app_info = w.apps.get(AGENT_APP_NAME)
    print(f"Agent app '{AGENT_APP_NAME}' already exists — skipping creation")
except Exception:
    print(f"Creating agent app '{AGENT_APP_NAME}'...")
    agent_app_info = w.apps.create_and_wait(
        app=App(
            name=AGENT_APP_NAME,
            description="LangGraph signal quality agent with MLflow tracing",
        ),
    )
    print("Agent app created.")

print(f"Agent app URL : {agent_app_info.url}")
agent_sp_id   = agent_app_info.service_principal_id
agent_sp_name = agent_app_info.service_principal_name
print(f"Agent SP      : {agent_sp_name} (ID: {agent_sp_id})")

# COMMAND ----------

# MAGIC %md ### 4. Grant agent app SP permissions

# COMMAND ----------

from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

agent_sp_details = w.service_principals.get(agent_sp_id)
agent_sp_app_id  = agent_sp_details.application_id
print(f"Agent SP application_id : {agent_sp_app_id}")

time.sleep(5)  # let SP propagate

# USAGE on catalog + schema
for stmt, label in [
    (f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{agent_sp_app_id}`",           f"USE CATALOG {CATALOG}"),
    (f"GRANT USE SCHEMA ON SCHEMA `{CATALOG}`.`{SCHEMA}` TO `{agent_sp_app_id}`",  f"USE SCHEMA {SCHEMA}"),
]:
    try:
        spark.sql(stmt)
        print(f"Granted {label}")
    except Exception as e:
        print(f"Note ({label}): {e}")

# EXECUTE on UC functions
for fn_name in [
    f"`{CATALOG}`.`{SCHEMA}`.get_signal_stats_for_buildings",
    f"`{CATALOG}`.`{SCHEMA}`.get_buildings_list",
]:
    try:
        spark.sql(f"GRANT EXECUTE ON FUNCTION {fn_name} TO `{agent_sp_app_id}`")
        print(f"Granted EXECUTE on {fn_name}")
    except Exception as e:
        print(f"Note (EXECUTE {fn_name}): {e}")

# COMMAND ----------

# MAGIC %md ### 5. Deploy agent app

# COMMAND ----------

from databricks.sdk.service.apps import AppDeployment

print(f"Deploying agent app '{AGENT_APP_NAME}' from {AGENT_APP_DIR} ...")
agent_deploy = w.apps.deploy(
    app_name=AGENT_APP_NAME,
    app_deployment=AppDeployment(source_code_path=AGENT_APP_DIR),
).result()

print(f"Deployment ID : {agent_deploy.deployment_id}")
print(f"Status        : {agent_deploy.status}")

# Capture URL for dash app wiring
AGENT_APP_URL = agent_app_info.url.rstrip("/")
print(f"\nAgent live at : {AGENT_APP_URL}")
print(f"Invocations   : {AGENT_APP_URL}/invocations")

# COMMAND ----------

# MAGIC %md ### 6. Generate dash app source files
# MAGIC
# MAGIC Key changes vs. previous version:
# MAGIC - Agent runs as a separate app; Dash calls `AGENT_APP_URL/invocations` via streaming HTTP
# MAGIC - Lasso/box selection **auto-sends** "Analyze cell signal quality for these buildings"
# MAGIC - MLflow tracing handled in the agent app

# COMMAND ----------

os.makedirs(APP_DIR, exist_ok=True)

# ── app.py ────────────────────────────────────────────────────────────────────
app_py = f'''import os, threading, uuid, time, json
import requests
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks import sql

TABLE        = "{TARGET_TABLE}"
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "{WAREHOUSE_ID}")
AGENT_APP_URL = os.getenv("AGENT_APP_URL", "").rstrip("/")
MAP_CENTER   = {{"lat": 33.3013, "lon": -111.8986}}

_SCHEMA  = ".".join(TABLE.split(".")[:2])
DEFAULT_ZOOM = 15

# customdata indices in scatter hover_data (order must match hover_data list in update_map)
CD_RSRP   = 0
CD_WIFI   = 1
CD_NETWORK = 2
CD_LABEL  = 3  # building_label (name fallback)
CD_OSM_ID = 4  # building_osm_id

SESSIONS      = {{}}
SESSIONS_LOCK = threading.Lock()

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


# ── Agent HTTP client ─────────────────────────────────────────────────────────

class _DatabricksAuth(requests.auth.AuthBase):
    """Attach Databricks workspace credentials to outgoing requests."""
    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers.update(w.config.authenticate())
        return r


def run_agent_via_http(session_id: str, messages_history: list):
    """Background thread: POST to agent app /invocations, stream SSE, update SESSIONS."""
    input_msgs = [
        {{"role": m["role"], "content": m["text"]}}
        for m in messages_history
    ]
    text_acc = ""
    try:
        with requests.post(
            f"{{AGENT_APP_URL}}/invocations",
            json={{"input": input_msgs, "stream": True}},
            auth=_DatabricksAuth(),
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_s = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line_s.startswith("data: "):
                    continue
                data_s = line_s[6:]
                if data_s == "[DONE]":
                    break
                try:
                    ev = json.loads(data_s)
                    if ev.get("type") == "text_delta":
                        text_acc += ev.get("text", "")
                except json.JSONDecodeError:
                    pass
                with SESSIONS_LOCK:
                    if session_id in SESSIONS:
                        SESSIONS[session_id]["text"] = text_acc
    except Exception as exc:
        text_acc = f"Agent error: {{exc}}"

    with SESSIONS_LOCK:
        if session_id in SESSIONS:
            SESSIONS[session_id]["text"] = text_acc or "No response received."
            SESSIONS[session_id]["done"] = True


def _start_agent_thread(session_id, history_for_agent):
    with SESSIONS_LOCK:
        SESSIONS[session_id] = {{"text": "", "done": False}}
    threading.Thread(
        target=run_agent_via_http,
        args=(session_id, history_for_agent),
        daemon=True,
    ).start()


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
            #chat-input:focus {{ outline: none; border-color: #e94560 !important; }}
            @keyframes blink-dot {{ 0%, 80%, 100% {{ opacity: 0; transform: scale(0.8); }} 40% {{ opacity: 1; transform: scale(1); }} }}
            .typing-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #888; margin: 0 3px; animation: blink-dot 1.4s infinite; }}
            .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
            .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
            .ai-md {{ color: #eee; font-size: 13px; line-height: 1.6; }}
            .ai-md p {{ margin: 0 0 8px 0; }}
            .ai-md ul, .ai-md ol {{ padding-left: 18px; margin: 4px 0; }}
            .ai-md li {{ margin: 3px 0; }}
            .ai-md strong {{ color: #fff; }}
            .ai-md h1, .ai-md h2, .ai-md h3 {{ font-size: 13px; font-weight: 700; margin: 8px 0 4px; color: #e94560; }}
            .ai-md code {{ background: #0a0e1a; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
            .ai-md pre {{ background: #0a0e1a; padding: 10px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }}
            .ai-md pre code {{ background: none; padding: 0; }}
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

AI_BTN_STYLE = {{
    "background": "#e94560", "color": "#fff", "border": "none",
    "borderRadius": "6px", "padding": "8px 18px", "cursor": "pointer",
    "fontWeight": "600", "fontSize": "14px", "whiteSpace": "nowrap",
    "flexShrink": "0",
}}
AI_PANEL_BASE = {{
    "flexDirection": "column",
    "position": "fixed",
    "right": "0",
    "top": "0",
    "width": "390px",
    "height": "100vh",
    "backgroundColor": "#1a1a2e",
    "color": "#eee",
    "zIndex": "1000",
    "boxShadow": "-6px 0 28px rgba(0,0,0,0.55)",
}}


def _render_messages(messages):
    if not messages:
        return [html.P(
            "Ask anything about cell signal quality at Chandler Fashion Center. "
            "Use lasso or box select in Point Map mode to auto-analyze selected buildings.",
            style={{"color": "#555", "fontSize": "13px", "textAlign": "center",
                   "marginTop": "40px", "padding": "0 20px", "lineHeight": "1.6"}},
        )]
    items = []
    for msg in messages:
        role = msg["role"]
        if role == "typing":
            items.append(_typing_bubble())
            continue
        is_user = role == "user"
        if is_user:
            content = html.Div(msg["text"], style={{
                "background": "#e94560", "color": "#fff",
                "borderRadius": "12px 12px 2px 12px",
                "padding": "10px 14px", "maxWidth": "88%",
                "fontSize": "13px", "lineHeight": "1.6",
            }})
        else:
            content = dcc.Markdown(msg["text"], className="ai-md", style={{
                "background": "#16213e",
                "borderRadius": "12px 12px 12px 2px",
                "padding": "10px 14px", "maxWidth": "88%",
            }})
        items.append(html.Div(content, style={{
            "display": "flex",
            "justifyContent": "flex-end" if is_user else "flex-start",
            "marginBottom": "10px",
        }}))
    return items


def _typing_bubble():
    return html.Div(
        html.Div([
            html.Span(className="typing-dot"),
            html.Span(className="typing-dot"),
            html.Span(className="typing-dot"),
        ], style={{"background": "#16213e", "borderRadius": "12px 12px 12px 2px",
                  "padding": "14px 16px", "display": "inline-flex", "alignItems": "center"}}),
        style={{"display": "flex", "justifyContent": "flex-start", "marginBottom": "10px"}},
    )


app.layout = html.Div([
    dcc.Store(id="chat-store",          data={{"conv_id": None, "messages": []}}),
    dcc.Store(id="selection-store",     data={{"buildings": [], "point_count": 0}}),
    dcc.Store(id="stream-state",        data={{"active": False, "session_id": None, "messages": []}}),
    dcc.Store(id="autoprompt-trigger",  data=None),
    dcc.Interval(id="stream-interval", interval=400, disabled=True, n_intervals=0),

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H2("Cell Signal Quality \u2014 Chandler Fashion Center",
                    style={{"margin": "0", "color": "#1a1a2e"}}),
            html.P("Interactive spatial heatmap of RSRP signal strength inside building polygons",
                   style={{"margin": "4px 0 0", "color": "#666", "fontSize": "14px"}}),
        ], style={{"flex": "1"}}),
        html.Button("\u2728 AI Assistant", id="ai-toggle-btn", n_clicks=0, style=AI_BTN_STYLE),
    ], style={{
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "16px 24px", "borderBottom": "2px solid #e0e0e0",
    }}),

    # ── Main body ─────────────────────────────────────────────────────────────
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

    # ── AI Assistant panel (fixed right overlay) ───────────────────────────
    html.Div([
        html.Div([
            html.Span("\u2728 AI Assistant",
                      style={{"fontWeight": "700", "fontSize": "15px", "color": "#e94560"}}),
            html.Button("\u00d7", id="ai-close-btn", n_clicks=0, style={{
                "background": "none", "border": "none", "color": "#888",
                "fontSize": "24px", "cursor": "pointer", "lineHeight": "1", "padding": "0",
            }}),
        ], style={{
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "14px 18px", "borderBottom": "1px solid #2a2a4a", "flexShrink": "0",
        }}),
        html.Div([
            html.Span(id="selection-banner-text",
                      style={{"fontSize": "12px", "color": "#eee", "flex": "1",
                              "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}}),
            html.Button("\u00d7 Clear", id="clear-selection-btn", n_clicks=0, style={{
                "background": "none", "border": "1px solid #555", "color": "#aaa",
                "borderRadius": "4px", "padding": "2px 8px", "cursor": "pointer",
                "fontSize": "11px", "flexShrink": "0",
            }}),
        ], id="selection-banner", style={{
            "display": "none",
            "alignItems": "center", "gap": "8px",
            "padding": "8px 14px", "backgroundColor": "#0f3460",
            "borderBottom": "1px solid #2a2a4a", "flexShrink": "0",
        }}),
        html.Div(
            _render_messages([]),
            id="chat-messages",
            style={{"flex": "1", "overflowY": "auto", "padding": "16px", "minHeight": "100px"}},
        ),
        html.Div([
            dcc.Input(
                id="chat-input", type="text", n_submit=0,
                placeholder="Ask about signal quality...",
                style={{
                    "flex": "1", "background": "#16213e", "color": "#eee",
                    "border": "1px solid #2a2a4a", "borderRadius": "6px",
                    "padding": "9px 12px", "fontSize": "13px",
                }},
            ),
            html.Button("Send", id="send-btn", n_clicks=0, style={{
                "background": "#e94560", "color": "#fff", "border": "none",
                "borderRadius": "6px", "padding": "9px 16px",
                "cursor": "pointer", "fontWeight": "600", "fontSize": "13px",
                "flexShrink": "0",
            }}),
        ], style={{
            "display": "flex", "gap": "8px", "padding": "12px 16px",
            "borderTop": "1px solid #2a2a4a", "flexShrink": "0",
        }}),
    ], id="ai-panel", style={{**AI_PANEL_BASE, "display": "none"}}),

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
            hover_data=["rsrp", "wifi_rssi", "network_type", "building_label", "building_osm_id"],
            center=center, zoom=DEFAULT_ZOOM, mapbox_style="carto-darkmatter",
            color_discrete_map=(QUALITY_COLORS if color_by == "signal_quality" else None),
            category_orders={{"signal_quality": ["Excellent", "Good", "Fair", "Poor"]}},
            opacity=0.7)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), uirevision="map")

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


@app.callback(
    Output("ai-panel", "style"),
    Output("ai-toggle-btn", "children"),
    Input("ai-toggle-btn", "n_clicks"),
    Input("ai-close-btn", "n_clicks"),
    State("ai-panel", "style"),
    prevent_initial_call=True,
)
def toggle_ai_panel(open_clicks, close_clicks, panel_style):
    is_open = panel_style.get("display") == "flex"
    if ctx.triggered_id == "ai-close-btn" or is_open:
        return {{**panel_style, "display": "none"}}, "\u2728 AI Assistant"
    return {{**panel_style, "display": "flex"}}, "\u00d7 Close"


@app.callback(
    Output("selection-store", "data"),
    Output("ai-panel", "style", allow_duplicate=True),
    Output("ai-toggle-btn", "children", allow_duplicate=True),
    Output("autoprompt-trigger", "data"),
    Input("map-graph", "selectedData"),
    State("ai-panel", "style"),
    prevent_initial_call=True,
)
def on_map_selection(selected_data, panel_style):
    """Fire on lasso or box select. Extract buildings, auto-open AI panel, trigger autoprompt."""
    empty_store = {{"buildings": [], "point_count": 0}}
    if not selected_data or not selected_data.get("points"):
        return empty_store, no_update, no_update, no_update

    points = selected_data["points"]
    seen   = {{}}  # label → osm_id (deduplicated)
    for p in points:
        cd = p.get("customdata") or []
        if len(cd) > CD_OSM_ID:
            label  = cd[CD_LABEL]
            osm_id = str(cd[CD_OSM_ID]) if cd[CD_OSM_ID] is not None else None
            if label not in seen:
                seen[label] = osm_id

    buildings = [{{"label": lbl, "osm_id": oid}} for lbl, oid in sorted(seen.items())]
    new_store = {{"buildings": buildings, "point_count": len(points)}}

    if not buildings:
        return new_store, no_update, no_update, no_update

    # Auto-open panel + fire autoprompt
    new_style = {{**panel_style, "display": "flex"}}
    trigger   = {{"prompt": "Analyze cell signal quality for these buildings", "ts": time.time()}}
    return new_store, new_style, "\u00d7 Close", trigger


@app.callback(
    Output("selection-banner", "style"),
    Output("selection-banner-text", "children"),
    Input("selection-store", "data"),
)
def update_selection_banner(store):
    buildings   = store.get("buildings", [])
    point_count = store.get("point_count", 0)
    hidden = {{"display": "none", "alignItems": "center", "gap": "8px",
               "padding": "8px 14px", "backgroundColor": "#0f3460",
               "borderBottom": "1px solid #2a2a4a", "flexShrink": "0"}}
    shown  = {{**hidden, "display": "flex"}}
    if not buildings:
        return hidden, ""
    names = ", ".join(b["label"] for b in buildings[:3])
    if len(buildings) > 3:
        names += f" +{{len(buildings)-3}} more"
    text = f"\U0001f4cd {{names}} \u00b7 {{point_count:,}} pts"
    return shown, text


@app.callback(
    Output("selection-store", "data", allow_duplicate=True),
    Input("clear-selection-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_selection(n_clicks):
    return {{"buildings": [], "point_count": 0}}


def _build_agent_history(question, prior_messages, selection):
    """Return (user_msg_for_ui, history_for_agent) with map selection context injected."""
    sel_buildings = selection.get("buildings", [])
    sel_count     = selection.get("point_count", 0)
    if sel_buildings:
        osm_ids = [b["osm_id"] for b in sel_buildings if b.get("osm_id")]
        labels  = [b["label"]  for b in sel_buildings]
        display = ", ".join(labels[:5])
        if len(labels) > 5:
            display += f" +{{len(labels)-5}} more"
        osm_list = ",".join(osm_ids) if osm_ids else ""
        context  = (
            f"[Map selection: {{sel_count:,}} readings from {{len(sel_buildings)}} building(s): {{display}}."
            + (f" OSM IDs for tool: {{osm_list}}." if osm_list else "") + "]"
        )
        full_question = context + "\\n\\n" + question
    else:
        full_question = question
    user_msg          = {{"role": "user", "text": question}}
    history_for_agent = prior_messages + [{{"role": "user", "text": full_question}}]
    return user_msg, history_for_agent


@app.callback(
    Output("chat-messages", "children"),
    Output("stream-state", "data"),
    Output("stream-interval", "disabled"),
    Output("chat-input", "value"),
    Input("send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    State("selection-store", "data"),
    prevent_initial_call=True,
)
def on_send(n_send, n_submit, question, store, selection):
    """Show user message + typing dots immediately, start agent thread, enable interval."""
    if not question or not question.strip():
        raise PreventUpdate
    question       = question.strip()
    prior_messages = list(store.get("messages", []))
    user_msg, history_for_agent = _build_agent_history(question, prior_messages, selection)

    messages_for_ui = prior_messages + [user_msg]
    session_id      = str(uuid.uuid4())
    _start_agent_thread(session_id, history_for_agent)

    typing_msgs  = messages_for_ui + [{{"role": "typing", "text": ""}}]
    stream_state = {{"active": True, "session_id": session_id, "messages": messages_for_ui}}
    return _render_messages(typing_msgs), stream_state, False, ""


@app.callback(
    Output("chat-messages", "children", allow_duplicate=True),
    Output("stream-state", "data", allow_duplicate=True),
    Output("stream-interval", "disabled", allow_duplicate=True),
    Output("chat-input", "value", allow_duplicate=True),
    Input("autoprompt-trigger", "data"),
    State("chat-store", "data"),
    State("selection-store", "data"),
    prevent_initial_call=True,
)
def on_autoprompt(trigger, store, selection):
    """Fires automatically when lasso/box selection contains buildings."""
    if not trigger:
        raise PreventUpdate
    buildings = selection.get("buildings", [])
    if not buildings:
        raise PreventUpdate

    question       = trigger.get("prompt", "Analyze cell signal quality for these buildings")
    prior_messages = list(store.get("messages", []))
    user_msg, history_for_agent = _build_agent_history(question, prior_messages, selection)

    messages_for_ui = prior_messages + [user_msg]
    session_id      = str(uuid.uuid4())
    _start_agent_thread(session_id, history_for_agent)

    typing_msgs  = messages_for_ui + [{{"role": "typing", "text": ""}}]
    stream_state = {{"active": True, "session_id": session_id, "messages": messages_for_ui}}
    return _render_messages(typing_msgs), stream_state, False, ""


@app.callback(
    Output("chat-messages", "children", allow_duplicate=True),
    Output("chat-store", "data"),
    Output("stream-state", "data", allow_duplicate=True),
    Output("stream-interval", "disabled", allow_duplicate=True),
    Input("stream-interval", "n_intervals"),
    State("stream-state", "data"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def poll_stream(n_intervals, stream_state, store):
    """Poll agent thread every 400 ms; show partial text until done."""
    if not stream_state or not stream_state.get("active"):
        raise PreventUpdate
    session_id = stream_state.get("session_id")
    if not session_id:
        raise PreventUpdate

    with SESSIONS_LOCK:
        sess = SESSIONS.get(session_id, {{}})
        text = sess.get("text", "")
        done = sess.get("done", False)

    messages_for_ui = list(stream_state.get("messages", []))

    if done:
        with SESSIONS_LOCK:
            SESSIONS.pop(session_id, None)
        final_msgs = messages_for_ui + [{{"role": "assistant", "text": text or "No response received."}}]
        new_store  = {{"conv_id": store.get("conv_id"), "messages": final_msgs}}
        cleared    = {{"active": False, "session_id": None, "messages": []}}
        return _render_messages(final_msgs), new_store, cleared, True

    if text:
        partial = messages_for_ui + [
            {{"role": "assistant", "text": text}},
            {{"role": "typing",    "text": ""}},
        ]
    else:
        partial = messages_for_ui + [{{"role": "typing", "text": ""}}]
    return _render_messages(partial), no_update, no_update, no_update


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
'''

with open(os.path.join(APP_DIR, "app.py"), "w") as f:
    f.write(app_py)
print(f"Wrote app.py ({len(app_py):,} chars)")

# ── requirements.txt ──────────────────────────────────────────────────────────
with open(os.path.join(APP_DIR, "requirements.txt"), "w") as f:
    f.write(
        "dash>=2.14.0\n"
        "plotly>=5.18.0\n"
        "pandas>=2.0.0\n"
        "requests>=2.31.0\n"
        "databricks-sql-connector>=3.0.0\n"
        "databricks-sdk>=0.20.0\n"
    )
print("Wrote dash requirements.txt")

# ── app.yaml (written here so it's in the snapshot when deploy is called) ─────
with open(os.path.join(APP_DIR, "app.yaml"), "w") as f:
    f.write(
        f"command:\n"
        f'  - "python"\n'
        f'  - "app.py"\n\n'
        f"env:\n"
        f"  - name: DATABRICKS_WAREHOUSE_ID\n"
        f"    description: SQL warehouse ID used to query the Delta table\n"
        f'    value: "{WAREHOUSE_ID}"\n'
        f"  - name: AGENT_APP_URL\n"
        f"    description: URL of the signal quality agent app\n"
        f'    value: "{AGENT_APP_URL}"\n'
    )
print(f"Wrote dash app.yaml (AGENT_APP_URL={AGENT_APP_URL})")
print(f"\nAll dash app files written to {APP_DIR}")

# COMMAND ----------

# MAGIC %md ### 7. Create or retrieve the dash Databricks App

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

# MAGIC %md ### 8. Grant dash app SP permissions
# MAGIC
# MAGIC Includes `CAN_USE` on the agent app so the dash app SP can call `/invocations`.

# COMMAND ----------

from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

sp_details = w.service_principals.get(sp_id)
sp_app_id  = sp_details.application_id
print(f"Dash SP application_id : {sp_app_id}")

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

# CAN_USE on the agent app (so dash SP can call /invocations)
try:
    w.permissions.update(
        request_object_type="apps",
        request_object_id=AGENT_APP_NAME,
        access_control_list=[
            AccessControlRequest(
                service_principal_name=sp_app_id,
                permission_level=PermissionLevel.CAN_USE,
            )
        ],
    )
    print(f"Granted CAN_USE on agent app '{AGENT_APP_NAME}'")
except Exception as e:
    print(f"Note (CAN_USE on agent app): {e}")

# SELECT on the intersected table
spark.sql(f"GRANT SELECT ON TABLE {TARGET_TABLE} TO `{sp_app_id}`")
print(f"Granted SELECT on {TARGET_TABLE}")

# USAGE on catalog + schema
for stmt, label in [
    (f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{sp_app_id}`",           f"USE CATALOG {CATALOG}"),
    (f"GRANT USE SCHEMA ON SCHEMA `{CATALOG}`.`{SCHEMA}` TO `{sp_app_id}`",  f"USE SCHEMA {SCHEMA}"),
]:
    try:
        spark.sql(stmt)
        print(f"Granted {label}")
    except Exception as e:
        print(f"Note ({label}): {e}")

# COMMAND ----------

# MAGIC %md ### 9. Deploy the dash app

# COMMAND ----------

from databricks.sdk.service.apps import AppDeployment

print(f"\nDeploying '{APP_NAME}' from {APP_DIR} ...")
result = w.apps.deploy(
    app_name=APP_NAME,
    app_deployment=AppDeployment(source_code_path=APP_DIR),
).result()

print(f"Deployment ID : {result.deployment_id}")
print(f"Status        : {result.status}")
print(f"\nDash app live at : {app_info.url}")
print(f"Agent app live at : {AGENT_APP_URL}")
