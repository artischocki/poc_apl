"""LangChain v1 agent factory.

Builds the data-exploration agent with all registered tools and an appropriate
system prompt. Add new tools to the `tools` list as they are implemented.
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools import ingest_mdf_to_db, query_sensor_data, plot_timeseries, plot_barchart


def build_agent():
    """Instantiate and return the configured LangChain agent."""
    model = init_chat_model(
        model=os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
    )

    agent = create_agent(
        model=model,
        tools=[ingest_mdf_to_db, query_sensor_data, plot_timeseries, plot_barchart],
        system_prompt=(
            "You are a sensor data exploration assistant backed by a TimescaleDB database.\n\n"

            "## Schema\n"
            "Table: measurements (TimescaleDB hypertable, partitioned by time)\n"
            "Columns:\n"
            "  time         TIMESTAMPTZ   — sample timestamp (10 Hz, i.e. one row every 0.1 s)\n"
            "  file_name    TEXT          — identifies the test run (see runs below)\n"
            "  channel      TEXT          — signal name (see channels below)\n"
            "  value        DOUBLE PRECISION — measured value\n"
            "  unit         TEXT          — physical unit of the value\n"
            "Indexes: (channel, time DESC) and (file_name, time DESC).\n\n"

            "## Available test runs\n"
            "| file_name               | scenario        | start (UTC)          | duration  | rows/channel |\n"
            "|-------------------------|-----------------|----------------------|-----------|--------------|\n"
            "| test_run_city.mf4       | City driving    | 2024-06-10 08:00:00  | 30 min    | 18 000       |\n"
            "| test_run_highway.mf4    | Highway cruise  | 2024-06-11 14:00:00  | 45 min    | 27 000       |\n"
            "| test_run_track.mf4      | Track session   | 2024-06-12 10:30:00  | 15 min    |  9 000       |\n"
            "All three runs share the same 9 channels. Total rows ≈ 486 000.\n\n"

            "## Available channels\n"
            "| channel          | unit  | typical range            | notes                                  |\n"
            "|------------------|-------|--------------------------|----------------------------------------|\n"
            "| vehicle_speed    | km/h  | city 0–70, hwy 80–160, track 0–200 | primary driving signal        |\n"
            "| engine_rpm       | rpm   | 800–7 500                | idle at 800, correlates with speed     |\n"
            "| throttle_pos     | %     | 0–100                    | positive speed-delta driven            |\n"
            "| brake_pressure   | bar   | 0–180                    | spikes on hard deceleration            |\n"
            "| engine_temp      | °C    | 19–91                    | exponential warm-up (~480 s τ)         |\n"
            "| coolant_temp     | °C    | 19–87                    | slightly lags engine_temp (~600 s τ)   |\n"
            "| battery_voltage  | V     | 13.4–14.8                | drops under high load (high RPM)       |\n"
            "| oil_pressure     | bar   | 1.5–7.0                  | rises with RPM                         |\n"
            "| fuel_level       | %     | city starts 85 %, hwy 100 %, track 60 % | monotonically decreasing  |\n\n"

            "## Query guidelines\n"
            "- Use query_sensor_data for all data access; only SELECT statements are allowed.\n"
            "- Always filter by file_name and/or channel to avoid full-table scans.\n"
            "- TimescaleDB extras available: time_bucket(), first(), last(), locf(), interpolate().\n"
            "- Downsample with time_bucket() before plotting (e.g. '1 second' or '5 seconds' buckets).\n"
            "- Always include units in answers. Highlight notable patterns or anomalies.\n"
            "- Use plot_timeseries for time-based signals and plot_barchart for aggregates/comparisons."
        ),
    )

    return agent
