"""MDF file tools for agent-driven data exploration.

Provides two tools:
- explore_mdf: lists all channels in an MDF file with basic metadata.
- read_mdf_channels: reads one or more channels and returns statistics + a data preview.
"""

import json
from pathlib import Path

import numpy as np
from langchain.tools import tool


def _signal_summary(signal) -> dict:
    """Return a compact summary dict for an asammdf Signal object."""
    samples = signal.samples
    timestamps = signal.timestamps

    stats: dict = {}
    if len(samples) > 0:
        try:
            stats = {
                "min": float(np.min(samples)),
                "max": float(np.max(samples)),
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
            }
        except (TypeError, ValueError):
            stats = {"note": "non-numeric channel — statistics not available"}

    # Include up to 5 samples from the start and end as a preview
    ts_list = timestamps.tolist()
    s_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
    preview_ts = ts_list[:5] + (ts_list[-5:] if len(ts_list) > 5 else [])
    preview_s = s_list[:5] + (s_list[-5:] if len(s_list) > 5 else [])

    return {
        "name": signal.name,
        "unit": signal.unit or "",
        "samples_count": len(samples),
        "time_start_s": float(timestamps[0]) if len(timestamps) > 0 else None,
        "time_end_s": float(timestamps[-1]) if len(timestamps) > 0 else None,
        **stats,
        "preview": {"timestamps": preview_ts, "values": preview_s},
    }


@tool
def explore_mdf(file_path: str) -> str:
    """Explore an MDF (.mdf / .mf4) file and return its metadata and the full list
    of available channels with their units and sample counts.

    Call this first to understand the file structure before reading specific channels.

    Args:
        file_path: Absolute or relative path to the MDF file.

    Returns:
        JSON string with file version, total channel count, and per-channel metadata.
    """
    try:
        from asammdf import MDF

        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})

        with MDF(str(path)) as mdf:
            channels = []
            for name in sorted(mdf.channels_db.keys()):
                try:
                    sig = mdf.get(name)
                    channels.append(
                        {
                            "name": name,
                            "unit": sig.unit or "",
                            "samples_count": len(sig.samples),
                        }
                    )
                except Exception as e:
                    channels.append({"name": name, "error": str(e)})

            return json.dumps(
                {
                    "file": path.name,
                    "version": str(mdf.version),
                    "total_channels": len(channels),
                    "channels": channels,
                },
                indent=2,
            )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def read_mdf_channels(file_path: str, channel_names: list[str]) -> str:
    """Read one or more channels from an MDF file and return statistics and a data preview.

    Use explore_mdf first to discover available channel names.

    Args:
        file_path: Absolute or relative path to the MDF file.
        channel_names: List of channel names to read.

    Returns:
        JSON string with per-channel statistics (min, max, mean, std), time range,
        and a preview of the first and last 5 samples.
    """
    try:
        from asammdf import MDF

        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})

        results = []
        with MDF(str(path)) as mdf:
            for name in channel_names:
                try:
                    sig = mdf.get(name)
                    results.append(_signal_summary(sig))
                except Exception as e:
                    results.append({"name": name, "error": str(e)})

        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
