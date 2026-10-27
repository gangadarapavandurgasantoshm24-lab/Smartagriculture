import base64
import datetime
import io
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import streamlit as st


def _ndarray_to_base64_png(arr: Optional[np.ndarray]) -> Optional[str]:
    """Convert an RGB uint8 image array to a Base64-encoded PNG data URI."""
    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        return None
    try:
        # Array is in RGB; cv2 expects BGR for imencode
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", bgr)
        if not success:
            return None
        b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception:
        return None


def init_session_history() -> None:
    """Initialize in-memory session scan history in Streamlit session state."""
    if "session_scans" not in st.session_state:
        st.session_state["session_scans"] = []
    if "active_report_scan_id" not in st.session_state:
        st.session_state["active_report_scan_id"] = None


def add_session_scan(
    filename: str,
    crop: str,
    condition: str,
    is_healthy: bool,
    predicted_class: str,
    confidence: float,
    top_predictions: List[Dict[str, Any]],
    display_image: Optional[np.ndarray] = None,
    overlay_image: Optional[np.ndarray] = None,
    notes: str = "",
    confirmed_crop: Optional[str] = None,
    guidance_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Store a diagnostic scan entry strictly in-memory in the current Streamlit session.
    No persistent files or database rows are created.
    """
    init_session_history()

    # Generate sequential unique session scan ID
    scan_count = len(st.session_state["session_scans"]) + 1
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_code = datetime.datetime.now().strftime("%Y%m%d")
    unique_id = f"SCAN-{time_code}-{scan_count:04d}"

    # Convert images to lightweight Base64 data URIs for report rendering
    img_b64 = _ndarray_to_base64_png(display_image)
    overlay_b64 = _ndarray_to_base64_png(overlay_image)

    scan_record = {
        "id": unique_id,
        "numeric_id": scan_count,
        "timestamp": timestamp_str,
        "filename": filename or "custom_leaf.jpg",
        "crop": crop,
        "condition": condition,
        "is_healthy": bool(is_healthy),
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "confidence_percent": float(confidence * 100.0),
        "top_predictions": top_predictions,
        "notes": notes,
        "confirmed_crop": confirmed_crop,
        "guidance_snapshot": guidance_snapshot,
        "image_b64": img_b64,
        "overlay_b64": overlay_b64,
        "has_images": bool(img_b64 is not None),
    }

    # Prepend newest scans first
    st.session_state["session_scans"].insert(0, scan_record)
    return scan_record


def get_session_scans(
    crop_filter: str = "All",
    status_filter: str = "All",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Retrieve filtered scan records from in-memory session state."""
    init_session_history()
    scans = st.session_state.get("session_scans", [])

    filtered = []
    for s in scans:
        if crop_filter != "All" and s.get("crop") != crop_filter:
            continue
        if status_filter == "Healthy" and not s.get("is_healthy"):
            continue
        if status_filter == "Diseased" and s.get("is_healthy"):
            continue
        filtered.append(s)

    return filtered[:limit]


def get_session_scan_by_id(scan_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an exact scan entry by unique scan ID."""
    init_session_history()
    for s in st.session_state.get("session_scans", []):
        if s.get("id") == scan_id:
            return s
    return None


def clear_session_history() -> None:
    """Clear all in-memory scan history and active report data."""
    st.session_state["session_scans"] = []
    st.session_state["active_report_scan_id"] = None


def get_session_scan_stats() -> Dict[str, Any]:
    """Calculate summary statistics for scans logged in the current session."""
    init_session_history()
    scans = st.session_state.get("session_scans", [])
    total = len(scans)
    healthy = sum(1 for s in scans if s.get("is_healthy"))
    diseased = total - healthy
    avg_conf = (
        sum(s.get("confidence", 0.0) for s in scans) / total if total > 0 else 0.0
    )

    return {
        "total_scans": total,
        "healthy_scans": healthy,
        "diseased_scans": diseased,
        "average_confidence": avg_conf,
    }
