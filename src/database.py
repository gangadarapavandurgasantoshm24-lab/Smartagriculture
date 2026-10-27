import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.config import DB_PATH, SCAN_UPLOADS_DIR


def get_db_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the scans table and storage directories."""
    SCAN_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                crop TEXT NOT NULL,
                condition TEXT NOT NULL,
                is_healthy INTEGER NOT NULL,
                confidence REAL NOT NULL,
                top_predictions_json TEXT,
                image_rel_path TEXT,
                overlay_rel_path TEXT,
                notes TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON scans(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_crop ON scans(crop);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_healthy ON scans(is_healthy);")
    conn.close()


def save_scan(
    filename: str,
    predicted_class: str,
    crop: str,
    condition: str,
    is_healthy: bool,
    confidence: float,
    top_predictions: List[Dict[str, Any]],
    display_image_rgb: np.ndarray,
    overlay_image_rgb: Optional[np.ndarray] = None,
    notes: str = "",
) -> int:
    """
    Save a diagnostic scan and thumbnail images to disk and SQLite.
    Returns the newly inserted scan id.
    """
    init_db()
    conn = get_db_connection()

    # Generate persistent image file paths
    import time

    unique_id = f"{int(time.time() * 1000)}"
    img_filename = f"scan_{unique_id}_orig.png"
    img_path = SCAN_UPLOADS_DIR / img_filename

    # Save original RGB as PNG
    bgr_orig = cv2.cvtColor(display_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), bgr_orig)

    overlay_filename = None
    if overlay_image_rgb is not None:
        overlay_filename = f"scan_{unique_id}_overlay.png"
        overlay_path = SCAN_UPLOADS_DIR / overlay_filename
        bgr_overlay = cv2.cvtColor(overlay_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), bgr_overlay)

    top_json = json.dumps(top_predictions)

    with conn:
        cursor = conn.execute(
            """
            INSERT INTO scans (
                filename, predicted_class, crop, condition, is_healthy,
                confidence, top_predictions_json, image_rel_path,
                overlay_rel_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                filename,
                predicted_class,
                crop,
                condition,
                1 if is_healthy else 0,
                float(confidence),
                top_json,
                img_filename,
                overlay_filename,
                notes,
            ),
        )
        scan_id = cursor.lastrowid

    conn.close()
    return scan_id


def get_all_scans(
    crop_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch diagnostic scan history with optional filtering.
    """
    init_db()
    conn = get_db_connection()

    query = "SELECT * FROM scans WHERE 1=1"
    params = []

    if crop_filter and crop_filter != "All":
        query += " AND crop = ?"
        params.append(crop_filter)

    if status_filter == "Healthy":
        query += " AND is_healthy = 1"
    elif status_filter == "Diseased":
        query += " AND is_healthy = 0"

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "filename": r["filename"],
                "predicted_class": r["predicted_class"],
                "crop": r["crop"],
                "condition": r["condition"],
                "is_healthy": bool(r["is_healthy"]),
                "confidence": float(r["confidence"]),
                "confidence_percent": float(r["confidence"] * 100.0),
                "top_predictions": json.loads(r["top_predictions_json"])
                if r["top_predictions_json"]
                else [],
                "image_rel_path": r["image_rel_path"],
                "overlay_rel_path": r["overlay_rel_path"],
                "notes": r["notes"] or "",
            }
        )

    conn.close()
    return results


def get_scan_stats() -> Dict[str, Any]:
    """Retrieve aggregate statistics from historical scans."""
    init_db()
    conn = get_db_connection()

    total_scans = conn.execute("SELECT COUNT(*) FROM scans;").fetchone()[0]
    healthy_scans = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE is_healthy = 1;"
    ).fetchone()[0]
    diseased_scans = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE is_healthy = 0;"
    ).fetchone()[0]

    avg_conf = conn.execute("SELECT AVG(confidence) FROM scans;").fetchone()[0] or 0.0

    top_crop_row = conn.execute(
        "SELECT crop, COUNT(*) as c FROM scans GROUP BY crop ORDER BY c DESC LIMIT 1;"
    ).fetchone()
    top_crop = top_crop_row["crop"] if top_crop_row else "None"

    conn.close()

    return {
        "total_scans": total_scans,
        "healthy_scans": healthy_scans,
        "diseased_scans": diseased_scans,
        "average_confidence": float(avg_conf),
        "most_scanned_crop": top_crop,
    }


def delete_scan(scan_id: int) -> bool:
    """Delete a scan record and its associated images."""
    init_db()
    conn = get_db_connection()
    row = conn.execute(
        "SELECT image_rel_path, overlay_rel_path FROM scans WHERE id = ?", (scan_id,)
    ).fetchone()

    if row:
        if row["image_rel_path"]:
            p = SCAN_UPLOADS_DIR / row["image_rel_path"]
            if p.exists():
                p.unlink()
        if row["overlay_rel_path"]:
            p = SCAN_UPLOADS_DIR / row["overlay_rel_path"]
            if p.exists():
                p.unlink()

        with conn:
            conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.close()
        return True

    conn.close()
    return False


def clear_all_scans() -> None:
    """Wipe all scan records and stored upload images."""
    init_db()
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM scans;")

    # Clean directory
    for f in SCAN_UPLOADS_DIR.glob("*.*"):
        try:
            f.unlink()
        except OSError:
            pass

    conn.close()
