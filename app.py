"""
Flask-Web-App für die Invertierung analoger Farbnegativ-Scans.

Upload per Drag & Drop, Echtzeit-Vorschau, einstellbare Parameter,
Auto-Korrektur per Histogramm-Analyse, Download der konvertierten Positive.
"""

import io
import time
import uuid
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

from invert_negatives import process_negative

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".heic", ".heif"}

# ---------------------------------------------------------------------------
# In-Memory Bild-Cache für Echtzeit-Vorschau
# ---------------------------------------------------------------------------
# Speichert: session_id -> {"full": np.ndarray, "thumb": np.ndarray, "ts": float}
_image_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
CACHE_MAX_AGE = 600  # 10 Minuten
CACHE_MAX_ITEMS = 20
THUMB_MAX_DIM = 800  # Thumbnail für schnelle Vorschau


def _cleanup_cache() -> None:
    """Entfernt abgelaufene Einträge aus dem Cache."""
    now = time.time()
    expired = [k for k, v in _image_cache.items() if now - v["ts"] > CACHE_MAX_AGE]
    for k in expired:
        del _image_cache[k]
    # Wenn immer noch zu voll, älteste entfernen
    while len(_image_cache) > CACHE_MAX_ITEMS:
        oldest = min(_image_cache, key=lambda k: _image_cache[k]["ts"])
        del _image_cache[oldest]


def _make_thumbnail(img: np.ndarray) -> np.ndarray:
    """Erstellt ein Thumbnail für schnelle Vorschau-Verarbeitung."""
    h, w = img.shape[:2]
    if max(h, w) <= THUMB_MAX_DIM:
        return img.copy()
    scale = THUMB_MAX_DIM / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Upload-Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_upload(file_storage) -> tuple[np.ndarray, str]:
    """Liest eine hochgeladene Datei als BGR numpy-Array ein."""
    filename = file_storage.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Format nicht unterstützt: {suffix}")

    raw = file_storage.read()

    if suffix in {".heic", ".heif"}:
        if not HEIC_SUPPORTED:
            raise ValueError("HEIC-Support nicht installiert (pip install pillow-heif)")
        pil_img = Image.open(io.BytesIO(raw))
        pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, filename

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
        Path(tmp_path).unlink(missing_ok=True)
        if img is None:
            raise ValueError("Bild konnte nicht dekodiert werden")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img, filename


def encode_image(img: np.ndarray, fmt: str, jpeg_quality: int = 92) -> tuple[bytes, str, str]:
    """Kodiert ein BGR-Bild in das gewünschte Format."""
    if fmt == "tiff":
        _, buf = cv2.imencode(".tif", img)
        return buf.tobytes(), "image/tiff", ".tif"
    elif fmt == "jpeg":
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return buf.tobytes(), "image/jpeg", ".jpg"
    else:
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes(), "image/png", ".png"


def _parse_params(form) -> dict:
    """Extrahiert alle Verarbeitungsparameter aus dem Request-Form."""
    return {
        "clip_percent": float(form.get("clip", 0.1)),
        "gamma": float(form.get("gamma", 1.2)),
        "temperature": float(form.get("temperature", 0)),
        "tint": float(form.get("tint", 0)),
        "gamma_r": float(form.get("gamma_r", 1.0)),
        "gamma_g": float(form.get("gamma_g", 1.0)),
        "gamma_b": float(form.get("gamma_b", 1.0)),
        "black_point": float(form.get("black_point", 0)),
        "white_point": float(form.get("white_point", 100)),
        "brightness": float(form.get("brightness", 0)),
        "contrast": float(form.get("contrast", 0)),
        "shadows": float(form.get("shadows", 0)),
        "highlights": float(form.get("highlights", 0)),
    }


# ---------------------------------------------------------------------------
# Auto-Korrektur: Histogramm-basierte Analyse
# ---------------------------------------------------------------------------

def analyze_negative(img: np.ndarray) -> dict:
    """Analysiert ein Farbnegativ und berechnet optimale Korrekturparameter.

    Heuristik:
    1. Invertiere das Bild.
    2. Berechne kanalgetrennte Histogramme.
    3. Bestimme Schwarz-/Weißpunkte aus Perzentilen.
    4. Leite Gamma-Korrektur pro Kanal aus Medianverschiebung ab.
    5. Schätze Temperatur/Tönung aus Kanal-Balance.
    6. Berechne Kontrast/Schatten/Highlights aus Verteilung.

    Returns:
        Dict mit allen Slider-Parametern.
    """
    max_val = np.iinfo(img.dtype).max

    # Schritt 1: Invertieren
    inverted = max_val - img

    # Auf float normalisieren [0, 1]
    inv_f = inverted.astype(np.float64) / max_val
    b, g, r = cv2.split(inv_f)

    # Schritt 2: Perzentile pro Kanal
    def channel_stats(ch):
        p01 = np.percentile(ch, 0.5)
        p99 = np.percentile(ch, 99.5)
        median = np.median(ch)
        std = np.std(ch)
        return p01, p99, median, std

    r_p01, r_p99, r_med, r_std = channel_stats(r)
    g_p01, g_p99, g_med, g_std = channel_stats(g)
    b_p01, b_p99, b_med, b_std = channel_stats(b)

    # Schritt 3: Clipping – aggressiveres Clipping wenn hohe Streuung
    avg_std = (r_std + g_std + b_std) / 3
    clip = round(min(max(avg_std * 2.0, 0.1), 1.5), 2)

    # Schritt 4: Schwarzpunkt / Weißpunkt aus Perzentilen
    avg_p01 = (r_p01 + g_p01 + b_p01) / 3
    avg_p99 = (r_p99 + g_p99 + b_p99) / 3
    black_point = round(max(avg_p01 * 100 - 1.0, 0), 1)
    white_point = round(min(avg_p99 * 100 + 1.0, 100), 1)

    # Schritt 5: Globales Gamma aus Gesamthelligkeit
    overall_median = (r_med + g_med + b_med) / 3
    # Ziel-Median ~0.45 (leicht unter Mitte für natürliches Aussehen)
    target_median = 0.45
    if overall_median > 0.01:
        global_gamma = round(
            np.log(target_median) / np.log(max(overall_median, 0.01)),
            2,
        )
        global_gamma = max(0.5, min(global_gamma, 2.5))
    else:
        global_gamma = 1.2

    # Schritt 6: Kanal-Gamma aus Median-Differenzen
    # Ziel: Alle Kanäle auf ähnlichen Median bringen (Grau-Welt-Annahme)
    target_ch_median = (r_med + g_med + b_med) / 3
    if target_ch_median < 0.01:
        target_ch_median = 0.3

    def calc_channel_gamma(ch_med, target):
        if ch_med < 0.01:
            return 1.0
        g = np.log(target) / np.log(max(ch_med, 0.01))
        return round(max(0.5, min(g, 2.0)), 2)

    gamma_r = calc_channel_gamma(r_med, target_ch_median)
    gamma_g = calc_channel_gamma(g_med, target_ch_median)
    gamma_b = calc_channel_gamma(b_med, target_ch_median)

    # Schritt 7: Temperatur aus R-B-Balance (nach Gamma-Korrektur)
    r_corrected = r_med ** (1.0 / gamma_r) if gamma_r != 0 else r_med
    b_corrected = b_med ** (1.0 / gamma_b) if gamma_b != 0 else b_med
    rb_diff = r_corrected - b_corrected
    # Positive Differenz = zu warm, negative = zu kalt
    temperature = round(np.clip(rb_diff * -200, -100, 100))

    # Tönung aus G-Balance
    g_corrected = g_med ** (1.0 / gamma_g) if gamma_g != 0 else g_med
    rg_avg = (r_corrected + b_corrected) / 2
    gb_diff = g_corrected - rg_avg
    tint = round(np.clip(gb_diff * -200, -100, 100))

    # Schritt 8: Kontrast aus Standardabweichung
    avg_std_norm = avg_std * 4  # Normalisieren auf ~1
    if avg_std_norm < 0.6:
        contrast = round(min((0.6 - avg_std_norm) * 80, 50))
    elif avg_std_norm > 1.0:
        contrast = round(max((1.0 - avg_std_norm) * 40, -30))
    else:
        contrast = 0

    # Schritt 9: Schatten/Highlights
    # Schatten anheben wenn unteres Quartil sehr dunkel
    lower_q = np.percentile(inv_f, 25)
    shadows = round(np.clip((0.2 - lower_q) * 100, -30, 40))

    # Highlights komprimieren wenn oberes Quartil sehr hell
    upper_q = np.percentile(inv_f, 75)
    highlights = round(np.clip((0.7 - upper_q) * 80, -40, 30))

    # Helligkeit: leichte Anpassung basierend auf Gesamthelligkeit
    brightness = round(np.clip((0.45 - overall_median) * 30, -20, 20))

    return {
        "clip": str(clip),
        "gamma": str(global_gamma),
        "temperature": str(temperature),
        "tint": str(tint),
        "gamma_r": str(gamma_r),
        "gamma_g": str(gamma_g),
        "gamma_b": str(gamma_b),
        "black_point": str(black_point),
        "white_point": str(white_point),
        "brightness": str(brightness),
        "contrast": str(contrast),
        "shadows": str(shadows),
        "highlights": str(highlights),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", heic_supported=HEIC_SUPPORTED)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Lädt ein Bild hoch und speichert es im Cache. Gibt eine Session-ID zurück."""
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei hochgeladen"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Keine Datei ausgewählt"}), 400

    try:
        img, filename = load_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    session_id = str(uuid.uuid4())
    thumb = _make_thumbnail(img)

    with _cache_lock:
        _cleanup_cache()
        _image_cache[session_id] = {
            "full": img,
            "thumb": thumb,
            "filename": filename,
            "ts": time.time(),
        }

    # Original-Thumbnail als Base64 für Anzeige
    thumb_8bit = thumb
    if thumb.dtype == np.uint16:
        thumb_8bit = (thumb / 256).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", thumb_8bit, [cv2.IMWRITE_JPEG_QUALITY, 85])
    import base64
    original_b64 = base64.b64encode(buf.tobytes()).decode()

    return jsonify({
        "session_id": session_id,
        "filename": filename,
        "width": img.shape[1],
        "height": img.shape[0],
        "depth": 16 if img.dtype == np.uint16 else 8,
        "original_preview": f"data:image/jpeg;base64,{original_b64}",
    })


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Echtzeit-Vorschau: Verarbeitet das gecachte Thumbnail mit aktuellen Parametern."""
    session_id = request.form.get("session_id", "")

    with _cache_lock:
        entry = _image_cache.get(session_id)
        if entry:
            entry["ts"] = time.time()  # Refresh

    if not entry:
        return jsonify({"error": "Session abgelaufen – bitte Bild erneut hochladen"}), 404

    params = _parse_params(request.form)
    thumb = entry["thumb"]

    result = process_negative(thumb, **params)

    if result.dtype == np.uint16:
        result = (result / 256).astype(np.uint8)

    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analysiert das gecachte Bild und gibt optimale Parameter zurück."""
    session_id = request.form.get("session_id", "")

    with _cache_lock:
        entry = _image_cache.get(session_id)

    if not entry:
        return jsonify({"error": "Session abgelaufen"}), 404

    # Analyse auf Thumbnail für Geschwindigkeit
    params = analyze_negative(entry["thumb"])
    return jsonify(params)


@app.route("/api/process", methods=["POST"])
def api_process():
    """Vollauflösende Verarbeitung und Download."""
    session_id = request.form.get("session_id", "")
    output_format = request.form.get("format", "png")

    with _cache_lock:
        entry = _image_cache.get(session_id)

    if not entry:
        # Fallback: Datei direkt aus Request
        if "file" not in request.files:
            return jsonify({"error": "Keine Datei und keine Session"}), 400
        file = request.files["file"]
        try:
            img, filename = load_upload(file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    else:
        img = entry["full"]
        filename = entry["filename"]

    params = _parse_params(request.form)
    result = process_negative(img, **params)

    data, mimetype, ext = encode_image(result, output_format)
    stem = Path(filename).stem
    out_name = f"{stem}_positive{ext}"

    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=out_name,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
