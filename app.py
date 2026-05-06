"""
Flask-Web-App für die Invertierung analoger Farbnegativ-Scans.

Upload per Drag & Drop, Echtzeit-Vorschau, einstellbare Parameter,
Auto-Korrektur per Histogramm-Analyse, Download der konvertierten Positive.
"""

import io
import base64
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

from invert_negatives import process_negative, apply_crop

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".heic", ".heif"}

# ---------------------------------------------------------------------------
# In-Memory Bild-Cache für Echtzeit-Vorschau
# ---------------------------------------------------------------------------
# Speichert: session_id -> {"raw_bytes": bytes, "suffix": str, "thumb": np.ndarray, "ts": float}
# Full-Res Bilder werden als komprimierte Bytes gespeichert (viel kleiner als numpy-Array)
_image_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
CACHE_MAX_AGE = 300  # 5 Minuten
CACHE_MAX_BYTES = 200 * 1024 * 1024  # 200 MB Gesamt-Cache-Limit
THUMB_MAX_DIM = 600  # Thumbnail für schnelle Vorschau


def _cache_size_bytes() -> int:
    """Schätzt den Speicherverbrauch des Caches."""
    total = 0
    for v in _image_cache.values():
        total += len(v.get("raw_bytes", b""))
        thumb = v.get("thumb")
        if thumb is not None:
            total += thumb.nbytes
    return total


def _cleanup_cache() -> None:
    """Entfernt abgelaufene Einträge und begrenzt Speicher."""
    now = time.time()
    expired = [k for k, v in _image_cache.items() if now - v["ts"] > CACHE_MAX_AGE]
    for k in expired:
        del _image_cache[k]
    # Speicher-basiert: älteste entfernen bis unter Limit
    while _image_cache and _cache_size_bytes() > CACHE_MAX_BYTES:
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
        "crop_x1": float(form.get("crop_x1", 0.0)),
        "crop_y1": float(form.get("crop_y1", 0.0)),
        "crop_x2": float(form.get("crop_x2", 1.0)),
        "crop_y2": float(form.get("crop_y2", 1.0)),
    }


# ---------------------------------------------------------------------------
# Auto-Korrektur: Histogramm-basierte Analyse
# ---------------------------------------------------------------------------

# Hard-Limits für die Auto-Korrektur — robust gegen asymmetrische Histogramme
# (z. B. Schnee- oder Himmel-dominierte Motive).
_AUTO_BP_MAX = 5.0    # Schwarzpunkt darf 5 % nie überschreiten
_AUTO_WP_MIN = 80.0   # Weißpunkt darf 80 % nie unterschreiten


def analyze_negative(img: np.ndarray) -> dict:
    """Berechnet robuste Auto-Korrektur-Parameter für ein Farbnegativ.

    Architektur:

    1. **Schwarz-/Weißpunkt** strikt perzentilbasiert (0.5 % / 99.5 %) mit
       harten Clamps: BP ∈ [0, 5] %, WP ∈ [80, 100] %. Verhindert, dass
       Motive mit dominanten hellen Flächen (Schnee, Himmel) extreme
       Schwarzpunkte erzeugen und so die Tiefenzeichnung zerstören.

    2. **Median-Luminanz** (Rec. 709 gewichtet) als Helligkeitsmetrik für
       Auto-Exposure und Gamma. Outlier-resistent gegenüber großflächigen
       einfarbigen Bereichen — anders als der arithmetische Mittelwert.

    3. **Parameter-Neutralität**: Schatten, Highlights, Kontrast und
       Helligkeit werden nicht algorithmisch gesetzt, sondern auf 0
       (Neutralwert) zurückgegeben. Die Basiskontrastierung erfolgt
       ausschließlich über Schwarzpunkt, Weißpunkt und globales Gamma.
       Die chromatischen Korrekturen (RGB-Gamma, Temperatur, Tönung)
       bleiben als separate Berechnung erhalten.

    Args:
        img: Negativ-Scan (BGR, uint8 oder uint16). Wenn Crop aktiv ist,
            sollte der Aufrufer das Bild bereits zugeschnitten übergeben,
            damit nur der gewählte Ausschnitt analysiert wird.

    Returns:
        Dict mit allen Slider-Parametern als JSON-serialisierbare Strings.
    """
    max_val = np.iinfo(img.dtype).max

    # Invertieren und auf [0, 1] normalisieren
    inv_f = (max_val - img).astype(np.float64) / max_val
    b, g, r = cv2.split(inv_f)

    # ------------------------------------------------------------------
    # 1. Schwarz-/Weißpunkt aus Perzentilen, mit harten Clamps
    # ------------------------------------------------------------------
    r_p01, r_p99 = np.percentile(r, [0.5, 99.5])
    g_p01, g_p99 = np.percentile(g, [0.5, 99.5])
    b_p01, b_p99 = np.percentile(b, [0.5, 99.5])

    avg_p01 = (r_p01 + g_p01 + b_p01) / 3.0
    avg_p99 = (r_p99 + g_p99 + b_p99) / 3.0

    black_point = float(np.clip(avg_p01 * 100.0, 0.0, _AUTO_BP_MAX))
    white_point = float(np.clip(avg_p99 * 100.0, _AUTO_WP_MIN, 100.0))

    # ------------------------------------------------------------------
    # 2. Globales Gamma aus Median-Luminanz (Rec. 709)
    # ------------------------------------------------------------------
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lum_median = float(np.median(luminance))

    target_median = 0.45  # natürlich wirkende Mitteltöne
    # apply_gamma rechnet pixel^(1/gamma). Damit median^(1/gamma) = target gilt:
    #   gamma = log(median) / log(target)
    # gamma > 1 hellt auf, < 1 dunkelt ab — passt zur Slider-Semantik.
    if lum_median > 0.01 and target_median > 0.01:
        global_gamma = float(
            np.clip(np.log(max(lum_median, 0.01)) / np.log(target_median),
                    0.5, 2.5)
        )
    else:
        global_gamma = 1.2

    # ------------------------------------------------------------------
    # 3. Chromatische Korrektur (unverändert: RGB-Gamma, Temperatur, Tönung)
    #    Verwenden weiterhin Kanal-Mediane — robust gegen Outlier
    # ------------------------------------------------------------------
    r_med = float(np.median(r))
    g_med = float(np.median(g))
    b_med = float(np.median(b))

    target_ch = (r_med + g_med + b_med) / 3.0
    if target_ch < 0.05:
        target_ch = 0.3

    def calc_channel_gamma(ch_med: float, target: float) -> float:
        if ch_med < 0.02 or target < 0.02:
            return 1.0
        raw = np.log(max(ch_med, 0.01)) / np.log(max(target, 0.01))
        # 50 % Stärke — natürliche Farbabstufungen erhalten
        raw = 1.0 + (raw - 1.0) * 0.5
        return float(np.clip(raw, 0.7, 1.6))

    gamma_r = calc_channel_gamma(r_med, target_ch)
    gamma_g = calc_channel_gamma(g_med, target_ch)
    gamma_b = calc_channel_gamma(b_med, target_ch)

    # Grün als Anker normalisieren
    if gamma_g > 0:
        gamma_r = float(np.clip(gamma_r / gamma_g, 0.7, 1.6))
        gamma_b = float(np.clip(gamma_b / gamma_g, 0.7, 1.6))
        gamma_g = 1.0

    # Temperatur aus R/B-Median-Verhältnis
    if r_med > 0.01 and b_med > 0.01:
        temperature = int(np.clip((1.0 - r_med / b_med) * 60.0, -40, 60))
    else:
        temperature = 30

    # Tönung aus G vs. (R+B)/2
    rb_avg = (r_med + b_med) / 2.0
    if rb_avg > 0.01:
        tint = int(np.clip((1.0 - g_med / rb_avg) * 40.0, -40, 40))
    else:
        tint = 0

    # ------------------------------------------------------------------
    # 4. Parameter-Neutralität: Schatten/Highlights/Kontrast/Helligkeit = 0
    #    Basiskontrast erfolgt ausschließlich über BP/WP + Gamma.
    # ------------------------------------------------------------------
    return {
        "clip":        "0.1",
        "gamma":       str(round(global_gamma, 2)),
        "temperature": str(temperature),
        "tint":        str(tint),
        "gamma_r":     str(round(gamma_r, 2)),
        "gamma_g":     str(round(gamma_g, 2)),
        "gamma_b":     str(round(gamma_b, 2)),
        "black_point": str(round(black_point, 1)),
        "white_point": str(round(white_point, 1)),
        "brightness":  "0",
        "contrast":    "0",
        "shadows":     "0",
        "highlights":  "0",
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

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    # Rohe Bytes lesen (komprimiert, spart RAM vs. entpacktes numpy-Array)
    raw_bytes = file.read()
    file.seek(0)

    try:
        img, filename = load_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    session_id = str(uuid.uuid4())
    thumb = _make_thumbnail(img)
    width, height = img.shape[1], img.shape[0]
    depth = 16 if img.dtype == np.uint16 else 8

    # Full-Res Array sofort freigeben – nur komprimierte Bytes + Thumb cachen
    del img

    with _cache_lock:
        _cleanup_cache()
        _image_cache[session_id] = {
            "raw_bytes": raw_bytes,
            "suffix": suffix,
            "thumb": thumb,
            "filename": filename,
            "ts": time.time(),
        }

    # Original-Thumbnail als Base64 für Anzeige
    thumb_8bit = thumb
    if thumb.dtype == np.uint16:
        thumb_8bit = (thumb / 256).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", thumb_8bit, [cv2.IMWRITE_JPEG_QUALITY, 85])
    original_b64 = base64.b64encode(buf.tobytes()).decode()

    return jsonify({
        "session_id": session_id,
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
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

    # Crop berücksichtigen: Analyse nur auf dem ausgewählten Bildbereich
    crop_x1 = float(request.form.get("crop_x1", 0.0))
    crop_y1 = float(request.form.get("crop_y1", 0.0))
    crop_x2 = float(request.form.get("crop_x2", 1.0))
    crop_y2 = float(request.form.get("crop_y2", 1.0))

    # Analyse auf Thumbnail für Geschwindigkeit
    try:
        thumb = apply_crop(entry["thumb"], crop_x1, crop_y1, crop_x2, crop_y2)
        params = analyze_negative(thumb)
        return jsonify(params)
    except Exception as e:
        return jsonify({"error": f"Analyse-Fehler: {str(e)}"}), 500


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
        # Full-Res aus gecachten komprimierten Bytes rekonstruieren
        raw_bytes = entry["raw_bytes"]
        suffix = entry["suffix"]
        filename = entry["filename"]
        if suffix in {".heic", ".heif"}:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return jsonify({"error": "Bild konnte nicht dekodiert werden"}), 500
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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


@app.route("/api/status")
def api_status():
    """Gibt Server-Auslastung zurück: RAM, Cache, CPU."""
    import psutil
    proc = psutil.Process()
    mem = proc.memory_info()
    vm = psutil.virtual_memory()

    with _cache_lock:
        cache_items = len(_image_cache)
        cache_bytes = _cache_size_bytes()

    return jsonify({
        "ram_used_mb": round(mem.rss / 1024 / 1024, 1),
        "ram_total_mb": round(vm.total / 1024 / 1024, 1),
        "ram_percent": round(mem.rss / vm.total * 100, 1),
        "cache_items": cache_items,
        "cache_mb": round(cache_bytes / 1024 / 1024, 1),
        "cache_limit_mb": round(CACHE_MAX_BYTES / 1024 / 1024),
        "cpu_percent": proc.cpu_percent(interval=0),
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
