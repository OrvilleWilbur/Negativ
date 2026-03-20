"""
Flask-Web-App für die Invertierung analoger Farbnegativ-Scans.

Upload per Drag & Drop, Vorschau, einstellbare Parameter (Gamma, Clipping),
wählbares Ausgabeformat, Download der konvertierten Positive.
"""

import io
import uuid
import tempfile
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


def load_upload(file_storage) -> tuple[np.ndarray, str]:
    """Liest eine hochgeladene Datei als BGR numpy-Array ein.

    Returns:
        (image, original_filename)
    """
    filename = file_storage.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Format nicht unterstützt: {suffix}")

    raw = file_storage.read()

    # HEIC/HEIF über Pillow
    if suffix in {".heic", ".heif"}:
        if not HEIC_SUPPORTED:
            raise ValueError("HEIC-Support nicht installiert (pip install pillow-heif)")
        pil_img = Image.open(io.BytesIO(raw))
        pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, filename

    # Alles andere über OpenCV
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback: TIFF 16-bit über temporäre Datei
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
    """Kodiert ein BGR-Bild in das gewünschte Format.

    Returns:
        (bytes, mimetype, file_extension)
    """
    if fmt == "tiff":
        success, buf = cv2.imencode(".tif", img)
        return buf.tobytes(), "image/tiff", ".tif"
    elif fmt == "jpeg":
        # 16-bit → 8-bit für JPEG
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return buf.tobytes(), "image/jpeg", ".jpg"
    else:  # png
        success, buf = cv2.imencode(".png", img)
        return buf.tobytes(), "image/png", ".png"


@app.route("/")
def index():
    return render_template("index.html", heic_supported=HEIC_SUPPORTED)


@app.route("/api/process", methods=["POST"])
def api_process():
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei hochgeladen"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Keine Datei ausgewählt"}), 400

    clip = float(request.form.get("clip", 0.1))
    gamma = float(request.form.get("gamma", 1.2))
    output_format = request.form.get("format", "png")

    try:
        img, original_name = load_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = process_negative(img, clip, gamma)

    data, mimetype, ext = encode_image(result, output_format)
    stem = Path(original_name).stem
    out_name = f"{stem}_positive{ext}"

    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=out_name,
    )


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Erzeugt eine JPEG-Vorschau (max 1200px) für die Browser-Anzeige."""
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei"}), 400

    file = request.files["file"]
    clip = float(request.form.get("clip", 0.1))
    gamma = float(request.form.get("gamma", 1.2))

    try:
        img, _ = load_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = process_negative(img, clip, gamma)

    # Auf 8-bit konvertieren für Vorschau
    if result.dtype == np.uint16:
        result = (result / 256).astype(np.uint8)

    # Skalieren für schnelle Übertragung
    h, w = result.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
