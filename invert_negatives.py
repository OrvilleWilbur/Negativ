"""
Stapelverarbeitung und Invertierung analoger Farbnegativ-Scans.

Neutralisiert die Orangemaske (D-min) von Farbnegativfilmen (z. B. Kodak Portra)
durch kanalgetrennte Histogrammnormalisierung nach der Invertierung.
Unterstützt verlustfreie 16-Bit-TIFF-Verarbeitung.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Konfigurierbare Konstanten
# ---------------------------------------------------------------------------
CLIP_PERCENT: float = 0.1        # Prozent der Pixel, die oben/unten abgeschnitten werden
GAMMA: float = 1.2               # Gamma-Korrektur für mittlere Tonwerte
SUPPORTED_EXTENSIONS: set[str] = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Hard-Clamps für die kanalgetrennte Spreizung. Verhindern, dass eine
# extreme Histogramm-Verteilung in einem Kanal (z. B. Schnee-dominierter
# Rotkanal) zu einer aggressiven Spreizung führt, die Tiefenzeichnung
# verbrennt oder Highlights ausreißt.
NORMALIZE_BP_MAX: float = 0.05   # Per-Kanal-Schwarzpunkt darf 5% nie überschreiten
NORMALIZE_WP_MIN: float = 0.80   # Per-Kanal-Weißpunkt darf 80% nie unterschreiten


# ---------------------------------------------------------------------------
# Kernfunktionen
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Liest ein Bild im Originalformat (8- oder 16-Bit) ein.

    Args:
        path: Pfad zur Bilddatei.

    Returns:
        BGR-Bildmatrix als numpy-Array.

    Raises:
        FileNotFoundError: Datei existiert nicht.
        ValueError: Datei konnte nicht dekodiert werden.
    """
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Datei konnte nicht gelesen werden: {path}")

    # Graustufenbilder in 3-Kanal-BGR konvertieren
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 4-Kanal-Bilder (BGRA) → 3 Kanäle
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def invert(img: np.ndarray) -> np.ndarray:
    """Mathematische Invertierung: Wert = Max - Pixel.

    Args:
        img: Eingabebild (uint8 oder uint16).

    Returns:
        Invertiertes Bild gleichen Typs.
    """
    max_val = np.iinfo(img.dtype).max
    return max_val - img


def normalize_channel(channel: np.ndarray, clip_percent: float) -> np.ndarray:
    """Kanalgetrennte Histogrammspreizung mit Clipping und Hard-Clamps.

    Berechnet pro Kanal unabhängig die Perzentile für Schwarz- und Weißpunkt
    (clip_percent % an jedem Ende) und spreizt die verbleibenden Werte linear
    auf das volle Spektrum. Dies eliminiert die spezifische Dichte der
    orangefarbenen Filmmaske physikalisch korrekt — jeder Kanal erhält sein
    eigenes Min/Max-Mapping.

    Zusätzlich werden harte Sicherheitsgrenzen erzwungen
    (`NORMALIZE_BP_MAX`, `NORMALIZE_WP_MIN`): selbst wenn ein Kanal ein
    extrem asymmetrisches Histogramm hat (z. B. Schnee-dominierter Rotkanal
    mit 0,5 %-Perzentil bei 40 % Helligkeit), werden Tiefen und Lichter
    erhalten, weil das Mapping nicht aggressiver als
    [BP_MAX, WP_MIN] → [0, max] werden darf.

    Args:
        channel: Einzelner Farbkanal (2D, uint8 oder uint16).
        clip_percent: Prozentsatz der Pixel, der an beiden Enden abgeschnitten wird.

    Returns:
        Normalisierter Kanal gleichen Typs.
    """
    max_val = np.iinfo(channel.dtype).max
    total_pixels = channel.size

    # Histogramm berechnen
    hist = cv2.calcHist([channel], [0], None, [max_val + 1], [0, max_val + 1])
    hist = hist.flatten()

    # Kumulative Summe für Perzentil-Berechnung
    cumsum = np.cumsum(hist)
    clip_count = total_pixels * (clip_percent / 100.0)

    # Untere/obere Schwelle aus Perzentilen
    low = int(np.searchsorted(cumsum, clip_count))
    high = int(np.searchsorted(cumsum, total_pixels - clip_count))

    # Hard-Clamps: Per-Kanal-BP nie über 5 %, Per-Kanal-WP nie unter 80 %.
    # Wirkt nur, wenn das Perzentil außerhalb des sicheren Bereichs liegt
    # — bei normalen Bildern bleibt das Mapping unverändert.
    low = min(low, int(max_val * NORMALIZE_BP_MAX))
    high = max(high, int(max_val * NORMALIZE_WP_MIN))

    # Sicherheitsgrenzen
    if low >= high:
        low, high = 0, max_val

    # Lineare Spreizung auf [0, max_val]
    result = channel.astype(np.float64)
    result = (result - low) / (high - low) * max_val
    np.clip(result, 0, max_val, out=result)

    return result.astype(channel.dtype)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Wendet eine Gamma-Korrektur an.

    Args:
        img: Eingabebild (uint8 oder uint16).
        gamma: Gamma-Wert. >1 hellt Mitteltöne auf, <1 dunkelt ab.

    Returns:
        Gamma-korrigiertes Bild gleichen Typs.
    """
    max_val = np.iinfo(img.dtype).max
    inv_gamma = 1.0 / gamma

    # Normalisieren → Gamma → Zurückskalieren
    result = img.astype(np.float64) / max_val
    np.power(result, inv_gamma, out=result)
    result *= max_val
    np.clip(result, 0, max_val, out=result)

    return result.astype(img.dtype)


def apply_channel_gamma(
    channel: np.ndarray, gamma: float
) -> np.ndarray:
    """Wendet Gamma-Korrektur auf einen einzelnen Kanal an.

    Args:
        channel: Einzelner Farbkanal (2D, uint8 oder uint16).
        gamma: Gamma-Wert für diesen Kanal.

    Returns:
        Gamma-korrigierter Kanal gleichen Typs.
    """
    if gamma == 1.0:
        return channel
    max_val = np.iinfo(channel.dtype).max
    inv_gamma = 1.0 / gamma
    result = channel.astype(np.float64) / max_val
    np.power(result, inv_gamma, out=result)
    result *= max_val
    np.clip(result, 0, max_val, out=result)
    return result.astype(channel.dtype)


def apply_white_balance(
    img: np.ndarray, temperature: float, tint: float
) -> np.ndarray:
    """Weißabgleich über Temperatur und Tönung.

    Temperatur verschiebt die Blau-Gelb-Balance (Rot+Grün vs. Blau),
    Tönung verschiebt die Grün-Magenta-Balance.

    Args:
        img: BGR-Bild (uint8 oder uint16).
        temperature: -100 (kühler/blauer) bis +100 (wärmer/gelber). 0 = neutral.
        tint: -100 (grüner) bis +100 (magenta). 0 = neutral.

    Returns:
        Farbangepasstes Bild gleichen Typs.
    """
    if temperature == 0.0 and tint == 0.0:
        return img

    max_val = np.iinfo(img.dtype).max
    result = img.astype(np.float64)

    b, g, r = cv2.split(result)

    # Temperatur: positiv = wärmer (Rot hoch, Blau runter)
    # Skalierung: ±100 → ±20% Anpassung
    temp_factor = temperature / 500.0
    r *= (1.0 + temp_factor)
    b *= (1.0 - temp_factor)

    # Tönung: positiv = magenta (Grün runter), negativ = grüner (Grün hoch)
    tint_factor = tint / 500.0
    g *= (1.0 - tint_factor)

    np.clip(r, 0, max_val, out=r)
    np.clip(g, 0, max_val, out=g)
    np.clip(b, 0, max_val, out=b)

    return cv2.merge([b, g, r]).astype(img.dtype)


def apply_input_levels(
    img: np.ndarray, black_point: float, white_point: float
) -> np.ndarray:
    """Input Levels: Setzt Schwarz- und Weißpunkt manuell.

    Alle Werte unterhalb des Schwarzpunkts werden schwarz,
    alle oberhalb des Weißpunkts werden weiß.
    Der Bereich dazwischen wird linear auf [0, max] gespreizt.

    Args:
        img: Eingabebild (uint8 oder uint16).
        black_point: Schwarzpunkt als Prozent des Maximalwerts (0-100).
        white_point: Weißpunkt als Prozent des Maximalwerts (0-100).

    Returns:
        Level-korrigiertes Bild gleichen Typs.
    """
    if black_point == 0.0 and white_point == 100.0:
        return img

    max_val = np.iinfo(img.dtype).max
    low = max_val * (black_point / 100.0)
    high = max_val * (white_point / 100.0)

    if low >= high:
        return img

    result = img.astype(np.float64)
    result = (result - low) / (high - low) * max_val
    np.clip(result, 0, max_val, out=result)

    return result.astype(img.dtype)


def apply_brightness_contrast(
    img: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    """Helligkeit und Kontrast anpassen.

    Args:
        img: Eingabebild (uint8 oder uint16).
        brightness: -100 (dunkler) bis +100 (heller). 0 = neutral.
        contrast: -100 (flacher) bis +100 (steiler). 0 = neutral.

    Returns:
        Angepasstes Bild gleichen Typs.
    """
    if brightness == 0.0 and contrast == 0.0:
        return img

    max_val = np.iinfo(img.dtype).max
    result = img.astype(np.float64)

    # Kontrast: Skalierung um den Mittelpunkt (max_val / 2)
    # contrast -100..+100 → Faktor 0.0..3.0
    if contrast >= 0:
        factor = 1.0 + contrast / 50.0   # 0 → 1.0, 100 → 3.0
    else:
        factor = 1.0 + contrast / 100.0  # -100 → 0.0, 0 → 1.0

    mid = max_val / 2.0
    result = (result - mid) * factor + mid

    # Helligkeit: Linearer Offset
    # brightness -100..+100 → ±30% des Maximalwerts
    offset = (brightness / 100.0) * max_val * 0.3
    result += offset

    np.clip(result, 0, max_val, out=result)
    return result.astype(img.dtype)


def apply_shadow_highlight(
    img: np.ndarray, shadows: float, highlights: float
) -> np.ndarray:
    """Schatten anheben und Highlights komprimieren.

    Verwendet eine selektive Tonkurvenkorrektur:
    - Schatten: Hebt dunkle Töne an, ohne helle zu beeinflussen.
    - Highlights: Komprimiert helle Töne, ohne dunkle zu beeinflussen.

    Args:
        img: Eingabebild (uint8 oder uint16).
        shadows: -100 (Schatten abdunkeln) bis +100 (Schatten anheben). 0 = neutral.
        highlights: -100 (Highlights komprimieren) bis +100 (Highlights aufhellen). 0 = neutral.

    Returns:
        Tonwert-korrigiertes Bild gleichen Typs.
    """
    if shadows == 0.0 and highlights == 0.0:
        return img

    max_val = np.iinfo(img.dtype).max
    result = img.astype(np.float64) / max_val  # Normalisieren auf [0, 1]

    # Schatten: Wirkt auf dunkle Bereiche (gewichtet mit (1-x)²)
    if shadows != 0.0:
        shadow_strength = shadows / 100.0 * 0.4  # Maximal ±40% Anhebung
        shadow_mask = (1.0 - result) ** 2  # Stärkste Wirkung bei Schwarz
        result += shadow_mask * shadow_strength

    # Highlights: Wirkt auf helle Bereiche (gewichtet mit x²)
    if highlights != 0.0:
        highlight_strength = highlights / 100.0 * 0.4
        highlight_mask = result ** 2  # Stärkste Wirkung bei Weiß
        result += highlight_mask * highlight_strength

    np.clip(result, 0.0, 1.0, out=result)
    result *= max_val

    return result.astype(img.dtype)


def apply_rotation(img: np.ndarray, rotation: int) -> np.ndarray:
    """Rotiert das Bild im Uhrzeigersinn um ``rotation * 90``°.

    Args:
        img: Eingabebild.
        rotation: Anzahl der 90°-Schritte im Uhrzeigersinn (0–3 oder beliebig;
            wird modulo 4 genommen).

    Returns:
        Rotiertes Bild. Bei ``rotation == 0`` wird das Original zurückgegeben.
    """
    rotation = int(rotation) % 4
    if rotation == 0:
        return img
    if rotation == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def apply_crop(
    img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """Schneidet das Bild auf einen Rechteck-Bereich zu.

    Koordinaten als Bruchteil der Bildgröße (0.0 - 1.0). Der Crop wird VOR
    der Histogramm-Normalisierung angewendet, damit die Orangemaske nur auf
    dem gewählten Bildausschnitt analysiert wird (z. B. ohne Filmrand).

    Args:
        img: Eingabebild.
        x1: Linke Kante als Anteil (0.0 = links, 1.0 = rechts).
        y1: Obere Kante als Anteil.
        x2: Rechte Kante als Anteil.
        y2: Untere Kante als Anteil.

    Returns:
        Zugeschnittenes Bild (Original-Referenz wenn kein Crop nötig).
    """
    # Kein Crop nötig
    if x1 <= 0.0 and y1 <= 0.0 and x2 >= 1.0 and y2 >= 1.0:
        return img

    h, w = img.shape[:2]
    x1_p = max(0, min(w - 1, int(round(x1 * w))))
    y1_p = max(0, min(h - 1, int(round(y1 * h))))
    x2_p = max(x1_p + 1, min(w, int(round(x2 * w))))
    y2_p = max(y1_p + 1, min(h, int(round(y2 * h))))

    return img[y1_p:y2_p, x1_p:x2_p]


def process_negative(
    img: np.ndarray,
    clip_percent: float,
    gamma: float,
    temperature: float = 0.0,
    tint: float = 0.0,
    gamma_r: float = 1.0,
    gamma_g: float = 1.0,
    gamma_b: float = 1.0,
    black_point: float = 0.0,
    white_point: float = 100.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
    crop_x1: float = 0.0,
    crop_y1: float = 0.0,
    crop_x2: float = 1.0,
    crop_y2: float = 1.0,
    rotation: int = 0,
) -> np.ndarray:
    """Vollständige Verarbeitungspipeline für ein einzelnes Farbnegativ.

    0a. Rotation (vor Crop, damit Crop-Koordinaten zur sichtbaren Orientierung passen)
    0b. Cropping (vor allen Farbberechnungen, damit Histogramm nur den Ausschnitt sieht)
    1. Invertierung
    2. Kanalgetrennte Histogrammnormalisierung (Orangemaske-Neutralisierung)
    3. Input Levels (Schwarz-/Weißpunkt)
    4. Gamma-Korrektur (global)
    5. Chromatische Korrektur: Weißabgleich (Temperatur/Tönung)
    6. Chromatische Korrektur: Kanalgetrennte Gamma-Anpassung (RGB-Kurven)
    7. Helligkeit / Kontrast
    8. Schatten / Highlights

    Args:
        img: Rohscan des Farbnegativs (BGR, uint8 oder uint16).
        clip_percent: Clipping-Prozentsatz für die Normalisierung.
        gamma: Globaler Gamma-Korrekturwert.
        temperature: Weißabgleich Temperatur (-100 bis +100).
        tint: Weißabgleich Tönung (-100 bis +100).
        gamma_r: Gamma für den Rot-Kanal (>1 = heller/wärmer).
        gamma_g: Gamma für den Grün-Kanal.
        gamma_b: Gamma für den Blau-Kanal (<1 = gelber, entfernt Blaustich).
        black_point: Schwarzpunkt als Prozent (0-100).
        white_point: Weißpunkt als Prozent (0-100).
        brightness: Helligkeit (-100 bis +100).
        contrast: Kontrast (-100 bis +100).
        shadows: Schatten anheben/abdunkeln (-100 bis +100).
        highlights: Highlights komprimieren/aufhellen (-100 bis +100).
        crop_x1, crop_y1, crop_x2, crop_y2: Crop-Bereich als Anteile (0.0-1.0).
        rotation: Anzahl 90°-Drehungen im Uhrzeigersinn (0–3).

    Returns:
        Verarbeitetes Positivbild.
    """
    # Schritt 0a: Rotation (vor Crop)
    img = apply_rotation(img, rotation)

    # Schritt 0b: Cropping (Koordinaten beziehen sich auf das gedrehte Bild)
    img = apply_crop(img, crop_x1, crop_y1, crop_x2, crop_y2)

    # Schritt 1: Invertierung
    inverted = invert(img)

    # Schritt 2: Kanalgetrennte Normalisierung
    channels = cv2.split(inverted)  # B, G, R
    normalized = [normalize_channel(ch, clip_percent) for ch in channels]
    merged = cv2.merge(normalized)

    # Schritt 3: Input Levels (Schwarz-/Weißpunkt)
    corrected = apply_input_levels(merged, black_point, white_point)

    # Schritt 4: Globale Gamma-Korrektur
    corrected = apply_gamma(corrected, gamma)

    # Schritt 5: Weißabgleich
    corrected = apply_white_balance(corrected, temperature, tint)

    # Schritt 6: Kanalgetrennte Gamma-Korrektur (RGB-Kurven)
    if gamma_r != 1.0 or gamma_g != 1.0 or gamma_b != 1.0:
        b, g, r = cv2.split(corrected)
        r = apply_channel_gamma(r, gamma_r)
        g = apply_channel_gamma(g, gamma_g)
        b = apply_channel_gamma(b, gamma_b)
        corrected = cv2.merge([b, g, r])

    # Schritt 7: Helligkeit / Kontrast
    corrected = apply_brightness_contrast(corrected, brightness, contrast)

    # Schritt 8: Schatten / Highlights
    corrected = apply_shadow_highlight(corrected, shadows, highlights)

    return corrected


def batch_process(
    input_dir: Path,
    output_dir: Path,
    clip_percent: float = CLIP_PERCENT,
    gamma: float = GAMMA,
) -> None:
    """Durchsucht einen Ordner und verarbeitet alle unterstützten Bilddateien.

    Args:
        input_dir: Verzeichnis mit den Negativ-Scans.
        output_dir: Zielverzeichnis für die konvertierten Positive.
        clip_percent: Clipping-Prozentsatz für die Normalisierung.
        gamma: Gamma-Korrekturwert.
    """
    if not input_dir.is_dir():
        print(f"Fehler: Eingabeverzeichnis existiert nicht: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Alle unterstützten Dateien sammeln
    files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"Keine unterstützten Bilddateien in {input_dir} gefunden.")
        print(f"Unterstützte Formate: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"Verarbeite {len(files)} Dateien  |  Clipping: {clip_percent}%  |  Gamma: {gamma}")

    errors: list[str] = []

    for filepath in tqdm(files, desc="Invertierung", unit="Bild"):
        try:
            img = load_image(filepath)
            result = process_negative(img, clip_percent, gamma)

            # Ausgabepfad: TIFF-Eingaben → TIFF, alles andere → PNG (verlustfrei)
            if filepath.suffix.lower() in {".tif", ".tiff"}:
                out_name = filepath.stem + "_pos.tif"
            else:
                out_name = filepath.stem + "_pos.png"

            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), result)

        except Exception as e:
            errors.append(f"{filepath.name}: {e}")

    if errors:
        print(f"\n{len(errors)} Fehler aufgetreten:")
        for err in errors:
            print(f"  - {err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invertiert analoge Farbnegativ-Scans und neutralisiert die Orangemaske.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Verzeichnis mit den Negativ-Scans",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Ausgabeverzeichnis (Standard: <input_dir>_positive)",
    )
    parser.add_argument(
        "-c", "--clip",
        type=float,
        default=CLIP_PERCENT,
        help=f"Clipping-Prozentsatz pro Kanal (Standard: {CLIP_PERCENT})",
    )
    parser.add_argument(
        "-g", "--gamma",
        type=float,
        default=GAMMA,
        help=f"Gamma-Korrektur (Standard: {GAMMA})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(str(args.input_dir) + "_positive")
    batch_process(args.input_dir, output_dir, args.clip, args.gamma)


if __name__ == "__main__":
    main()
