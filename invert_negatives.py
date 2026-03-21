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
    """Kanalgetrennte Histogrammspreizung mit Clipping.

    Schneidet die dunkelsten und hellsten `clip_percent` % der Pixel ab
    und spreizt die verbleibenden Werte linear auf das volle Spektrum.
    Dies neutralisiert die Orangemaske des Farbnegativs.

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

    # Untere Schwelle: erstes Bin, dessen kumulative Summe > clip_count
    low = np.searchsorted(cumsum, clip_count)
    # Obere Schwelle: erstes Bin, dessen kumulative Summe > total - clip_count
    high = np.searchsorted(cumsum, total_pixels - clip_count)

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


def process_negative(
    img: np.ndarray,
    clip_percent: float,
    gamma: float,
    temperature: float = 0.0,
    tint: float = 0.0,
    gamma_r: float = 1.0,
    gamma_g: float = 1.0,
    gamma_b: float = 1.0,
) -> np.ndarray:
    """Vollständige Verarbeitungspipeline für ein einzelnes Farbnegativ.

    1. Invertierung
    2. Kanalgetrennte Histogrammnormalisierung (Orangemaske-Neutralisierung)
    3. Gamma-Korrektur (global)
    4. Chromatische Korrektur: Weißabgleich (Temperatur/Tönung)
    5. Chromatische Korrektur: Kanalgetrennte Gamma-Anpassung (RGB-Kurven)

    Args:
        img: Rohscan des Farbnegativs (BGR, uint8 oder uint16).
        clip_percent: Clipping-Prozentsatz für die Normalisierung.
        gamma: Globaler Gamma-Korrekturwert.
        temperature: Weißabgleich Temperatur (-100 bis +100).
        tint: Weißabgleich Tönung (-100 bis +100).
        gamma_r: Gamma für den Rot-Kanal (>1 = heller/wärmer).
        gamma_g: Gamma für den Grün-Kanal.
        gamma_b: Gamma für den Blau-Kanal (<1 = gelber, entfernt Blaustich).

    Returns:
        Verarbeitetes Positivbild.
    """
    # Schritt 1: Invertierung
    inverted = invert(img)

    # Schritt 2: Kanalgetrennte Normalisierung
    channels = cv2.split(inverted)  # B, G, R
    normalized = [normalize_channel(ch, clip_percent) for ch in channels]
    merged = cv2.merge(normalized)

    # Schritt 3: Globale Gamma-Korrektur
    corrected = apply_gamma(merged, gamma)

    # Schritt 4: Weißabgleich
    corrected = apply_white_balance(corrected, temperature, tint)

    # Schritt 5: Kanalgetrennte Gamma-Korrektur (RGB-Kurven)
    if gamma_r != 1.0 or gamma_g != 1.0 or gamma_b != 1.0:
        b, g, r = cv2.split(corrected)
        r = apply_channel_gamma(r, gamma_r)
        g = apply_channel_gamma(g, gamma_g)
        b = apply_channel_gamma(b, gamma_b)
        corrected = cv2.merge([b, g, r])

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
