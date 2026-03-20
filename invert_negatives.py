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


def process_negative(img: np.ndarray, clip_percent: float, gamma: float) -> np.ndarray:
    """Vollständige Verarbeitungspipeline für ein einzelnes Farbnegativ.

    1. Invertierung
    2. Kanalgetrennte Histogrammnormalisierung (Orangemaske-Neutralisierung)
    3. Gamma-Korrektur

    Args:
        img: Rohscan des Farbnegativs (BGR, uint8 oder uint16).
        clip_percent: Clipping-Prozentsatz für die Normalisierung.
        gamma: Gamma-Korrekturwert.

    Returns:
        Verarbeitetes Positivbild.
    """
    # Schritt 1: Invertierung
    inverted = invert(img)

    # Schritt 2: Kanalgetrennte Normalisierung
    channels = cv2.split(inverted)  # B, G, R
    normalized = [normalize_channel(ch, clip_percent) for ch in channels]
    merged = cv2.merge(normalized)

    # Schritt 3: Gamma-Korrektur
    corrected = apply_gamma(merged, gamma)

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
