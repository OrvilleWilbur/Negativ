# Negativ – Color Negative Film Inverter

Batch-converts scanned color negatives (e.g. Kodak Portra, Fuji 400H) into positives with proper color correction.

A naive inversion (`Max - Pixel`) produces a heavy blue cast because of the orange base mask (D-min) inherent to color negative film. This tool solves that problem through **per-channel histogram normalization** after inversion.

## How it works

1. **Invert** the negative scan (`Max - Pixel`)
2. **Per-channel normalization** – compute the histogram for each B/G/R channel independently, clip the darkest and brightest 0.1 % of pixels (dust/scratch resilience), then linearly stretch the remaining values to the full range. This removes the orange mask color cast.
3. **Gamma correction** – a moderate gamma lift (default 1.2) adjusts midtones for perceptual brightness.

Supports **16-bit TIFF** files natively to avoid banding artifacts that occur with 8-bit processing.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic – processes all images in ./scans, writes results to ./scans_positive
python invert_negatives.py ./scans

# Custom output directory, clip percentage and gamma
python invert_negatives.py ./scans -o ./positive -c 0.5 -g 1.3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | `<input>_positive` | Output directory |
| `-c, --clip` | `0.1` | Histogram clip percentage per channel |
| `-g, --gamma` | `1.2` | Gamma correction value |

### Supported formats

`.tif` `.tiff` `.png` `.jpg` `.jpeg`

TIFF inputs are saved as TIFF (preserving 16-bit depth), everything else is saved as lossless PNG.

## License

MIT
