# Residual NSS demonstration

This directory demonstrates centre-bias-residualised normalised scanpath saliency (rNSS). The compact example uses participant `A110_002`, video 33974, the Full Video Type condition, correct-action trials, and loops 1-4.

The fixation CSV contains 417 fixation-frame rows representing 23 eligible fixations. The feature archive contains all 138 video frames at 320 x 480 pixels.

## Candidate maps and calculation

The four object-region candidates are Active Object, Active Hand, Contextual Object, and Background. The seven GBVS candidates are Color, DKL Color, Flicker, Intensity, Motion, Orientation, and Contrast.

For each candidate and frame, the scripts:

1. blur ROI masks with sigma 58 px; GBVS maps are not blurred;
2. generate an anisotropic Gaussian centre bias with sigma-x 140 px and sigma-y 74 px;
3. population-z-score the candidate and centre-bias maps;
4. regress candidate activation on centre bias across image pixels;
5. population-z-score the residual map;
6. sample the residual map at the fixation coordinate;
7. average frame samples within each fixation;
8. calculate a fixation-duration-weighted participant summary.

This is a single-participant demonstration and does not reproduce the group mixed-effects analysis.

## Files

- `compute_residual_nss_compact.py`: lightweight workflow using the NPZ archive.
- `compute_residual_nss_raw.py`: original-file workflow using JSON, `objects.xlsx`, and MATLAB v7.3 MAT data.
- `fixations_33974_A110_002.csv`: reduced fixation input.
- `features_33974.npz`: compact feature input.
- `example_output_33974_A110_002.xlsx`: unformatted example output.

## Compact workflow

Open `compute_residual_nss_compact.py` and edit the `USER SETTINGS` block near
the top. Enter the local paths to the downloaded fixation CSV and NPZ feature
archive, and choose a path for the output workbook:

```python
FIXATIONS_PATH = Path(r"C:\path\to\fixations_33974_A110_002.csv")
FEATURES_PATH = Path(r"C:\path\to\features_33974.npz")
OUTPUT_PATH = Path(r"C:\path\to\rnss_33974_A110_002.xlsx")
```

The participant, video, maximum loop, sigmas, and optional validation controls
are defined immediately below those paths. After saving the edited script, run
it from the repository root:

```bash
python "Eye Tracking/demos/compute_residual_nss_compact.py"
```

The script does not accept command-line arguments. All inputs and analysis
settings are deliberately visible at the top of the file, consistently with
the repository's other demonstration scripts.

The NPZ stores the categorical ROI map losslessly. Each full-resolution GBVS map uses per-frame, per-feature min-max uint8 quantisation, with the original minimum and maximum retained as metadata. Because rNSS z-scores each map, affine scale is removed; validation against the original float64 maps found a maximum participant-summary difference of 0.00103.

The compact script is intentionally fixed to video 33974 and validates 138 frames at 320 x 480 pixels.

## Raw workflow for video 33974

Open `compute_residual_nss_raw.py` and edit all five paths in its `USER
SETTINGS` block:

```python
FIXATIONS_PATH = Path(r"C:\path\to\fixations_33974_A110_002.csv")
OBJECTS_JSON_PATH = Path(r"C:\path\to\33974.json")
OBJECT_MAPPING_PATH = Path(r"C:\path\to\objects.xlsx")
SALIENCY_MAT_PATH = Path(r"C:\path\to\gbvs_33974.mat")
OUTPUT_PATH = Path(r"C:\path\to\raw_rnss_33974_A110_002.xlsx")
```

After saving the edited script, run:

```bash
python "Eye Tracking/demos/compute_residual_nss_raw.py"
```

The raw assets are not stored in the Git clone. The raw script streams the segmentation JSON with `ijson`, reads MATLAB v7.3/HDF5 references with `h5py`, and does not require MATLAB. Supply the corresponding `.mat` file.

`objects.xlsx` (mapping of objects to objct categories - Active Hand, Active Object, Contextual Objects) is unnecessary for compact execution because its object-index/category mapping is embedded in the NPZ. It is required by the raw script and should be distributed with the raw OSF assets.

## Another participant or video

The raw script processes one participant and one video per run. It infers frame
count and image dimensions from the supplied MAT file, selects the requested
participant/video from either the reduced fixation schema or
`expanded_frames.csv`, filters to Full Video Type/correct-action rows/loops up
to `MAX_LOOP`, and validates JSON masks and fixation coordinates against the
inferred dimensions.

Edit the same settings block, for example:

```python
FIXATIONS_PATH = Path(r"C:\path\to\expanded_frames.csv")
OBJECTS_JSON_PATH = Path(r"C:\path\to\VIDEO_ID.json")
OBJECT_MAPPING_PATH = Path(r"C:\path\to\objects.xlsx")
SALIENCY_MAT_PATH = Path(r"C:\path\to\gbvs_VIDEO_ID.mat")
OUTPUT_PATH = Path(r"C:\path\to\rnss_PARTICIPANT_ID_VIDEO_ID.xlsx")

PARTICIPANT_ID = "PARTICIPANT_ID"
VIDEO_ID = 12345
MAX_LOOP = 4
```

The default sigmas reproduce the original 320 x 480 analysis. For another
resolution, edit `CENTER_SIGMA_X`, `CENTER_SIGMA_Y`, and `ROI_BLUR_SIGMA` to
resolution-appropriate pixel values.

For a faster structural check without calculating rNSS or writing a workbook,
set `VALIDATE_ONLY = True` and `SKIP_CHECKSUMS = True` in the settings block.

## Output workbook

Both scripts write an intentionally unformatted workbook:

- `feature_summary`: one duration-weighted participant result per feature;
- `fixation_level`: one result per feature and fixation;
- `parameters`: filters, dimensions, sigmas, indexing, aggregation, filenames, versions, and checksums.
