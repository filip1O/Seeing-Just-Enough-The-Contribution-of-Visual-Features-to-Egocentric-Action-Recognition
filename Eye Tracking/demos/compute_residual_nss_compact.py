#!/usr/bin/env python3
"""Compact residual-NSS demo for participant A110_002 and video 33974.

This script reads the compact features_33974.npz archive; it does not need the
original segmentation JSON, objects.xlsx, MATLAB, or original GBVS .mat file.

Before running the script, edit the clearly labelled USER SETTINGS block near
the top of this file. All paths and analysis choices are defined there; this
script deliberately does not accept command-line arguments.

Run from the repository root with:

    python "Eye Tracking/demos/compute_residual_nss_compact.py"

The numbered comments in main mirror the messages printed while the script
runs. Mathematical helper functions are explained where they are used, so a
new user can follow the calculation from input to output.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS — EDIT THESE VALUES BEFORE RUNNING THE SCRIPT
# =============================================================================
# Enter the complete local path to each downloaded input file. The leading r
# makes Windows backslashes safe inside the quoted text. Forward-slash paths
# can be used on macOS and Linux, for example Path("/Users/name/data/file.csv").
FIXATIONS_PATH = Path(r"REPLACE_WITH_PATH_TO/fixations_33974_A110_002.csv")
FEATURES_PATH = Path(r"REPLACE_WITH_PATH_TO/features_33974.npz")

# Choose where the new, deliberately unformatted result workbook will be saved.
OUTPUT_PATH = Path(r"REPLACE_WITH_OUTPUT_PATH/rnss_33974_A110_002.xlsx")

# These values select the participant's eligible fixation rows. The compact
# feature archive itself is specific to video 33974.
PARTICIPANT_ID = "A110_002"
VIDEO_ID = 33974
MAX_LOOP = 4

# These pixel-based parameters reproduce the original analysis.
CENTER_SIGMA_X = 140.0
CENTER_SIGMA_Y = 74.0
ROI_BLUR_SIGMA = 58.0

# Set VALIDATE_ONLY to True to inspect the inputs without calculating rNSS or
# writing a workbook. Set SKIP_CHECKSUMS to True to omit SHA-256 file hashes.
VALIDATE_ONLY = False
SKIP_CHECKSUMS = False
# =============================================================================
# END OF USER SETTINGS — THE ANALYSIS CODE STARTS BELOW
# =============================================================================


# ---------------------------------------------------------------------------
# Feature labels and fixation-table schemas
# ---------------------------------------------------------------------------

# Each ROI pixel is stored as one small integer. These dictionaries translate
# between stored integers and the names written to the result workbook.
ROI_CODE_TO_FEATURE = {
    0: "roi_background",
    1: "roi_active_object",
    2: "roi_active_hand",
    3: "roi_contextual_object",
}
ROI_FEATURE_TO_CODE = {value: key for key, value in ROI_CODE_TO_FEATURE.items()}

# This order controls the order of ROI rows in the output workbook.
ROI_FEATURES = [
    "roi_active_object",
    "roi_active_hand",
    "roi_contextual_object",
    "roi_background",
]

# Checking this order prevents, for example, a motion map from being reported
# under the orientation label.
EXPECTED_SALIENCY_FEATURES = [
    "color",
    "dklcolor",
    "flicker",
    "intensity",
    "motion",
    "orientation",
    "contrast",
]
EXPECTED_FRAME_COUNT = 138
EXPECTED_HEIGHT = 320
EXPECTED_WIDTH = 480
EXPECTED_VIDEO = 33974
# The reduced demonstration table already uses these analysis-ready names.
REDUCED_FIXATION_COLUMNS = [
    "participant_id",
    "video_id",
    "video_type",
    "prime_type",
    "classification_difficulty",
    "action_match",
    "trial",
    "fixation_id",
    "loop",
    "frame",
    "duration_ms",
    "x_px",
    "y_px",
]

# expanded_frames.csv uses the names on the left. Values on the right are the
# common internal names used throughout the calculation.
FULL_FIXATION_COLUMNS = {
    "Subject": "participant_id",
    "video_num": "video_id",
    "video_type": "video_type",
    "prime_type": "prime_type",
    "classification_difficulty": "classification_difficulty",
    "action_match": "action_match",
    "Trial": "trial",
    "FixationCount": "fixation_id",
    "loop": "loop",
    "frame": "frame",
    "Duration": "duration_ms",
    "reref_rescaled_PositionX": "x_px",
    "reref_rescaled_PositionY": "y_px",
}


class DemoInputError(RuntimeError):
    """Raised when an input cannot support the requested rNSS calculation."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a reproducibility checksum without loading a large file into RAM."""
    digest = hashlib.sha256()
    # Chunked reading is especially important for multi-gigabyte raw files.
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    """Fail early when a table omits a column required by the analysis."""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise DemoInputError(f"{label} is missing required columns: {', '.join(missing)}")


def _filter_fixations(
    data: pd.DataFrame,
    participant: str,
    video: int,
    max_loop: int,
) -> pd.DataFrame:
    """Apply participant, video, condition, accuracy, and loop filters."""
    # Normalising case tolerates values such as "Full" and "full".
    text_video_type = data["video_type"].astype("string").str.lower()
    # Each comparison produces one Boolean per row. '&' retains rows that meet
    # every criterion simultaneously.
    mask = (
        data["participant_id"].astype("string").eq(str(participant))
        & pd.to_numeric(data["video_id"], errors="coerce").eq(int(video))
        & text_video_type.eq("full")
        & pd.to_numeric(data["action_match"], errors="coerce").eq(1)
        & pd.to_numeric(data["loop"], errors="coerce").le(int(max_loop))
    )
    return data.loc[mask, REDUCED_FIXATION_COLUMNS].copy()


def read_fixations(
    path: Path,
    participant: str,
    video: int,
    max_loop: int,
) -> tuple[pd.DataFrame, str]:
    """Read either the reduced demo CSV or the full expanded fixation table."""
    if path.suffix.lower() != ".csv":
        raise DemoInputError("Fixation input must be a .csv file.")
    if not path.is_file():
        raise DemoInputError(f"Fixation file does not exist: {path}")

    # Read only the header first so the script can identify the schema cheaply.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(REDUCED_FIXATION_COLUMNS).issubset(header):
        # The reduced file is small enough to read in one operation.
        data = pd.read_csv(path, usecols=REDUCED_FIXATION_COLUMNS)
        data = _filter_fixations(data, participant, video, max_loop)
        schema = "demo_csv"
    elif set(FULL_FIXATION_COLUMNS).issubset(header):
        # Filter the large expanded table chunk by chunk to control memory use.
        selected: list[pd.DataFrame] = []
        usecols = list(FULL_FIXATION_COLUMNS)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
            chunk = chunk.rename(columns=FULL_FIXATION_COLUMNS)
            filtered = _filter_fixations(chunk, participant, video, max_loop)
            if not filtered.empty:
                selected.append(filtered)
        if not selected:
            data = pd.DataFrame(columns=REDUCED_FIXATION_COLUMNS)
        else:
            data = pd.concat(selected, ignore_index=True)
        schema = "expanded_frames"
    else:
        raise DemoInputError(
            "Fixation CSV is neither the reduced demo schema nor the "
            "expanded_frames.csv schema."
        )

    if data.empty:
        raise DemoInputError(
            f"No eligible rows for participant={participant}, video={video}, "
            f"video_type=full, action_match=1, loop<={max_loop}."
        )

    # Explicit conversion catches malformed numeric values before they are
    # used as frame indices, coordinates, durations, or filters.
    numeric = [
        "video_id",
        "action_match",
        "trial",
        "fixation_id",
        "loop",
        "frame",
        "duration_ms",
        "x_px",
        "y_px",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    # Deterministic ordering makes the fixation-level output reproducible.
    data = data.sort_values(["trial", "loop", "frame", "fixation_id"]).reset_index(drop=True)
    return data, schema


def validate_fixations(data: pd.DataFrame, frame_count: int, height: int, width: int) -> None:
    """Check that each selected fixation can index the supplied feature maps."""
    _require_columns(data.columns, REDUCED_FIXATION_COLUMNS, "Fixation data")
    if data[REDUCED_FIXATION_COLUMNS].isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise DemoInputError(f"Fixation data contain missing values in: {', '.join(bad)}")
    if not np.allclose(data["frame"], np.rint(data["frame"])):
        raise DemoInputError("Frame values must be integers using one-based indexing.")
    frames = data["frame"].astype(int)
    if frames.min() < 1 or frames.max() > frame_count:
        raise DemoInputError(
            f"Fixation frames must be in 1..{frame_count}; observed {frames.min()}..{frames.max()}."
        )
    if (data["duration_ms"] <= 0).any():
        raise DemoInputError("Every fixation duration must be positive.")
    if ((data["x_px"] < 1) | (data["x_px"] > width)).any():
        raise DemoInputError(f"Fixation x coordinates must be in 1..{width}.")
    if ((data["y_px"] < 1) | (data["y_px"] > height)).any():
        raise DemoInputError(f"Fixation y coordinates must be in 1..{height}.")


def zscore_map(values: np.ndarray) -> np.ndarray:
    """Population-z-score all pixels in one map (standard deviation ddof=0)."""
    # float64 gives the same calculation for uint8 and floating-point inputs.
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values)
    standard_deviation = np.nanstd(values, ddof=0)
    # A constant map contains no spatial information. Returning zeros avoids
    # division by zero and represents its zero predictive contribution.
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / standard_deviation


def gaussian_center_bias(height: int, width: int, sigma_x: float, sigma_y: float) -> np.ndarray:
    """Create an anisotropic Gaussian centred on the middle of the image."""
    # Broadcasting combines one y column and one x row into a full image grid.
    yy = np.arange(height, dtype=np.float64)[:, None]
    xx = np.arange(width, dtype=np.float64)[None, :]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    return np.exp(
        -(
            ((xx - center_x) ** 2) / (2.0 * sigma_x**2)
            + ((yy - center_y) ** 2) / (2.0 * sigma_y**2)
        )
    )


def residual_nss_map(candidate: np.ndarray, center_bias_z: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Remove linear centre bias from a candidate map and z-score the residual."""
    # Standardising first places every candidate feature on a common scale.
    candidate_z = zscore_map(candidate)
    # Treat each pixel as one regression observation: x is centre bias and y
    # is candidate-feature activation at the same location.
    x = center_bias_z.ravel()
    y = candidate_z.ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.nanvar(x[valid]) <= epsilon:
        return candidate_z
    # Centring both variables means the one-predictor OLS slope does not need
    # a separately fitted intercept.
    x_centered = x[valid] - np.nanmean(x[valid])
    y_centered = y[valid] - np.nanmean(y[valid])
    beta = np.nanmean(x_centered * y_centered) / max(epsilon, np.nanvar(x[valid]))
    # Remove the fitted centre component and standardise what remains.
    return zscore_map(candidate_z - beta * center_bias_z)


def blur_roi(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply a Gaussian blur to a binary ROI mask before rNSS calculation."""
    if sigma_px <= 0:
        return mask.astype(np.float32)
    # Six standard deviations retain nearly all Gaussian mass. OR with 1
    # guarantees the odd kernel size required by OpenCV.
    kernel_size = int(max(3, int(math.ceil(sigma_px * 6)) | 1))
    return cv2.GaussianBlur(
        mask.astype(np.float32),
        (kernel_size, kernel_size),
        sigmaX=sigma_px,
        sigmaY=sigma_px,
        borderType=cv2.BORDER_REPLICATE,
    )


@dataclass(frozen=True)
class Candidate:
    """Name one candidate map and identify it as ROI or saliency."""
    name: str
    feature_type: str


class FeatureSource:
    """Minimal interface implemented by each script's feature loader."""
    frame_count: int
    height: int
    width: int
    candidates: list[Candidate]

    def get_base_map(self, candidate: Candidate, frame_index: int) -> np.ndarray:
        """Return one candidate map; concrete feature sources implement this."""
        raise NotImplementedError

    def close(self) -> None:
        """Release open file handles; sources without handles may do nothing."""
        return None


class CompactFeatureSource(FeatureSource):
    """Load and validate the GitHub-sized feature archive for video 33974."""

    def __init__(self, path: Path) -> None:
        """Open the NPZ and verify all arrays needed by the computation."""
        if path.suffix.lower() != ".npz":
            raise DemoInputError("Compact feature input must be a .npz file.")
        if not path.is_file():
            raise DemoInputError(f"Compact feature file does not exist: {path}")
        # Pickled Python objects are unnecessary here and are disabled so the
        # archive contains only ordinary NumPy arrays and a JSON string.
        self.archive = np.load(path, allow_pickle=False)

        # Naming every required array produces a clear error for an incomplete
        # or accidentally replaced feature archive.
        required = {
            "frame_numbers",
            "roi_category_map",
            "saliency_names",
            "saliency_maps",
            "saliency_min",
            "saliency_max",
            "metadata_json",
        }
        missing = sorted(required - set(self.archive.files))
        if missing:
            raise DemoInputError(f"Compact feature archive is missing: {', '.join(missing)}")

        # ROI data have shape [frame, y, x]. Saliency data have shape
        # [feature, frame, y, x].
        self.frame_numbers = self.archive["frame_numbers"].astype(int)
        self.roi_category_map = self.archive["roi_category_map"]
        self.saliency_names = [str(value) for value in self.archive["saliency_names"].tolist()]
        self.saliency_maps = self.archive["saliency_maps"]
        self.saliency_min = self.archive["saliency_min"]
        self.saliency_max = self.archive["saliency_max"]
        self.metadata = json.loads(str(self.archive["metadata_json"].item()))

        # The following checks verify relationships between arrays before any
        # gaze coordinate is used to index them.
        if self.roi_category_map.ndim != 3:
            raise DemoInputError("roi_category_map must have shape [frame, height, width].")
        self.frame_count, self.height, self.width = self.roi_category_map.shape
        if (self.frame_count, self.height, self.width) != (
            EXPECTED_FRAME_COUNT,
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH,
        ):
            raise DemoInputError(
                "This video-33974 demo expects 138 maps at 320x480 pixels; "
                f"observed {self.frame_count} maps at {self.height}x{self.width}."
            )
        if self.saliency_maps.shape != (
            len(self.saliency_names),
            self.frame_count,
            self.height,
            self.width,
        ):
            raise DemoInputError("saliency_maps has an inconsistent shape.")
        expected_ranges_shape = (len(self.saliency_names), self.frame_count)
        if (
            self.saliency_min.shape != expected_ranges_shape
            or self.saliency_max.shape != expected_ranges_shape
        ):
            raise DemoInputError(
                "saliency_min and saliency_max must have shape "
                f"{expected_ranges_shape}."
            )
        if not np.isfinite(self.saliency_min).all() or not np.isfinite(self.saliency_max).all():
            raise DemoInputError("Compact saliency minima and maxima must all be finite.")
        if np.any(self.saliency_max < self.saliency_min):
            raise DemoInputError("A compact saliency maximum is smaller than its minimum.")
        if not np.array_equal(self.frame_numbers, np.arange(1, self.frame_count + 1)):
            raise DemoInputError("Compact frame numbers must be consecutive and one-based.")
        if self.saliency_names != EXPECTED_SALIENCY_FEATURES:
            raise DemoInputError(
                "Unexpected compact saliency feature order: " + ", ".join(self.saliency_names)
            )
        category_values = set(np.unique(self.roi_category_map).tolist())
        expected_category_values = set(ROI_CODE_TO_FEATURE)
        if category_values != expected_category_values:
            raise DemoInputError(
                "Compact ROI maps must collectively contain codes 0, 1, 2, and 3; "
                f"observed {sorted(category_values)}."
            )
        # A single ordered list lets the same computation loop handle all 11
        # candidate maps.
        self.candidates = [Candidate(name, "roi") for name in ROI_FEATURES] + [
            Candidate(name, "saliency") for name in self.saliency_names
        ]

    def get_base_map(self, candidate: Candidate, frame_index: int) -> np.ndarray:
        """Return one two-dimensional candidate map for one video frame."""
        if candidate.feature_type == "roi":
            # Comparing the categorical image with one code produces the
            # binary mask required by the ROI-blurring step.
            return self.roi_category_map[frame_index] == ROI_FEATURE_TO_CODE[candidate.name]
        feature_index = self.saliency_names.index(candidate.name)
        # The stored uint8 map is used directly. Restoring its saved minimum
        # and maximum would only apply an affine transformation, and the next
        # rNSS step population-z-scores the map, removing that transformation.
        return self.saliency_maps[feature_index, frame_index]

    def close(self) -> None:
        """Close the underlying NumPy archive."""
        self.archive.close()


def compute_rnss(
    fixations: pd.DataFrame,
    features: FeatureSource,
    center_sigma_x: float,
    center_sigma_y: float,
    roi_blur_sigma: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same rNSS calculation to every candidate feature map.

    The comments below deliberately spell out the procedure. They correspond
    to the method described in the README and are intended for readers who are
    new to fixation-map analysis.
    """
    # Substep A: check the selected rows and translate one-based video/eye
    # coordinates into zero-based NumPy indices.
    validate_fixations(fixations, features.frame_count, features.height, features.width)
    data = fixations.copy()
    data["frame_index"] = data["frame"].astype(int) - 1
    data["x_index"] = np.rint(data["x_px"]).astype(int) - 1
    data["y_index"] = np.rint(data["y_px"]).astype(int) - 1

    # Substep B: construct and population-z-score the Gaussian image-centre
    # bias once. It is the same size and shape as every visual feature map.
    center_bias_z = zscore_map(
        gaussian_center_bias(features.height, features.width, center_sigma_x, center_sigma_y)
    )
    # Grouping rows by frame lets each expensive residual map be calculated
    # once and then sampled for every fixation row on that frame.
    frame_groups = {int(frame): group for frame, group in data.groupby("frame_index", sort=True)}
    records: list[dict[str, object]] = []
    for candidate in features.candidates:
        for frame_index, rows in frame_groups.items():
            # Substep C: retrieve one ROI or GBVS map. Manuscript-matched ROI
            # maps are blurred (sigma=58 px by default); GBVS maps are not.
            candidate_map = features.get_base_map(candidate, frame_index)
            if candidate.feature_type == "roi":
                candidate_map = blur_roi(candidate_map, roi_blur_sigma)
            # Substep D: z-score the candidate, regress it on centre bias,
            # and z-score the residual image. This residual is the rNSS map.
            residual_map = residual_nss_map(candidate_map, center_bias_z)

            # Substep E: sample the residual image at every fixation coordinate
            # belonging to this video frame.
            x_indices = rows["x_index"].to_numpy(dtype=int)
            y_indices = rows["y_index"].to_numpy(dtype=int)
            sampled = residual_map[y_indices, x_indices]
            for (_, row), value in zip(rows.iterrows(), sampled, strict=True):
                records.append(
                    {
                        "participant_id": str(row["participant_id"]),
                        "video_id": int(row["video_id"]),
                        "trial": int(row["trial"]),
                        "fixation_id": int(row["fixation_id"]),
                        "duration_ms": float(row["duration_ms"]),
                        "feature_type": candidate.feature_type,
                        "feature": candidate.name,
                        "frame_rnss": float(value),
                    }
                )
    frame_level = pd.DataFrame(records)
    if frame_level.empty:
        raise DemoInputError("No frame-level rNSS values were produced.")

    # Substep F: a fixation can span several video frames. Average its sampled
    # frame values to obtain one rNSS value per fixation and feature.
    fixation_level = (
        frame_level.groupby(
            [
                "participant_id",
                "video_id",
                "trial",
                "fixation_id",
                "feature_type",
                "feature",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            duration_ms=("duration_ms", "first"),
            n_frames=("frame_rnss", "size"),
            fixation_rnss=("frame_rnss", "mean"),
        )
    )
    # Substep G: combine the fixation results using fixation duration as the
    # weight. This produces the participant-level feature_summary sheet.
    summaries: list[dict[str, object]] = []
    for candidate in features.candidates:
        subset = fixation_level.loc[fixation_level["feature"].eq(candidate.name)]
        weights = subset["duration_ms"].to_numpy(dtype=float)
        values = subset["fixation_rnss"].to_numpy(dtype=float)
        # Only finite results with positive durations can contribute to a
        # duration-weighted average.
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            rnss = np.nan
        else:
            rnss = float(np.average(values[valid], weights=weights[valid]))
        summaries.append(
            {
                "participant_id": str(fixations["participant_id"].iloc[0]),
                "video_id": int(fixations["video_id"].iloc[0]),
                "feature_type": candidate.feature_type,
                "feature": candidate.name,
                "rnss": rnss,
                "n_fixations": int(valid.sum()),
                "n_frame_samples": int(
                    frame_level.loc[frame_level["feature"].eq(candidate.name)].shape[0]
                ),
                "duration_weighted": True,
            }
        )
    summary = pd.DataFrame(summaries)
    if not np.isfinite(summary["rnss"]).all():
        raise DemoInputError("One or more summary rNSS values are non-finite.")
    return summary, fixation_level


def _parameter_rows(parameters: dict[str, object]) -> pd.DataFrame:
    """Flatten nested provenance metadata into two readable worksheet columns."""
    rows: list[dict[str, object]] = []

    def append_value(key: str, value: object) -> None:
        # Dot-separated keys preserve the original hierarchy without placing a
        # large opaque dictionary into one Excel cell.
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                append_value(f"{key}.{nested_key}", nested_value)
        elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
            for index, item in enumerate(value):
                append_value(f"{key}[{index}]", item)
        else:
            rows.append(
                {
                    "parameter": key,
                    "value": json.dumps(value) if isinstance(value, list) else value,
                }
            )

    for parameter, parameter_value in parameters.items():
        append_value(parameter, parameter_value)
    return pd.DataFrame(rows)


def write_output_workbook(
    output_path: Path,
    summary: pd.DataFrame,
    fixation_level: pd.DataFrame,
    parameters: dict[str, object],
) -> None:
    """Write values to an intentionally unformatted three-sheet workbook."""
    if output_path.suffix.lower() != ".xlsx":
        raise DemoInputError("Result output must use the .xlsx extension.")
    try:
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError as exc:
        raise DemoInputError("Writing .xlsx output requires openpyxl.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Creating sheets directly with openpyxl avoids pandas' default header
    # styling. No colours, fonts, borders, widths, filters, or number formats
    # are assigned anywhere in this function.
    workbook = Workbook()
    workbook.remove(workbook.active)
    for sheet_name, table in (
        ("feature_summary", summary),
        ("fixation_level", fixation_level),
        ("parameters", _parameter_rows(parameters)),
    ):
        worksheet = workbook.create_sheet(sheet_name)
        for row in dataframe_to_rows(table, index=False, header=True):
            worksheet.append(row)
    workbook.save(output_path)


def print_step(number: int, title: str, explanation: str) -> None:
    """Give a novice user a short progress message before each major action."""
    print(f"\nSTEP {number}: {title}")
    print(explanation)


def main() -> int:
    """Run the compact analysis using only the USER SETTINGS above."""
    # Command-line paths were intentionally removed to match the other demos
    # in this repository. Rejecting extra text prevents an old command from
    # appearing to work while the script actually uses the settings above.
    if len(sys.argv) > 1:
        raise DemoInputError(
            "This script does not accept command-line arguments. "
            "Edit the USER SETTINGS block at the top of the file instead."
        )
    if any(
        "REPLACE_WITH_" in str(path)
        for path in (FIXATIONS_PATH, FEATURES_PATH, OUTPUT_PATH)
    ):
        raise DemoInputError(
            "Edit FIXATIONS_PATH, FEATURES_PATH, and OUTPUT_PATH in the "
            "USER SETTINGS block before running the script."
        )
    if VIDEO_ID != EXPECTED_VIDEO:
        raise DemoInputError(
            f"This demonstration is specific to video {EXPECTED_VIDEO}; "
            f"VIDEO_ID is currently {VIDEO_ID}."
        )
    if MAX_LOOP < 1:
        raise DemoInputError("MAX_LOOP must be at least 1.")
    if CENTER_SIGMA_X <= 0 or CENTER_SIGMA_Y <= 0:
        raise DemoInputError("Centre-bias sigmas must both be positive.")
    if ROI_BLUR_SIGMA < 0:
        raise DemoInputError("ROI_BLUR_SIGMA cannot be negative.")

    # ------------------------------------------------------------------
    # STEP 1 — Read and select the fixation rows used by this demo.
    # The reader accepts either the small GitHub CSV or expanded_frames.csv.
    # ------------------------------------------------------------------
    print_step(
        1,
        "Read fixation data",
        f"Selecting participant {PARTICIPANT_ID}, video {VIDEO_ID}, "
        f"Full Video Type, correct-action rows, and loops 1-{MAX_LOOP}.",
    )
    fixations, fixation_schema = read_fixations(
        FIXATIONS_PATH, PARTICIPANT_ID, VIDEO_ID, MAX_LOOP
    )
    fixation_count = fixations.drop_duplicates(
        ["participant_id", "trial", "fixation_id"]
    ).shape[0]
    print(f"Selected {len(fixations)} frame rows from {fixation_count} fixations.")

    # ------------------------------------------------------------------
    # STEP 2 — Open the compact feature archive and check its structure.
    # ROI categories are lossless. GBVS maps are uint8 quantisations; rNSS
    # z-scoring makes restoration of their stored affine scale unnecessary.
    # ------------------------------------------------------------------
    print_step(
        2,
        "Load compact visual features",
        f"Opening {FEATURES_PATH.name} and checking frame numbers, image size, "
        "four ROI codes, and seven GBVS feature names.",
    )
    source = CompactFeatureSource(FEATURES_PATH)
    try:
        validate_fixations(fixations, source.frame_count, source.height, source.width)
        print(
            f"Validated {source.frame_count} feature frames at "
            f"{source.width}x{source.height} pixels."
        )

        # Checksums make it possible to identify the exact input files later.
        source_checksums: dict[str, str] = {}
        if not SKIP_CHECKSUMS:
            source_checksums["fixations"] = sha256_file(FIXATIONS_PATH)
            source_checksums["features"] = sha256_file(FEATURES_PATH)

        if VALIDATE_ONLY:
            print("Validation completed; no result workbook was written.")
            return 0

        # ------------------------------------------------------------------
        # STEP 3 — Calculate rNSS for all four ROI and seven GBVS candidates.
        # Every candidate uses the same centre-bias regression and sampling
        # functions, so compact and raw results are directly comparable.
        # ------------------------------------------------------------------
        print_step(
            3,
            "Compute residual NSS",
            "Blurring ROI masks, removing Gaussian image-centre bias, sampling "
            "each fixation, and applying fixation-duration weighting.",
        )
        summary, fixation_level = compute_rnss(
            fixations,
            source,
            center_sigma_x=CENTER_SIGMA_X,
            center_sigma_y=CENTER_SIGMA_Y,
            roi_blur_sigma=ROI_BLUR_SIGMA,
        )

        # ------------------------------------------------------------------
        # STEP 4 — Save plain values to three sheets. No colours, fonts,
        # borders, filters, frozen panes, widths, or number styles are added.
        # ------------------------------------------------------------------
        print_step(
            4,
            "Write the result workbook",
            "Saving feature_summary, fixation_level, and parameters as "
            "unformatted worksheet values.",
        )
        parameters = {
            "input_source_format": "compact_npz",
            "fixation_schema": fixation_schema,
            "participant_id": PARTICIPANT_ID,
            "video_id": VIDEO_ID,
            "video_type": "full",
            "action_match": 1,
            "max_loop": MAX_LOOP,
            "frame_rows": len(fixations),
            "unique_fixations": fixation_count,
            "feature_frames": source.frame_count,
            "map_height_px": source.height,
            "map_width_px": source.width,
            "center_bias_sigma_x_px": CENTER_SIGMA_X,
            "center_bias_sigma_y_px": CENTER_SIGMA_Y,
            "roi_blur_sigma_px": ROI_BLUR_SIGMA,
            "zscore_ddof": 0,
            "frame_indexing": "one-based in fixation CSV",
            "coordinate_indexing": "one-based x/y rounded to nearest pixel",
            "fixation_aggregation": "mean across all video frames spanned by a fixation",
            "participant_aggregation": "fixation-duration-weighted mean",
            "candidate_features": [candidate.name for candidate in source.candidates],
            "input_filenames": {
                "fixations": FIXATIONS_PATH.name,
                "features": FEATURES_PATH.name,
            },
            "checksums_sha256": source_checksums,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "opencv_version": cv2.__version__,
            "compact_metadata": source.metadata,
        }
        write_output_workbook(OUTPUT_PATH, summary, fixation_level, parameters)
        print(summary.to_string(index=False))
        print(f"\nFinished. Wrote rNSS workbook: {OUTPUT_PATH}")
        return 0
    finally:
        source.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoInputError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
