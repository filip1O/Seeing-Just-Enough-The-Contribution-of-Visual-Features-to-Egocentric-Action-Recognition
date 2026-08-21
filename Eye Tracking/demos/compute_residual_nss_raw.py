#!/usr/bin/env python3
"""Residual-NSS computation from original raw feature files.

This script reads the original segmentation JSON, objects.xlsx mapping, and
MATLAB v7.3 GBVS data file directly. MATLAB itself is not required: h5py
reads the supplied .mat file.

The raw files are intentionally kept outside the GitHub clone. After they are
downloaded from OSF, enter their local paths in the clearly labelled USER
SETTINGS block near the top of this file. Participant, video, output path, and
analysis choices are defined in the same block; this script deliberately does
not accept command-line arguments. Frame count and map dimensions are inferred
from the supplied MAT file. Numbered comments and progress messages explain
each stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS — EDIT THESE VALUES BEFORE RUNNING THE SCRIPT
# =============================================================================
# Enter the complete local path to every input file.
FIXATIONS_PATH = Path(r"REPLACE_WITH_PATH_TO/fixations_33974_A110_002.csv")
OBJECTS_JSON_PATH = Path(r"REPLACE_WITH_PATH_TO/33974.json")
OBJECT_MAPPING_PATH = Path(r"REPLACE_WITH_PATH_TO/objects.xlsx")
SALIENCY_MAT_PATH = Path(r"REPLACE_WITH_PATH_TO/gbvs_33974.mat")

# Choose where the new, deliberately unformatted result workbook will be saved.
OUTPUT_PATH = Path(r"REPLACE_WITH_OUTPUT_PATH/raw_rnss_33974_A110_002.xlsx")

# These values select one participant and one video from the fixation table.
# To analyse another video, also supply that video's matching JSON, MAT file,
# and a mapping workbook containing the corresponding object-index rows.
PARTICIPANT_ID = "A110_002"
VIDEO_ID = 33974
MAX_LOOP = 4

# These pixel-based parameters reproduce the original analysis.
CENTER_SIGMA_X = 140.0
CENTER_SIGMA_Y = 74.0
ROI_BLUR_SIGMA = 58.0

# Set VALIDATE_ONLY to True to inspect the inputs without calculating rNSS or
# writing a workbook. Hashing multi-gigabyte raw files can be slow, so set
# SKIP_CHECKSUMS to True if checksums are unnecessary for the current run.
VALIDATE_ONLY = False
SKIP_CHECKSUMS = False
# =============================================================================
# END OF USER SETTINGS — THE ANALYSIS CODE STARTS BELOW
# =============================================================================


# ---------------------------------------------------------------------------
# Feature labels and fixation-table schemas
# ---------------------------------------------------------------------------

# Each ROI pixel is represented by one small integer. These dictionaries
# translate between integer codes and the feature names used in the results.
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


def _decode_matlab_string(h5_file: object, reference: object) -> str:
    """Decode one MATLAB v7.3 character array reached through an HDF5 reference."""
    dataset = h5_file[reference]
    # MATLAB stores character codes as numbers and uses column-major ordering.
    values = np.asarray(dataset[()]).ravel(order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0)


class RawFeatureSource(FeatureSource):
    """Read one video's original ROI and GBVS maps without requiring MATLAB."""

    def __init__(
        self,
        objects_json: Path,
        object_mapping: Path,
        saliency_mat: Path,
        video: int,
    ) -> None:
        """Open and cross-check one video's JSON, mapping workbook, and MAT file."""
        # File-extension checks catch the common error of supplying the MATLAB
        # analysis script (.m) instead of its saved feature arrays (.mat).
        if saliency_mat.suffix.lower() == ".m":
            raise DemoInputError(
                "A MATLAB .m file is source code, not the required map data. "
                "Use the original MATLAB v7.3 .mat file."
            )
        if objects_json.suffix.lower() != ".json":
            raise DemoInputError("Segmentation input must be the original .json file.")
        if object_mapping.suffix.lower() != ".xlsx":
            raise DemoInputError("Object mapping input must be the original .xlsx workbook.")
        if saliency_mat.suffix.lower() != ".mat":
            raise DemoInputError("Saliency input must be the original MATLAB v7.3 .mat file.")
        for label, path in (
            ("Segmentation JSON", objects_json),
            ("Object mapping workbook", object_mapping),
            ("Saliency MAT", saliency_mat),
        ):
            if not path.is_file():
                raise DemoInputError(f"{label} does not exist: {path}")

        # h5py is imported only by this raw workflow; compact users do not need
        # it merely to import or ask for help from their script.
        try:
            import h5py
        except ImportError as exc:
            raise DemoInputError(
                "Reading MATLAB v7.3 input requires h5py; install requirements.txt."
            ) from exc

        self.objects_json_path = objects_json
        self.object_mapping_path = object_mapping
        self.video = int(video)
        try:
            self.h5 = h5py.File(saliency_mat, "r")
        except OSError as exc:
            raise DemoInputError(
                "The saliency file is not a readable MATLAB v7.3/HDF5 .mat file."
            ) from exc
        # outred is the MATLAB cell array containing one referenced structure
        # per video frame.
        if "outred" not in self.h5:
            raise DemoInputError("Saliency MAT file does not contain the expected 'outred' variable.")
        self.frame_references = list(np.asarray(self.h5["outred"][()]).ravel(order="F"))
        self.frame_count = len(self.frame_references)
        if self.frame_count == 0:
            raise DemoInputError("Saliency MAT file contains no frames.")

        # The first frame defines feature names and image dimensions. Every
        # later map is checked against these values when it is read.
        first_group = self.h5[self.frame_references[0]]
        if "map_types" not in first_group or "top_level_feat_maps" not in first_group:
            raise DemoInputError("A saliency frame is missing map_types or top_level_feat_maps.")
        self.saliency_names = [
            _decode_matlab_string(self.h5, reference)
            for reference in np.asarray(first_group["map_types"][()]).ravel(order="F")
        ]
        if self.saliency_names != EXPECTED_SALIENCY_FEATURES:
            raise DemoInputError(
                "Unexpected MAT saliency features: " + ", ".join(self.saliency_names)
            )
        first_map_reference = np.asarray(first_group["top_level_feat_maps"][()]).ravel(order="F")[0]
        # HDF5 exposes the MATLAB map as [x, y]; transposing gives conventional
        # image orientation [y, x], or [height, width].
        first_map = np.asarray(self.h5[first_map_reference][()]).T
        if first_map.ndim != 2:
            raise DemoInputError("Saliency maps must be two-dimensional.")
        self.height, self.width = first_map.shape

        # Convert object IDs to the four analysis categories before processing
        # the segmentation masks.
        self.object_index_to_category = self._read_object_mapping()
        self.roi_category_map = self._stream_roi_category_map()
        self.candidates = [Candidate(name, "roi") for name in ROI_FEATURES] + [
            Candidate(name, "saliency") for name in self.saliency_names
        ]

    def _read_object_mapping(self) -> dict[int, str]:
        """Map video-specific JSON object indices to the four ROI features."""
        mapping = pd.read_excel(self.object_mapping_path, sheet_name="objects")
        required = ["video_num", "object_index_video", "object_category_noArm"]
        _require_columns(mapping.columns, required, "Object mapping workbook")
        # objects.xlsx contains many videos; only the requested video's rows
        # are relevant to this JSON file.
        selected = mapping.loc[pd.to_numeric(mapping["video_num"], errors="coerce").eq(self.video)].copy()
        if selected.empty:
            raise DemoInputError(f"Object mapping contains no rows for video {self.video}.")
        index_to_category: dict[int, str] = {}
        # Spreadsheet category labels are translated once here so downstream
        # code uses the same feature names as the output workbook.
        category_to_feature = {
            "background": "roi_background",
            "active_object": "roi_active_object",
            "active_hand": "roi_active_hand",
            "contextual_object": "roi_contextual_object",
        }
        for _, row in selected.iterrows():
            object_index = int(row["object_index_video"])
            category = str(row["object_category_noArm"]).strip().lower().replace(" ", "_")
            if category not in category_to_feature:
                raise DemoInputError(
                    f"Unsupported object category for index {object_index}: {category}"
                )
            feature = category_to_feature[category]
            if object_index in index_to_category and index_to_category[object_index] != feature:
                raise DemoInputError(f"Object index {object_index} maps to multiple ROI categories.")
            index_to_category[object_index] = feature
        return index_to_category

    def _stream_roi_category_map(self) -> np.ndarray:
        """Stream all JSON frames and combine object masks into ROI-code images."""
        try:
            import ijson
        except ImportError as exc:
            raise DemoInputError(
                "Streaming the segmentation JSON requires ijson; install requirements.txt."
            ) from exc

        # The completed array is small compared with the source JSON because
        # every pixel stores just one ROI category code.
        category_map = np.empty((self.frame_count, self.height, self.width), dtype=np.uint8)
        seen: set[int] = set()
        # The top-level JSON key is the selected video number.
        prefix = f"{self.video}.frames"
        with self.objects_json_path.open("rb") as handle:
            try:
                iterator: Iterator[tuple[str, dict[str, object]]] = ijson.kvitems(handle, prefix)
                for frame_key, frame_data in iterator:
                    try:
                        zero_based_frame = int(frame_key)
                    except (TypeError, ValueError) as exc:
                        raise DemoInputError(f"Invalid JSON frame key: {frame_key!r}") from exc
                    if zero_based_frame < 0 or zero_based_frame >= self.frame_count:
                        raise DemoInputError(
                            f"JSON frame {zero_based_frame} is outside 0..{self.frame_count - 1}."
                        )
                    if zero_based_frame in seen:
                        raise DemoInputError(f"JSON frame {zero_based_frame} occurs more than once.")
                    objects = frame_data.get("objects") if isinstance(frame_data, dict) else None
                    if not isinstance(objects, dict) or not objects:
                        raise DemoInputError(f"JSON frame {zero_based_frame} contains no objects.")
                    # 255 is a temporary "unassigned" sentinel. It is not a
                    # valid ROI code, so uncovered pixels are easy to detect.
                    output = np.full((self.height, self.width), 255, dtype=np.uint8)
                    for object_index_text, object_data in objects.items():
                        object_index = int(object_index_text)
                        if object_index not in self.object_index_to_category:
                            raise DemoInputError(
                                f"JSON object index {object_index} has no mapping in objects.xlsx."
                            )
                        if not isinstance(object_data, dict) or "mask" not in object_data:
                            raise DemoInputError(
                                f"Frame {zero_based_frame}, object {object_index} has no mask."
                            )
                        mask = np.asarray(object_data["mask"], dtype=bool)
                        if mask.shape != (self.height, self.width):
                            raise DemoInputError(
                                f"Frame {zero_based_frame}, object {object_index} mask shape "
                                f"{mask.shape} != {(self.height, self.width)}."
                            )
                        # A pixel must belong to exactly one object mask.
                        if np.any(mask & (output != 255)):
                            raise DemoInputError(
                                f"Frame {zero_based_frame} contains overlapping object masks."
                            )
                        feature = self.object_index_to_category[object_index]
                        output[mask] = ROI_FEATURE_TO_CODE[feature]
                    if np.any(output == 255):
                        uncovered = int(np.count_nonzero(output == 255))
                        raise DemoInputError(
                            f"Frame {zero_based_frame} has {uncovered} pixels outside all object masks."
                        )
                    category_map[zero_based_frame] = output
                    seen.add(zero_based_frame)
            except DemoInputError:
                raise
            except Exception as exc:
                raise DemoInputError(f"Could not stream segmentation JSON: {exc}") from exc
        # The MAT file defines how many frames are expected from the JSON.
        expected = set(range(self.frame_count))
        if seen != expected:
            missing = sorted(expected - seen)
            raise DemoInputError(f"Segmentation JSON is missing frames: {missing[:20]}")
        return category_map

    def _saliency_map(self, frame_index: int, feature_index: int) -> np.ndarray:
        """Dereference one GBVS map and return it in [height, width] orientation."""
        frame_group = self.h5[self.frame_references[frame_index]]
        map_references = np.asarray(frame_group["top_level_feat_maps"][()]).ravel(order="F")
        if len(map_references) != len(self.saliency_names):
            raise DemoInputError(f"MAT frame {frame_index + 1} has the wrong feature count.")
        values = np.asarray(self.h5[map_references[feature_index]][()]).T
        if values.shape != (self.height, self.width):
            raise DemoInputError(
                f"MAT frame {frame_index + 1}, feature {self.saliency_names[feature_index]} "
                f"shape {values.shape} != {(self.height, self.width)}."
            )
        return values

    def get_base_map(self, candidate: Candidate, frame_index: int) -> np.ndarray:
        """Return one two-dimensional candidate map for one video frame."""
        if candidate.feature_type == "roi":
            return self.roi_category_map[frame_index] == ROI_FEATURE_TO_CODE[candidate.name]
        feature_index = self.saliency_names.index(candidate.name)
        return self._saliency_map(frame_index, feature_index)

    def close(self) -> None:
        """Close the HDF5 handle used to access MATLAB v7.3 data."""
        self.h5.close()


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
            # Substep C: retrieve one ROI or GBVS map. ROI maps are blurred
            # (sigma=58 px by default); GBVS maps are not. For another image
            # resolution, users can change this pixel-based sigma at the CLI.
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
    """Run the raw-file analysis using only the USER SETTINGS above."""
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
        for path in (
            FIXATIONS_PATH,
            OBJECTS_JSON_PATH,
            OBJECT_MAPPING_PATH,
            SALIENCY_MAT_PATH,
            OUTPUT_PATH,
        )
    ):
        raise DemoInputError(
            "Edit all input and output paths in the USER SETTINGS block "
            "before running the script."
        )
    if MAX_LOOP < 1:
        raise DemoInputError("MAX_LOOP must be at least 1.")
    if CENTER_SIGMA_X <= 0 or CENTER_SIGMA_Y <= 0:
        raise DemoInputError("Centre-bias sigmas must both be positive.")
    if ROI_BLUR_SIGMA < 0:
        raise DemoInputError("ROI_BLUR_SIGMA cannot be negative.")

    # ------------------------------------------------------------------
    # STEP 1 — Read either the small demo CSV or the full expanded table.
    # The same filters are applied in both cases.
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
    # STEP 2 — Open all original feature sources.
    # - objects.xlsx translates JSON object indices into four ROI categories.
    # - The supplied JSON is streamed frame by frame instead of loaded into RAM.
    # - The supplied MAT file is MATLAB v7.3/HDF5 data read without MATLAB.
    # Runtime depends on that video's frame count, dimensions, and file sizes.
    # ------------------------------------------------------------------
    print_step(
        2,
        "Read and validate original visual features",
        f"Loading {OBJECT_MAPPING_PATH.name}, streaming {OBJECTS_JSON_PATH.name}, "
        f"and opening seven GBVS maps in {SALIENCY_MAT_PATH.name}. "
        "This can take several minutes.",
    )
    source = RawFeatureSource(
        OBJECTS_JSON_PATH,
        OBJECT_MAPPING_PATH,
        SALIENCY_MAT_PATH,
        VIDEO_ID,
    )
    try:
        validate_fixations(fixations, source.frame_count, source.height, source.width)
        print(
            f"Validated {source.frame_count} feature frames at "
            f"{source.width}x{source.height} pixels."
        )

        # ------------------------------------------------------------------
        # STEP 3 — Record provenance for the raw input files.
        # SHA-256 hashes identify the exact raw files used in the calculation.
        # Set SKIP_CHECKSUMS=True above for a faster validation-only test.
        # ------------------------------------------------------------------
        print_step(
            3,
            "Record input provenance",
            "Calculating SHA-256 checksums unless SKIP_CHECKSUMS is True.",
        )
        raw_paths = {
            "objects_json": OBJECTS_JSON_PATH.resolve(),
            "object_mapping": OBJECT_MAPPING_PATH.resolve(),
            "saliency_mat": SALIENCY_MAT_PATH.resolve(),
        }
        source_checksums: dict[str, str] = {}
        if not SKIP_CHECKSUMS:
            source_checksums["fixations"] = sha256_file(FIXATIONS_PATH)
            for name, path in raw_paths.items():
                source_checksums[name] = sha256_file(path)
        if VALIDATE_ONLY:
            print("Validation completed; no result workbook was written.")
            return 0

        # ------------------------------------------------------------------
        # STEP 4 — Calculate rNSS with the same mathematical procedure as
        # the compact script, now using the unquantised GBVS feature values.
        # ------------------------------------------------------------------
        print_step(
            4,
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
        # STEP 5 — Save plain values only. No workbook styling is applied.
        # ------------------------------------------------------------------
        print_step(
            5,
            "Write the result workbook",
            "Saving feature_summary, fixation_level, and parameters as "
            "unformatted worksheet values.",
        )
        parameters = {
            "input_source_format": "raw_json_xlsx_mat",
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
                "objects_json": OBJECTS_JSON_PATH.name,
                "object_mapping": OBJECT_MAPPING_PATH.name,
                "saliency_mat": SALIENCY_MAT_PATH.name,
            },
            "checksums_sha256": source_checksums,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "opencv_version": cv2.__version__,
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
