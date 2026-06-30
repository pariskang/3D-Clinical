"""Gated real-case generation from CT + TotalSegmentator.

This path requires the optional ``real`` extra (nibabel, scipy) and an external
TotalSegmentator install. It is NOT exercised by the offline test suite. All heavy
imports are lazy so importing this module never pulls in nibabel/scipy.
"""

from __future__ import annotations

_INSTALL_HINT = (
    "build_real_case requires the optional 'real' extra and TotalSegmentator.\n"
    "Install with:\n"
    "    pip install -e '.[real]'\n"
    "    pip install TotalSegmentator\n"
    "Then run TotalSegmentator on your CT to produce per-organ segmentation masks,\n"
    "and pass the CT path plus the segmentation directory to build_real_case()."
)


def build_real_case(ct_path: str, seg_dir: str | None = None, out_dir: str = "cases/real-000"):
    """Build a real case from a CT volume + TotalSegmentator masks (gated).

    Raises a clear RuntimeError with install/run instructions if the optional
    dependencies are unavailable.
    """
    try:
        import nibabel  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(_INSTALL_HINT) from exc

    # pragma: no cover below — real ingestion not part of the offline MVP.
    raise RuntimeError(  # pragma: no cover
        "build_real_case: real CT ingestion is not implemented in the offline MVP. "
        + _INSTALL_HINT
    )
