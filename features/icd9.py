"""ICD-9 code -> 3-character category -> chapter mapping.
Reused in Phase 4 to construct graph hierarchy edges.
"""

from typing import Optional
import pandas as pd

_CHAPTER_RANGES = [
    (1, 139, "infectious"),
    (140, 239, "neoplasms"),
    (240, 279, "endocrine_metabolic"),
    (280, 289, "blood"),
    (290, 319, "mental"),
    (320, 389, "nervous_sense"),
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
    (520, 579, "digestive"),
    (580, 629, "genitourinary"),
    (630, 679, "pregnancy"),
    (680, 709, "skin"),
    (710, 739, "musculoskeletal"),
    (740, 759, "congenital"),
    (760, 779, "perinatal"),
    (780, 799, "ill_defined"),
    (800, 999, "injury"),
]


def _normalize(code: object) -> Optional[str]:
    if code is None or pd.isna(code):
        return None
    s = str(code).strip().upper()
    if not s or s in {"NAN", "NONE", "?"}:
        return None
    return s


def category(code: object) -> Optional[str]:
    """Extract the 3-character ICD-9 category (or 4-char E-code category).

    Examples:
        '250.83' -> '250'
        '428'    -> '428'
        'V57'    -> 'V57'
        'E909.2' -> 'E909'
        None/NaN -> None
    """
    s = _normalize(code)
    if s is None:
        return None
    prefix = s.split(".", 1)[0]
    if prefix.startswith("E"):
        return prefix[:4]
    return prefix[:3]


def chapter(code: object) -> str:
    """Map an ICD-9 code to its clinical chapter."""
    cat = category(code)
    if cat is None:
        return "missing"
    if cat.startswith("V"):
        return "supplementary_V"
    if cat.startswith("E"):
        return "external_E"

    try:
        num = int(cat)
    except ValueError:
        return "unknown"

    for low, high, name in _CHAPTER_RANGES:
        if low <= num <= high:
            return name
    return "unknown"