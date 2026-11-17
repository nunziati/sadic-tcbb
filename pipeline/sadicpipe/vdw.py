"""Van der Waals radii and solvent probe radius.

Atomic radii: the table reported in the paper (Rowland & Taylor 1996 as reported by
Batsanov; P and Se completed from Batsanov 2001). The solvent probe radius is a separate,
explicit parameter with the conventional water value 1.4 A as default.
"""

import warnings

PROBE_RADIUS_WATER = 1.4

VDW_RADII = {
    "H": 1.10,
    "C": 1.77,
    "N": 1.64,
    "O": 1.58,
    "S": 1.81,
    "P": 1.87,
    "SE": 1.90,
}

# Elements mapped onto table entries (D = deuterium in neutron structures).
ELEMENT_ALIASES = {"D": "H"}

FALLBACK_RADIUS = 1.80

def radius_of(element, table=None):
    """Radius for an element symbol (case-insensitive), with alias + fallback."""
    table = VDW_RADII if table is None else table
    el = element.upper()
    el = ELEMENT_ALIASES.get(el, el)
    if el in table:
        return table[el]
    warnings.warn(f"Unknown element '{element}': using fallback radius {FALLBACK_RADIUS} A")
    return FALLBACK_RADIUS
