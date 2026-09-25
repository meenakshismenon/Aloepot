import math


def vpd_from_temp_rh(temp_c: float, rh_pct: float) -> float:
    """Return vapor pressure deficit (kPa) from temperature and RH."""
    if not 0 <= rh_pct <= 100:
        raise ValueError("Relative humidity must be between 0 and 100 percent.")

    saturation_vapor_pressure = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    actual_vapor_pressure = saturation_vapor_pressure * (rh_pct / 100.0)
    return saturation_vapor_pressure - actual_vapor_pressure


def vpd_from_row(temp_c: float, humidity_pct: float) -> float:
    return vpd_from_temp_rh(temp_c, humidity_pct)
