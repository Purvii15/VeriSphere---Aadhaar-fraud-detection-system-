import re

# Verhoeff algorithm tables
# These are fixed lookup tables — don't change them
_MULT = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_PERM = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_check(number):
    """Returns True if the number passes the Verhoeff checksum."""
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _MULT[c][_PERM[i % 8][int(digit)]]
    return c == 0


def validate_aadhaar_number(aadhaar):
    """
    Validate an Aadhaar number.
    Checks: exactly 12 digits, doesn't start with 0 or 1, passes Verhoeff checksum.
    """
    digits = re.sub(r"\D", "", aadhaar)

    if len(digits) != 12:
        return {
            "digits": digits,
            "format_valid": False,
            "verhoeff_valid": False,
            "valid": False,
            "reason": f"Expected 12 digits, got {len(digits)}",
        }

    if digits[0] in ("0", "1"):
        return {
            "digits": digits,
            "format_valid": False,
            "verhoeff_valid": False,
            "valid": False,
            "reason": "Aadhaar number can't start with 0 or 1",
        }

    verhoeff_ok = _verhoeff_check(digits)
    return {
        "digits": digits,
        "format_valid": True,
        "verhoeff_valid": verhoeff_ok,
        "valid": verhoeff_ok,
        "reason": "OK" if verhoeff_ok else "Verhoeff checksum failed — number may be tampered",
    }


def validate_dob_format(dob):
    """Quick check that DOB looks like DD/MM/YYYY or DD-MM-YYYY."""
    return bool(re.match(r"\d{2}[/\-]\d{2}[/\-]\d{4}", dob.strip()))
