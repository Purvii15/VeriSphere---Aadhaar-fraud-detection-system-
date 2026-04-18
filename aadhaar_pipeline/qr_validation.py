"""
Aadhaar QR code validation using the official decoding pipeline:
  numeric string → big integer → hex → bytes → GZIP decompress → XML parse
"""
import cv2
import gzip
import json
import xml.etree.ElementTree as ET
from io import BytesIO

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except Exception as e:
    PYZBAR_AVAILABLE = False
    print(f"Warning: pyzbar not available. Error: {e}")

# zxing-cpp — self-contained, no external DLLs needed
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except Exception as e:
    ZXING_AVAILABLE = False
    print(f"Warning: zxingcpp not available. Error: {e}")

# QReader — YOLO-based robust QR detector
try:
    from qreader import QReader
    _qreader = QReader()
    QREADER_AVAILABLE = True
except Exception as e:
    _qreader = None
    QREADER_AVAILABLE = False

# OpenCV QR detector as last resort
_cv_qr_detector = cv2.QRCodeDetector()

print(f"QR backends: pyzbar={PYZBAR_AVAILABLE}, zxing={ZXING_AVAILABLE}, qreader={QREADER_AVAILABLE}")


class QRValidator:
    """Detect and decode Aadhaar QR codes using the official numeric→XML pipeline."""

    # ------------------------------------------------------------------ #
    #  QR DETECTION
    # ------------------------------------------------------------------ #

    def _preprocess_variants(self, image):
        """Yield preprocessed image variants for robust QR detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        yield image                                                          # original
        yield gray                                                           # grayscale
        yield cv2.adaptiveThreshold(                                         # adaptive thresh
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        yield cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]           # binary thresh
        yield cv2.equalizeHist(gray)                                         # contrast
        yield cv2.fastNlMeansDenoising(gray)                                 # denoised

        for scale in (0.5, 1.5, 2.0):                                       # rescaled
            w = int(image.shape[1] * scale)
            h = int(image.shape[0] * scale)
            yield cv2.resize(image, (w, h))

    def detect_qr_code(self, image):
        """
        Detect and decode QR using zxingcpp → pyzbar → QReader → OpenCV chain.
        Returns (raw_bytes, qr_type) or (None, None).
        """
        # 1. zxingcpp — best option, self-contained, no DLL issues
        if ZXING_AVAILABLE:
            for variant in self._preprocess_variants(image):
                try:
                    # zxingcpp needs RGB PIL image or numpy array
                    if len(variant.shape) == 3:
                        rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                    else:
                        rgb = cv2.cvtColor(variant, cv2.COLOR_GRAY2RGB)
                    results = zxingcpp.read_barcodes(rgb)
                    for r in results:
                        if r.format.name in ('QRCode', 'MicroQRCode', 'RMQRCode'):
                            raw = r.bytes if r.bytes else r.text.encode('utf-8')
                            return raw, "QRCODE"
                except Exception:
                    continue

        # 2. pyzbar
        if PYZBAR_AVAILABLE:
            for variant in self._preprocess_variants(image):
                try:
                    codes = pyzbar.decode(variant)
                    if codes:
                        return codes[0].data, codes[0].type
                except Exception:
                    pass

        # 3. QReader (YOLO-based)
        if QREADER_AVAILABLE:
            try:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
                results = _qreader.detect_and_decode(image=rgb)
                for r in results:
                    if r:
                        return r.encode('utf-8'), "QRCODE"
                for scale in (2.0, 3.0):
                    h, w = image.shape[:2]
                    big = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                    results = _qreader.detect_and_decode(image=big)
                    for r in results:
                        if r:
                            return r.encode('utf-8'), "QRCODE"
            except Exception:
                pass

        # 4. OpenCV fallback
        for variant in self._preprocess_variants(image):
            try:
                gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if len(variant.shape) == 3 else variant
                data, points, _ = _cv_qr_detector.detectAndDecode(gray)
                if data:
                    return data.encode('utf-8'), "QRCODE"
            except Exception:
                continue

        return None, None

    # ------------------------------------------------------------------ #
    #  DECODING PIPELINE  numeric → bigint → hex → bytes → gzip → XML
    # ------------------------------------------------------------------ #

    def _numeric_to_bytes(self, numeric_str):
        """Convert the large numeric QR string to raw bytes via bigint → hex."""
        numeric_str = numeric_str.strip()
        big_int = int(numeric_str)          # step 1: numeric string → big integer
        hex_str = format(big_int, 'x')      # step 2: big integer → hex string
        if len(hex_str) % 2:                # ensure even length
            hex_str = '0' + hex_str
        return bytes.fromhex(hex_str)       # step 3: hex → bytes

    def _decompress_bytes(self, raw_bytes):
        """GZIP-decompress the raw bytes and return decompressed bytes."""
        with gzip.GzipFile(fileobj=BytesIO(raw_bytes)) as gz:
            return gz.read()

    def _parse_xml(self, xml_string):
        """Parse old-format Aadhaar QR (plain XML)."""
        root = ET.fromstring(xml_string)

        def attr(*keys):
            for k in keys:
                v = root.get(k)
                if v:
                    return v.strip()
            return None

        address_parts = [attr('co'), attr('house'), attr('lm'), attr('loc'),
                         attr('vtc'), attr('subdist'), attr('dist'), attr('state'), attr('pc')]
        address = ', '.join(p for p in address_parts if p) or None

        photo_b64 = None
        pht = root.find('Pht')
        if pht is not None and pht.text:
            photo_b64 = pht.text.strip()

        return {
            'uid': attr('uid'), 'name': attr('name'), 'dob': attr('dob'),
            'gender': attr('gender'), 'co': attr('co'), 'house': attr('house'),
            'landmark': attr('lm'), 'locality': attr('loc'), 'vtc': attr('vtc'),
            'subdist': attr('subdist'), 'district': attr('dist'), 'state': attr('state'),
            'pincode': attr('pc'), 'address': address, 'photo': photo_b64,
        }

    def _parse_secure_binary(self, data):
        """
        Parse new/secure Aadhaar QR format (post-2018).
        Text fields are 0xFF-delimited. Photo (JPEG2000) is appended after
        the last text field as a continuous binary blob.
        J2K codestream starts with SOC (ff4f) + SIZ (ff51) markers back-to-back.
        JP2 file box starts with 0000000c6a502020.
        """
        # Strict JPEG2000 signatures — require both SOC+SIZ for J2K codestream
        # to avoid false positives on 0xff4f appearing inside text fields
        JP2_SIGS = [
            b'\xff\x4f\xff\x51',                    # J2K codestream: SOC + SIZ markers
            b'\x00\x00\x00\x0c\x6a\x50\x20\x20',   # JP2 file signature box
            b'\xff\xd8\xff\xe0',                    # JPEG (JFIF)
            b'\xff\xd8\xff\xe1',                    # JPEG (EXIF)
        ]

        # Find where the photo binary starts
        photo_start = -1
        for sig in JP2_SIGS:
            idx = data.find(sig)
            if idx > 50:  # must be well after text fields
                photo_start = idx
                break

        if photo_start > 0:
            text_data  = data[:photo_start]
            photo_bytes = data[photo_start:]
        else:
            text_data  = data
            photo_bytes = None

        parts = text_data.split(b'\xff')

        def s(i):
            try:
                if i < len(parts):
                    val = parts[i].decode('utf-8', errors='replace').strip()
                    return val if val else None
                return None
            except Exception:
                return None

        # Auto-detect the base offset.
        # The version field is a single digit ("2" or "3").
        # When decoded from image via pyzbar there may be a leading binary byte
        # that creates an extra empty/garbage part before the version.
        # Find the first part that is exactly a single digit 2-9.
        import re as _re
        base = 0
        for _i, _p in enumerate(parts[:5]):
            try:
                _v = _p.decode('utf-8', errors='replace').strip()
                if _re.fullmatch(r'[2-9]', _v):
                    base = _i
                    break
            except Exception:
                pass

        def sf(i):
            return s(base + i)

        # Secure QR v3 field layout (0xFF-delimited), relative to base:
        # 0: version, 1: ref_number (first 4 = last 4 of UID),
        # 2: name, 3: dob, 4: gender, 5: co, 6: vtc (city),
        # 7: post_office, 8: house, 9: street, 10: pincode,
        # 11: district, 12: state, 13: locality, 14: subdist
        version    = sf(0)
        ref_number = sf(1)
        last4 = ref_number[:4] if ref_number and len(ref_number) >= 4 and ref_number[:4].isdigit() else None
        name     = sf(2)
        dob      = sf(3)
        gender   = sf(4)
        co       = sf(5)
        vtc      = sf(6)
        house    = sf(8)
        street   = sf(9)
        pincode  = sf(10)
        district = sf(11)
        state    = sf(12)
        landmark = sf(13)
        subdist  = sf(14)

        address_parts = [co, house, street, landmark, vtc, subdist, district, state, pincode]
        address = ', '.join(p for p in address_parts if p) or None

        photo_present = bool(photo_bytes and len(photo_bytes) > 100)

        return {
            'uid':        None,
            'last4':      last4,
            'name':       name,
            'dob':        dob,
            'gender':     gender,
            'co':         co,
            'house':      house,
            'landmark':   landmark,
            'locality':   street,
            'vtc':        vtc,
            'subdist':    subdist,
            'district':   district,
            'state':      state,
            'pincode':    pincode,
            'address':    address,
            'photo':      photo_bytes if photo_present else None,
            'qr_version': version,
        }

    def decode_qr_data(self, raw_bytes):
        """
        Full pipeline: raw QR bytes → structured dict.
        Handles both old (XML) and new (0xFF-delimited binary) formats.
        Returns (parsed_dict, error_string).  On success error_string is None.
        """
        try:
            numeric_str = raw_bytes.decode('utf-8', errors='ignore').strip()

            if not numeric_str.isdigit():
                return None, 'qr_data_not_numeric'

            compressed   = self._numeric_to_bytes(numeric_str)
            decompressed = self._decompress_bytes(compressed)

            # Detect format: old XML starts with '<', new binary uses 0xFF delimiters
            if decompressed.lstrip(b' \t\r\n').startswith(b'<'):
                xml_string = decompressed.decode('utf-8', errors='replace')
                parsed = self._parse_xml(xml_string)
                parsed['_format'] = 'old'
            else:
                parsed = self._parse_secure_binary(decompressed)
                parsed['_format'] = 'secure'

            return parsed, None

        except (ValueError, OverflowError) as e:
            return None, f'numeric_conversion_error: {e}'
        except (OSError, EOFError) as e:
            return None, f'gzip_error: {e}'
        except ET.ParseError as e:
            return None, f'xml_parse_error: {e}'
        except Exception as e:
            return None, f'decode_error: {e}'

    # ------------------------------------------------------------------ #
    #  COMPARISON HELPERS
    # ------------------------------------------------------------------ #

    def _norm(self, s):
        return str(s).lower().strip().replace(' ', '') if s else ''

    def _dates_match(self, d1, d2):
        # normalize separators including pipe character OCR sometimes reads
        def norm_date(d):
            d = str(d).strip()
            # extract digit sequences only
            import re
            parts = re.findall(r'\d+', d)
            return ''.join(parts)
        n1 = norm_date(d1)
        n2 = norm_date(d2)
        return bool(n1 and n2 and len(n1) >= 6 and n1 == n2)

    # ------------------------------------------------------------------ #
    #  PUBLIC ENTRY POINT
    # ------------------------------------------------------------------ #

    def validate_qr(self, image, ocr_data, qr_crop=None):
        """
        Detect QR, decode it, compare with OCR data.
        Tries qr_crop first (YOLO-detected region), then falls back to full image.
        Returns a result dict compatible with the pipeline.
        """
        raw_bytes = None

        # try YOLO crop first at multiple scales
        if qr_crop is not None and qr_crop.size > 0:
            h, w = qr_crop.shape[:2]
            for scale in [1.0, 2.0, 3.0]:
                attempt = qr_crop if scale == 1.0 else cv2.resize(
                    qr_crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                raw_bytes, _ = self.detect_qr_code(attempt)
                if raw_bytes:
                    break

        # fallback to full image
        if raw_bytes is None:
            raw_bytes, _ = self.detect_qr_code(image)

        if raw_bytes is None:
            return {
                'qr_found': False, 'qr_valid': False, 'qr_format': None,
                'match_score': 0, 'qr_raw_data': None, 'qr_parsed_data': None,
                'comparison_details': {}, 'error': 'No QR code detected in image',
            }

        parsed, error = self.decode_qr_data(raw_bytes)

        if parsed is None:
            return {
                'qr_found': True, 'qr_valid': False, 'qr_format': 'unknown',
                'match_score': 0,
                'qr_raw_data': raw_bytes.decode('utf-8', errors='replace')[:200],
                'qr_parsed_data': None, 'comparison_details': {},
                'error': error,
            }

        # Cross-reference QR data with OCR
        matches, total = 0, 0
        details = {}

        import re as _re
        def _extract_date(raw):
            """Pull just the date digits from noisy OCR output."""
            # DD/MM/YYYY or DD-MM-YYYY or DD|MM|YYYY
            m = _re.search(r'(\d{1,2})[\/\-\|\\](\d{1,2})[\/\-\|\\](\d{4})', raw)
            if m:
                return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
            # YYYY-MM-DD
            m = _re.search(r'(\d{4})[\/\-\|\\](\d{1,2})[\/\-\|\\](\d{1,2})', raw)
            if m:
                return f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
            return raw

        ocr_dob_clean = _extract_date(ocr_data.get('dob', ''))

        qr_fmt = parsed.get('_format', 'old')
        checks = [
            # secure format doesn't store full UID — skip aadhaar_number check
            ('aadhaar_number', parsed.get('uid') if qr_fmt != 'secure' else None,
             ocr_data.get('aadhaar_number'), 'exact'),
            # for secure format, compare last 4 digits
            ('last4', parsed.get('last4') if qr_fmt == 'secure' else None,
             _re.sub(r'\D', '', ocr_data.get('aadhaar_number', ''))[-4:] or None, 'exact'),
            ('name',           parsed.get('name'), ocr_data.get('name'),           'partial'),
            ('dob',            parsed.get('dob'),  ocr_dob_clean,                  'date'),
        ]

        for field, qr_val, ocr_val, mode in checks:
            if qr_val and ocr_val:
                total += 1
                if mode == 'exact':
                    match = self._norm(qr_val) == self._norm(ocr_val)
                elif mode == 'partial':
                    qn, on = self._norm(qr_val), self._norm(ocr_val)
                    match = qn in on or on in qn
                else:  # date
                    match = self._dates_match(qr_val, ocr_val)
                if match:
                    matches += 1
                # store cleaned ocr_value so UI shows readable text
                display_ocr = ocr_dob_clean if field == 'dob' else ocr_val
                details[field] = {'qr_value': qr_val, 'ocr_value': display_ocr, 'match': match}

        match_score = (matches / total * 100) if total else 0

        return {
            'qr_found': True,
            'qr_valid': True,
            'qr_format': qr_fmt,
            'qr_raw_data': raw_bytes.decode('utf-8', errors='replace')[:200],
            'qr_parsed_data': parsed,
            'match_score': match_score,
            'matches': matches,
            'total_checks': total,
            'comparison_details': details,
            'error': None,
        }

    # ------------------------------------------------------------------ #
    #  STANDALONE HELPER  (for testing / CLI use)
    # ------------------------------------------------------------------ #

    def process_image(self, image_path):
        """
        Process an Aadhaar card image file and return the full JSON result.
        Useful for standalone testing.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not read image: {image_path}'}

        raw_bytes, _ = self.detect_qr_code(image)
        if raw_bytes is None:
            return {'error': 'No QR code detected'}

        parsed, error = self.decode_qr_data(raw_bytes)
        if parsed is None:
            return {'error': error}

        # Remove photo from JSON output (binary blob)
        result = {k: v for k, v in parsed.items() if k != 'photo'}
        result['photo_present'] = parsed.get('photo') is not None
        return result
