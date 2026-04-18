"""
ngrok http 5000
py flask_app.py
Flask web server for Aadhaar Fraud Detection
Serves the Stitch HTML frontend and exposes /analyze API endpoint.
"""
import os
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

from aadhaar_pipeline.pipeline import run_pipeline
from aadhaar_pipeline.detector import load_model
from aadhaar_pipeline.tampering import load_tampering_model

# Base directory = folder where flask_app.py lives
BASE_DIR = Path(__file__).parent

app = Flask(__name__)

# cache models at startup
_yolo = None
_resnet = None

def get_models():
    global _yolo, _resnet
    if _yolo is None:
        _yolo = load_model(str(BASE_DIR / "aadhaar_best.pt"))
    if _resnet is None:
        _resnet = load_tampering_model(None, device="cpu")
    return _yolo, _resnet


_MAIN_HTML = """<!DOCTYPE html>
<html class="dark" lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>VeriSphere | Aadhaar Fraud Detection</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%238083ff' d='M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z'/%3E%3C/svg%3E"/>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
  :root {
    --app-bg: #0b0c10;
    --card-bg: rgba(28,31,41,0.4);
    --border-glass: rgba(255,255,255,0.08);
    --surface-container-highest: rgba(255,255,255,0.06);
    --on-surface-variant: #c7c4d7;
  }
  * { box-sizing: border-box; }
  body { background: var(--app-bg); color: #e0e2ef; font-family: 'Inter', sans-serif; min-height: 100vh; overflow-x: hidden; }
  h1, h2, .font-black { font-family: 'Outfit', sans-serif !important; }
  .glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border-glass);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
  }
  .text-on-surface-variant { color: var(--on-surface-variant); }
  .bg-surface-container-highest { background: var(--surface-container-highest); }
  .border-white\\/5 { border-color: rgba(255,255,255,0.05); }
  .divide-y > * + * { border-top: 1px solid rgba(255,255,255,0.05); }
  .stagger-load > * { animation: fadeUp 0.5s ease both; }
  @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
  .border-dashed { border: 2px dashed rgba(128,131,255,0.3); transition: border-color 0.2s; cursor: pointer; }
  .border-dashed:hover { border-color: rgba(128,131,255,0.7); }
  .bg-blob { position:fixed; filter:blur(120px); opacity:0.25; border-radius:50%; pointer-events:none; z-index:0; }
</style>
</head>
<body>
  <div class="bg-blob" style="top:-10%;left:-10%;width:500px;height:500px;background:rgba(87,27,193,0.5);"></div>
  <div class="bg-blob" style="bottom:-10%;right:-10%;width:600px;height:600px;background:rgba(128,131,255,0.3);"></div>

  <!-- Header -->
  <header style="background:rgba(16,19,28,0.9);border-bottom:1px solid rgba(255,255,255,0.06);backdrop-filter:blur(12px);position:sticky;top:0;z-index:50;"
          class="flex justify-between items-center px-6 py-4">
    <span class="text-xl font-black" style="background:linear-gradient(to right,#8083ff,#571bc1);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
      🛡️ VeriSphere
    </span>
    <div class="flex gap-6">
      <a href="/" class="text-sm font-semibold border-b-2 pb-1" style="color:#8083ff;border-color:#8083ff">Dashboard</a>
      <a href="/qr-decode" class="text-sm font-medium text-on-surface-variant hover:text-white transition-colors">QR Decode</a>
    </div>
  </header>

  <!-- Main layout -->
  <main class="relative z-10 max-w-screen-xl mx-auto px-4 py-8 grid xl:grid-cols-12 gap-6">

    <!-- Left column: upload -->
    <div class="xl:col-span-5 flex flex-col gap-6">
      <div>
        <h1 class="text-3xl font-black tracking-tight mb-1">Aadhaar Fraud Detection</h1>
        <p class="text-sm text-on-surface-variant">Upload an Aadhaar card image for multi-stage verification.</p>
      </div>

      <!-- Upload zone -->
      <div class="glass-card rounded-xl p-6 flex flex-col gap-4">
        <div class="border-dashed rounded-xl p-8 flex flex-col items-center justify-center gap-3 text-center" style="min-height:180px">
          <span style="font-size:2.5rem">☁️</span>
          <div>
            <p class="font-semibold text-sm">Click or drag &amp; drop to upload</p>
            <p class="text-xs text-on-surface-variant mt-1">JPG, PNG, WEBP supported</p>
          </div>
        </div>

        <!-- Preview -->
        <div class="aspect-[1.58/1] rounded-xl overflow-hidden relative bg-surface-container-highest flex items-center justify-center" style="min-height:160px">
          <span class="text-on-surface-variant text-xs">No image selected</span>
        </div>

        <!-- Analyze button -->
        <button disabled
          class="w-full py-3 rounded-xl font-bold text-sm opacity-50 cursor-not-allowed bg-surface-container-highest text-on-surface-variant border border-white/5 transition-all">
          🔍&nbsp;Analyze Card
        </button>
      </div>

      <!-- Info card -->
      <div class="glass-card rounded-xl p-5">
        <h3 class="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-3">Verification Pipeline</h3>
        <div class="space-y-2 text-xs text-on-surface-variant">
          <div class="flex items-center gap-2"><span style="color:#8083ff">①</span> YOLOv8 region detection</div>
          <div class="flex items-center gap-2"><span style="color:#34d399">②</span> EasyOCR text extraction</div>
          <div class="flex items-center gap-2"><span style="color:#facc15">③</span> Verhoeff checksum validation</div>
          <div class="flex items-center gap-2"><span style="color:#60a5fa">④</span> QR decode &amp; cross-check</div>
          <div class="flex items-center gap-2"><span style="color:#f87171">⑤</span> SSIM photo matching</div>
          <div class="flex items-center gap-2"><span style="color:#a78bfa">⑥</span> ELA + ResNet forensics</div>
        </div>
      </div>
    </div>

    <!-- Right column: results -->
    <div class="xl:col-span-7 flex flex-col">
      <div class="glass-card rounded-xl flex-1 flex flex-col items-center justify-center p-12 text-center">
        <div class="relative mb-8">
          <div class="absolute inset-0 bg-[#8083ff]/10 blur-[60px] rounded-full"></div>
          <span style="font-size:6rem;opacity:0.15">🛡️</span>
        </div>
        <h2 class="text-3xl font-black mb-4 tracking-tight">Ready to Analyse</h2>
        <p class="text-on-surface-variant max-w-md mx-auto leading-relaxed">
          Upload an Aadhaar card image and press <strong>Analyse Card</strong> to begin multi-stage verification.
        </p>
      </div>
    </div>

  </main>
</body>
</html>"""


@app.route("/")
def index():
    js = _get_glue_js()
    icon_css = """
<style>
.material-symbols-outlined { font-family: 'Material Symbols Outlined', sans-serif; }
button.material-symbols-outlined, span.material-symbols-outlined {
  display: inline-flex; align-items: center; justify-content: center;
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
  const emojiMap = {
    cloud_upload: '☁️', search: '🔍', shield_with_heart: '🛡️',
    shield: '🛡️', grid_view: '⊞', qr_code_scanner: '📷',
    smart_card_reader: '💳', inventory_2: '📦', hub: '🔗',
    security: '🔒', history: '🕐', settings: '⚙️', image: '🖼️',
    speed: '⚡', verified: '✅', query_stats: '📊', add: '＋',
    manage_search: '🔎', text_fields: '🔤', face: '🧑', policy: '🛡️',
    zoom_in: '🔍', progress_activity: '⏳'
  };
  function checkAndReplace() {
    const test = document.createElement('span');
    test.className = 'material-symbols-outlined';
    test.style.cssText = 'position:absolute;visibility:hidden;font-size:24px';
    test.textContent = 'home';
    document.body.appendChild(test);
    const loaded = test.offsetWidth < 30;
    document.body.removeChild(test);
    if (!loaded) {
      document.querySelectorAll('.material-symbols-outlined').forEach(el => {
        const name = el.textContent.trim();
        if (emojiMap[name]) el.textContent = emojiMap[name];
      });
    }
  }
  setTimeout(checkAndReplace, 1500);
});
</script>
"""
    html = _MAIN_HTML.replace("</head>", icon_css + "\n</head>")
    html = html.replace("</body>", js + "\n</body>")
    return render_template_string(html)


@app.route("/qr-decode")
def qr_decode_page():
    return render_template_string(_QR_DECODE_HTML)


@app.route("/decode-qr", methods=["POST"])
def decode_qr():
    data = request.get_json()
    raw = (data or {}).get("raw", "").strip()
    if not raw:
        return jsonify({"error": "No QR data provided"}), 400
    try:
        from aadhaar_pipeline.qr_validation import QRValidator
        import io
        validator = QRValidator()
        raw_bytes = raw.encode("utf-8")
        parsed, error = validator.decode_qr_data(raw_bytes)
        if parsed is None:
            return jsonify({"error": error or "Failed to decode QR data"})

        # extract and convert photo
        photo_b64 = None
        _photo = parsed.pop("photo", None)
        if isinstance(_photo, (bytes, bytearray)) and len(_photo) > 100:
            try:
                from PIL import Image as _Img
                img = _Img.open(io.BytesIO(bytes(_photo)))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                photo_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                try:
                    arr = np.frombuffer(bytes(_photo), dtype=np.uint8)
                    img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img_cv is not None:
                        _, buf = cv2.imencode(".jpg", img_cv)
                        photo_b64 = base64.b64encode(buf).decode()
                except Exception:
                    pass

        result = {k: (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v)
                  for k, v in parsed.items()}
        result["photo_b64"] = photo_b64
        return jsonify({"success": True, "data": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    tmp = BASE_DIR / "temp_upload.jpg"
    
    # ensure parent dir exists and save
    tmp.parent.mkdir(parents=True, exist_ok=True)
    file.save(str(tmp))
    
    if not tmp.exists():
        return jsonify({"error": "Failed to save uploaded image"}), 500

    try:
        yolo, resnet = get_models()
        result = run_pipeline(
            image_path=str(tmp),
            yolo_weights=str(BASE_DIR / "aadhaar_best.pt"),
            device="cpu",
            verbose=False,
            yolo_model=yolo,
            resnet_tuple=resnet,
        )

        # build annotated image as base64 for the frontend
        image = cv2.imread(str(tmp))
        annotated = _annotate(image, result["detections"])
        _, buf = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buf).decode()

        tmp.unlink(missing_ok=True)
        result["annotated_image"] = annotated_b64
        # make result JSON-serialisable
        result.pop("qr_fields", None)
        result = _make_serialisable(result)
        return jsonify(result)

    except Exception as e:
        tmp.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


def _annotate(image, detections):
    colors = {
        "aadhaar_number": (99, 102, 241),
        "address":        (251, 191, 36),
        "dob":            (52, 211, 153),
        "face":           (248, 113, 113),
        "name":           (167, 139, 250),
        "qr_code":        (56, 189, 248),
    }
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        col = colors.get(det["class_name"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        lbl = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1, max(y1 - 18, 0)), (x1 + tw + 6, max(y1, 18)), col, -1)
        cv2.putText(out, lbl, (x1 + 3, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1)
    return out


_QR_DECODE_HTML = """<!DOCTYPE html>
<html class="dark" lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>VeriSphere | QR Decode</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%238083ff' d='M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z'/%3E%3C/svg%3E"/>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
    <style>
      :root {
        --app-bg: #0b0c10;
        --card-bg: rgba(28, 31, 41, 0.4);
        --border-glass: rgba(255, 255, 255, 0.08);
      }
      body { background: var(--app-bg); color: #e0e2ef; font-family: 'Inter', sans-serif; position: relative; overflow-x: hidden; }
      h1, .font-black { font-family: 'Outfit', sans-serif !important; }
      .glass { 
        background: var(--card-bg); 
        backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--border-glass); 
        border-radius: 12px; 
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
      }
      .field-row { display:flex; justify-content:space-between; align-items:flex-start;
                   padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
      .field-row:last-child { border-bottom:none; }
      .field-label { font-size:11px; font-weight:700; text-transform:uppercase;
                     letter-spacing:.08em; color:#c7c4d7; min-width:130px; }
      .field-value { font-size:13px; font-weight:500; text-align:right; max-width:65%; word-break:break-word; }
      
      .bg-blob { position: absolute; filter: blur(100px); opacity: 0.3; border-radius: 50%; z-index: -1; }
    </style>
  </head>
  <body class="min-h-screen relative">
    <div class="bg-blob" style="top:-10%;left:-5%;width:400px;height:400px;background:rgba(87,27,193,0.4);"></div>
    <div class="bg-blob" style="bottom:-10%;right:-5%;width:500px;height:500px;background:rgba(128,131,255,0.3);"></div>

<!-- TopNav -->
<header style="background:#10131c;border-bottom:1px solid rgba(255,255,255,0.06)"
        class="flex justify-between items-center px-6 py-4">
  <a href="/" class="text-xl font-black" style="background:linear-gradient(to right,#8083ff,#571bc1);-webkit-background-clip:text;-webkit-text-fill-color:transparent">VeriSphere</a>
  <div class="flex gap-8">
    <a href="/" class="text-sm font-medium" style="color:#c7c4d7">Dashboard</a>
    <a href="/qr-decode" class="text-sm font-medium border-b-2 pb-1" style="color:#8083ff;border-color:#8083ff">QR Decode</a>
  </div>
</header>

<main class="max-w-2xl mx-auto px-6 py-12">
  <div class="mb-8">
    <h1 class="text-3xl font-black tracking-tight mb-2">📷 QR Code Decoder</h1>
    <p style="color:#c7c4d7" class="text-sm">Paste raw QR data from a handheld scanner to decode Aadhaar fields instantly.</p>
  </div>

  <!-- Input -->
  <div class="glass p-6 mb-6">
    <label class="block text-xs font-bold uppercase tracking-widest mb-3" style="color:#c7c4d7">Raw QR Data</label>
    <textarea id="qr-input" rows="5" placeholder="Paste QR string here and click Decode..."
      class="w-full rounded-lg p-3 text-sm font-mono resize-none focus:outline-none"
      style="background:#10131c;border:1px solid rgba(128,131,255,0.3);color:#e0e2ef;"></textarea>
    <div class="flex gap-3 mt-4">
      <button id="decode-btn" onclick="decodeQR()"
        class="flex-1 py-3 rounded-xl font-bold text-white text-sm"
        style="background:linear-gradient(to right,#8083ff,#571bc1);cursor:pointer">
        🔍 Decode QR
      </button>
      <button onclick="document.getElementById('qr-input').value='';document.getElementById('result').innerHTML=''"
        class="px-5 py-3 rounded-xl font-bold text-sm"
        style="background:#272a34;color:#c7c4d7;cursor:pointer">
        Clear
      </button>
    </div>
  </div>

  <!-- Result -->
  <div id="result"></div>
</main>

<script>
async function decodeQR() {
  const raw = document.getElementById('qr-input').value.trim();
  if (!raw) return;

  const btn = document.getElementById('decode-btn');
  btn.textContent = '⏳ Decoding...';
  btn.disabled = true;

  try {
    const res  = await fetch('/decode-qr', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({raw})
    });
    const json = await res.json();
    renderResult(json);
  } catch(e) {
    document.getElementById('result').innerHTML =
      `<div class="glass p-6" style="border:1px solid #f87171;color:#f87171">Error: ${e}</div>`;
  } finally {
    btn.textContent = '🔍 Decode QR';
    btn.disabled = false;
  }
}

function renderResult(json) {
  const el = document.getElementById('result');
  if (json.error) {
    el.innerHTML = `<div class="glass p-6" style="border:1px solid #f87171">
      <p style="color:#f87171" class="font-bold">❌ Decode Failed</p>
      <p class="text-sm mt-1" style="color:#c7c4d7">${json.error}</p>
    </div>`;
    return;
  }

  const d = json.data;
  const fmt = d._format || 'unknown';
  const fmtColor = fmt === 'secure' ? '#4ade80' : '#facc15';

  const fields = [
    ['Format',    fmt.toUpperCase()],
    ['Name',      d.name],
    ['Date of Birth', d.dob],
    ['Gender',    d.gender === 'M' ? 'Male' : d.gender === 'F' ? 'Female' : d.gender],
    ['UID / Last 4', d.uid || (d.last4 ? `xxxx xxxx ${d.last4}` : null)],
    ['Address',   d.address],
    ['Pincode',   d.pincode],
    ['State',     d.state],
    ['District',  d.district],
    ['VTC',       d.vtc],
    ['Version',   d.qr_version],
  ].filter(([,v]) => v);

  const rows = fields.map(([label, val]) => `
    <div class="field-row">
      <span class="field-label">${label}</span>
      <span class="field-value">${val}</span>
    </div>`).join('');

  const photoHtml = d.photo_b64
    ? `<div class="mt-6 pt-5" style="border-top:1px solid rgba(255,255,255,0.06)">
         <p class="field-label mb-3">Photo (from QR)</p>
         <img src="data:image/jpeg;base64,${d.photo_b64}"
              style="width:100px;height:120px;object-fit:cover;border-radius:8px;border:1px solid rgba(255,255,255,0.1);cursor:zoom-in"
              onclick="window.open(this.src)" title="Click to enlarge" />
       </div>`
    : `<p style="color:#9ca3af;font-size:11px;margin-top:16px;font-style:italic">
         📷 No photo embedded in this QR code
       </p>`;

  el.innerHTML = `
    <div class="glass p-6">
      <div class="flex justify-between items-center mb-5">
        <h2 class="font-black text-lg">Decoded Fields</h2>
        <span class="text-xs font-bold px-3 py-1 rounded-full"
              style="background:${fmtColor}22;color:${fmtColor}">${fmt.toUpperCase()} FORMAT</span>
      </div>
      ${rows}
      ${photoHtml}
    </div>`;
}

// auto-focus input and support Enter key
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('qr-input').focus();
  document.getElementById('qr-input').addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'Enter') decodeQR();
  });
});
</script>
</body>
</html>"""


def _make_serialisable(obj):
    """Recursively convert any non-JSON-safe types (bytes, numpy, etc.) to strings."""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    # numpy scalar types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def _get_glue_js():
    """JavaScript that wires the upload zone + button to /analyze and renders results."""
    return """
<script>
(function () {
  // ── state ──────────────────────────────────────────────────────────────────
  let selectedFile = null;

  // ── grab elements ──────────────────────────────────────────────────────────
  const uploadZone  = document.querySelector('.border-dashed');
  const analyzeBtn  = document.querySelector('button[disabled]') ||
                      [...document.querySelectorAll('button')].find(b => b.textContent.includes('Analyze'));
  const previewWrap = document.querySelector('.aspect-\\\\[1\\\\.58\\\\/1\\\\]') ||
                      document.querySelector('[class*="aspect"]');

  // hidden file input
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/jpeg,image/png,image/webp';
  fileInput.style.display = 'none';
  document.body.appendChild(fileInput);

  // ── upload zone click ──────────────────────────────────────────────────────
  if (uploadZone) uploadZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (!fileInput.files.length) return;
    selectedFile = fileInput.files[0];

    // ── clear old results — restore idle state ─────────────────────────────
    const resultsCol = document.querySelector('.xl\\\\:col-span-7');
    if (resultsCol) {
      resultsCol.innerHTML = `
        <div class="glass-card rounded-xl flex-1 flex flex-col items-center justify-center p-12 text-center border-white/[0.04]">
          <div class="relative mb-8">
            <div class="absolute inset-0 bg-[#8083ff]/10 blur-[60px] rounded-full"></div>
            <span style="font-size:6rem;opacity:0.15">🛡️</span>
          </div>
          <h2 class="text-3xl font-black mb-4 tracking-tight">Ready to Analyse</h2>
          <p class="text-on-surface-variant max-w-md mx-auto leading-relaxed mb-10">
            Card loaded. Press <strong>Analyse Card</strong> to begin the multi-stage verification process.
          </p>
        </div>`;
    }
    // show preview
    const url = URL.createObjectURL(selectedFile);
    if (previewWrap) {
      previewWrap.innerHTML = `
        <img src="${url}" class="w-full h-full object-contain rounded-xl cursor-zoom-in" title="Click to enlarge"
          onclick="_openLightbox(this.src)" />
        <div class="absolute bottom-2 left-2 right-2 flex items-center justify-between pointer-events-none">
          <span class="bg-black/60 text-white text-[10px] px-2 py-1 rounded max-w-[70%] truncate">
            � ${selectedFile.name}
          </span>
          <span class="bg-black/60 text-white text-[10px] px-2 py-1 rounded">
            🔍 Click to enlarge
          </span>
        </div>`;
    }
    // enable button — swap to gradient style
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed',
        'bg-surface-container-highest', 'text-on-surface-variant', 'border', 'border-white/5');
      analyzeBtn.classList.add('bg-gradient-to-r', 'from-[#8083ff]', 'to-[#571bc1]',
        'text-white', 'shadow-[0_0_20px_rgba(128,131,255,0.35)]', 'cursor-pointer',
        'hover:opacity-90', 'transition-all');
      // auto-start analysis immediately after file is selected
      analyzeBtn.click();
    }
  });

  // drag-and-drop
  if (uploadZone) {
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('border-[#8083ff]'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('border-[#8083ff]'));
    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      uploadZone.classList.remove('border-[#8083ff]');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
      }
    });
  }

  // ── analyze button ─────────────────────────────────────────────────────────
  const stages = [
    { icon: '📷', label: 'QR Decode',       color: '#60a5fa', ms: 900  },
    { icon: '🔤', label: 'OCR Extract',      color: '#8083ff', ms: 900  },
    { icon: '✅', label: 'Verhoeff Check',   color: '#4ade80', ms: 700  },
    { icon: '🧑', label: 'Photo Match',      color: '#facc15', ms: 900  },
    { icon: '🛡️', label: 'Fraud Scoring',    color: '#f87171', ms: 700  },
  ];

  function showProcessingAnimation(imgUrl) {
    const resultsCol = document.querySelector('.xl\\\\:col-span-7');
    if (!resultsCol) return;

    // build stage pills HTML
    const pills = stages.map((st, i) => `
      <div id="_st_${i}" class="flex flex-col items-center gap-2 opacity-30 transition-all duration-500" style="transition:opacity 0.6s,transform 0.6s">
        <div id="_stc_${i}" class="w-14 h-14 rounded-full border-2 flex items-center justify-center relative"
             style="border-color:${st.color}33">
          <span style="font-size:1.4rem;color:${st.color}99">${st.icon}</span>
          <svg id="_stsvg_${i}" class="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 56 56">
            <circle cx="28" cy="28" r="26" fill="none" stroke="${st.color}" stroke-width="2"
              stroke-dasharray="163" stroke-dashoffset="163"
              style="transition:stroke-dashoffset 0.7s ease-in-out" id="_stcirc_${i}"/>
          </svg>
        </div>
        <span class="text-[10px] font-bold uppercase tracking-wider" style="color:${st.color}88">${st.label}</span>
      </div>`).join('');

    resultsCol.innerHTML = `
      <div class="glass-card rounded-xl flex-1 flex flex-col items-center justify-center p-10 text-center relative overflow-hidden" id="_proc_panel">
        <!-- particle canvas -->
        <canvas id="_pcv" class="absolute inset-0 w-full h-full pointer-events-none opacity-40"></canvas>

        <!-- radar rings -->
        <div class="relative mb-8 flex items-center justify-center" style="width:160px;height:160px">
          ${[0,1,2].map(r=>`
          <div class="absolute rounded-full border border-[#8083ff]/20 animate-ping"
               style="width:${80+r*40}px;height:${80+r*40}px;animation-duration:${2.4+r*0.7}s;animation-delay:${r*0.5}s"></div>`).join('')}
          <div class="w-20 h-20 rounded-full bg-[#8083ff]/10 border border-[#8083ff]/30 flex items-center justify-center relative z-10"
               style="box-shadow:0 0 30px rgba(128,131,255,0.25)">
            ${imgUrl ? `<img src="${imgUrl}" class="w-full h-full object-cover rounded-full opacity-70"/>` :
              `<span style="font-size:2rem">🛡️</span>`}
          </div>
        </div>

        <!-- status text -->
        <div id="_proc_label" class="text-sm font-bold text-[#8083ff] mb-1 tracking-wide">Initialising...</div>
        <div class="text-xs text-on-surface-variant opacity-50 mb-8">Multi-stage cryptographic verification</div>

        <!-- stage pills -->
        <div class="flex gap-6 mb-8 flex-wrap justify-center">${pills}</div>

        <!-- overall progress bar -->
        <div class="w-full max-w-xs bg-surface-container-highest rounded-full h-1.5 overflow-hidden">
          <div id="_prog_bar" class="h-full rounded-full bg-gradient-to-r from-[#8083ff] to-[#571bc1]"
               style="width:0%;transition:width 0.7s ease-in-out;box-shadow:0 0 8px rgba(128,131,255,0.5)"></div>
        </div>
        <div id="_prog_pct" class="text-[10px] font-mono text-on-surface-variant opacity-40 mt-2">0%</div>
      </div>`;

    // particle canvas animation
    const cvs = document.getElementById('_pcv');
    if (cvs) {
      const ctx = cvs.getContext('2d');
      const pts = Array.from({length:40}, () => ({
        x: Math.random(), y: Math.random(),
        vx: (Math.random()-.5)*.0008, vy: (Math.random()-.5)*.0008,
        r: Math.random()*1.5+.5, a: Math.random()
      }));
      let raf;
      function drawPts() {
        cvs.width = cvs.offsetWidth; cvs.height = cvs.offsetHeight;
        ctx.clearRect(0,0,cvs.width,cvs.height);
        pts.forEach(p => {
          p.x += p.vx; p.y += p.vy;
          if (p.x<0||p.x>1) p.vx*=-1;
          if (p.y<0||p.y>1) p.vy*=-1;
          ctx.beginPath();
          ctx.arc(p.x*cvs.width, p.y*cvs.height, p.r, 0, Math.PI*2);
          ctx.fillStyle = `rgba(128,131,255,${p.a*0.6})`;
          ctx.fill();
        });
        raf = requestAnimationFrame(drawPts);
      }
      drawPts();
      resultsCol._stopParticles = () => cancelAnimationFrame(raf);
    }

    // animate stages sequentially — loop back on last stage while server is still working
    let elapsed = 0;
    let stageTimers = [];
    stages.forEach((st, i) => {
      const t = setTimeout(() => {
        const el    = document.getElementById(`_st_${i}`);
        const circ  = document.getElementById(`_stcirc_${i}`);
        const label = document.getElementById('_proc_label');
        if (el)    { el.style.opacity='1'; el.style.transform='scale(1.1)'; }
        if (circ)  circ.style.strokeDashoffset = '0';
        if (label) label.textContent = st.label + '...';
        const pct = Math.round((i+1)/stages.length*82);
        const bar   = document.getElementById('_prog_bar');
        const pctEl = document.getElementById('_prog_pct');
        if (bar)   bar.style.width = pct+'%';
        if (pctEl) pctEl.textContent = pct+'%';
        if (i > 0) {
          const prev = document.getElementById(`_st_${i-1}`);
          if (prev) { prev.style.opacity='0.45'; prev.style.transform='scale(1)'; }
        }
        // on last stage: keep pulsing the label so it doesn't feel stuck
        if (i === stages.length - 1) {
          const dots = ['Finalising verdict.', 'Finalising verdict..', 'Finalising verdict...'];
          let di = 0;
          const pulse = setInterval(() => {
            const lbl = document.getElementById('_proc_label');
            if (!lbl) { clearInterval(pulse); return; }
            lbl.textContent = dots[di++ % dots.length];
          }, 500);
          resultsCol._stopPulse = () => clearInterval(pulse);
        }
      }, elapsed);
      stageTimers.push(t);
      elapsed += st.ms;
    });
    resultsCol._stopStages = () => stageTimers.forEach(clearTimeout);
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      const imgUrl = previewWrap ? previewWrap.querySelector('img')?.src : null;
      showProcessingAnimation(imgUrl);

      // update button
      analyzeBtn.innerHTML = `<span style="display:inline-block;animation:spin 1s linear infinite">⏳</span>&nbsp;Analysing...`;
      analyzeBtn.disabled = true;

      const fd = new FormData();
      fd.append('image', selectedFile);

      try {
        const res  = await fetch('/analyze', { method: 'POST', body: fd });
        const data = await res.json();
        // finish progress bar
        const bar = document.getElementById('_prog_bar');
        const pctEl = document.getElementById('_prog_pct');
        const lastStage = document.getElementById(`_st_${stages.length-1}`);
        if (bar) bar.style.width = '100%';
        if (pctEl) pctEl.textContent = '100%';
        if (lastStage) { lastStage.style.opacity='1'; lastStage.style.transform='scale(1.1)'; }
        // stop particles and pulse
        const rc = document.querySelector('.xl\\\\:col-span-7');
        if (rc && rc._stopParticles) rc._stopParticles();
        if (rc && rc._stopPulse) rc._stopPulse();
        // short pause so user sees 100% before results
        await new Promise(r => setTimeout(r, 300));
        if (data.error) { alert('Error: ' + data.error); return; }
        renderResults(data);
      } catch (err) {
        alert('Request failed: ' + err);
      } finally {
        analyzeBtn.innerHTML = '🔍&nbsp;Analyze Card';
        analyzeBtn.disabled = false;
      }
    });
  }

  // inject spin keyframe + lightbox once
  if (!document.getElementById('_spin_style')) {
    const s = document.createElement('style');
    s.id = '_spin_style';
    s.textContent = `
      @keyframes spin { to { transform: rotate(360deg); } }
      #_lightbox { display:none; position:fixed; inset:0; z-index:9999; background:rgba(0,0,0,0.85);
        align-items:center; justify-content:center; cursor:zoom-out; }
      #_lightbox.open { display:flex; }
      #_lightbox img { max-width:90vw; max-height:90vh; object-fit:contain; border-radius:12px;
        box-shadow:0 0 60px rgba(128,131,255,0.3); }
    `;
    document.head.appendChild(s);

    // lightbox element
    const lb = document.createElement('div');
    lb.id = '_lightbox';
    lb.innerHTML = '<img id="_lightbox_img" src="" />';
    lb.addEventListener('click', () => lb.classList.remove('open'));
    document.body.appendChild(lb);
  }

  window._openLightbox = function(src) {
    document.getElementById('_lightbox_img').src = src;
    document.getElementById('_lightbox').classList.add('open');
  };

  // ── render results ─────────────────────────────────────────────────────────
  function renderResults(d) {
    // load the output page HTML and inject data
    const verdict     = d.verdict;       // "Genuine" | "Suspicious" | "Fake"
    const fraudScore  = Math.round(d.fraud_score * 100);
    const confidence  = Math.round(d.confidence * 100);
    const verhoeff    = d.verhoeff || {};
    const consistency = d.consistency || {};
    const tampering   = d.tampering || {};
    const reasons     = d.reasons || [];
    const annotated   = d.annotated_image;

    const verdictColor = { Genuine: '#4ade80', Suspicious: '#facc15', Fake: '#f87171' }[verdict] || '#8083ff';
    const verdictIcon  = { Genuine: '✓', Suspicious: '⚠', Fake: '✗' }[verdict] || '?';
    const verdictBorder= { Genuine: 'border-emerald-500/40', Suspicious: 'border-yellow-500/40', Fake: 'border-red-500/40' }[verdict] || '';
    const verdictGlow  = { Genuine: 'glow-green', Suspicious: '', Fake: '' }[verdict] || '';

    const qrAvail  = consistency.qr_available;
    const qrMatch  = Math.round(consistency.qr_match_score || 0);
    const qrFmt    = consistency.qr_format || 'unknown';
    const qrError  = consistency.qr_error || '';
    const qrFound  = qrAvail || (qrError && !qrError.includes('No QR code detected'));

    // Badge: ✓ good match | ⚠ not found or partial | ✗ found but not matching
    const qrBadge  = !qrAvail
      ? '⚠'
      : qrMatch >= 75 ? '✓' : qrMatch >= 50 ? '⚠' : '✗';
    const qrColor  = !qrAvail
      ? '#facc15'
      : qrMatch >= 75 ? '#4ade80' : qrMatch >= 50 ? '#facc15' : '#f87171';
    const qrLabel  = !qrAvail
      ? `Not detected — ${qrError || 'no QR found'}`
      : `${qrMatch}% match — ${qrFmt} format`;

    const vValid   = verhoeff.valid;
    const vColor   = vValid ? '#4ade80' : '#f87171';
    const vBadge   = vValid ? '✓' : '✗';
    const vLabel   = vValid ? 'Valid checksum — OK' : `Failed — ${verhoeff.reason || ''}`;

    const tLabel   = tampering.label || 'unknown';
    const tConf    = Math.round((tampering.confidence || 0) * 100);
    const tColor   = tLabel === 'real' ? '#4ade80' : tLabel === 'fake' ? '#f87171' : '#facc15';
    const tBadge   = tLabel === 'real' ? '✓' : tLabel === 'fake' ? '✗' : '⚠';
    const tText    = tLabel === 'real' ? `Clean (${tConf}%)` : tLabel === 'fake' ? `Tampered (${tConf}%)` : `Suspicious (${tConf}%)`;

    const forensics = tampering.forensics || {};
    const elaVal    = (forensics.ela_mean || 0).toFixed(2);
    const elaFlag   = forensics.ela_flagged;
    const noiseVal  = (forensics.noise_cv || 0).toFixed(3);
    const noiseFlag = forensics.noise_flagged;
    const sharpVal  = (forensics.sharpness_cv || 0).toFixed(3);
    const sharpFlag = forensics.sharpness_flagged;
    const flags     = forensics.flags_triggered || 0;
    const flagColor = flags >= 2 ? '#f87171' : flags === 1 ? '#facc15' : '#4ade80';

    const photoCmp    = consistency.photo_comparison || {};
    const photoDec    = photoCmp.decision || 'UNAVAILABLE';
    const photoSSIM   = photoCmp.ssim != null ? (photoCmp.ssim * 100).toFixed(1) + '%' : '—';
    const photoColor  = photoDec === 'MATCH' ? '#4ade80' : photoDec === 'SUSPICIOUS' ? '#facc15' : photoDec === 'NO_MATCH' ? '#f87171' : '#9ca3af';
    const photoBadge  = photoDec === 'MATCH' ? '✓' : photoDec === 'SUSPICIOUS' ? '⚠' : photoDec === 'NO_MATCH' ? '✗' : '—';
    const photoRow = `<div class="px-6 py-4 flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${photoColor}22;color:${photoColor}">${photoBadge}</div>
        <div><div class="text-sm font-medium">Face Photo Match</div><div class="text-xs text-on-surface-variant">${photoDec === 'UNAVAILABLE' ? 'No QR photo available' : photoDec + ' — SSIM ' + photoSSIM}</div></div>
      </div>
    </div>`;

    const findingsHtml = reasons.map(r =>
      `<li class="flex items-start space-x-3">
        <div class="mt-1 w-2 h-2 rounded-full flex-shrink-0" style="background:${verdictColor};box-shadow:0 0 8px ${verdictColor}"></div>
        <span class="text-sm text-on-surface-variant">${r}</span>
      </li>`
    ).join('');

    // ── QR parsed data ────────────────────────────────────────────────────────
    const qrParsed = consistency.qr_parsed_data || null;
    const qrComp   = consistency.qr_comparison || {};
    function qrRow(label, field, fallback) {
      const val   = (qrParsed && qrParsed[field] != null) ? qrParsed[field] : (fallback || '—');
      const comp  = qrComp[field];
      const match = comp ? comp.match : null;
      const dot   = match === false
        ? `<span style="color:#f87171" title="Mismatch with OCR">✗</span>`
        : match === true
          ? `<span style="color:#4ade80" title="Matches OCR">✓</span>`
          : '';
      return `<div class="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
        <span class="text-xs text-on-surface-variant uppercase tracking-wider">${label}</span>
        <span class="text-xs font-semibold text-right max-w-[60%] truncate">${val} ${dot}</span>
      </div>`;
    }
    // UID row — secure format stores only last 4 digits
    const last4Val  = qrParsed && qrParsed['last4'];
    const last4Comp = qrComp['last4'];
    const last4Match = last4Comp ? last4Comp.match : null;
    const last4Dot  = last4Match === false
      ? `<span style="color:#f87171" title="Mismatch with OCR">✗</span>`
      : last4Match === true
        ? `<span style="color:#4ade80" title="Matches OCR">✓</span>`
        : '';
    const uidDisplay = qrParsed && qrParsed['uid']
      ? qrParsed['uid']
      : last4Val
        ? `xxxx xxxx ${last4Val} ${last4Dot}`
        : '<span class="italic text-on-surface-variant/50">Not stored (secure QR)</span>';
    const uidRow = `<div class="flex justify-between items-center py-2 border-b border-white/5">
      <span class="text-xs text-on-surface-variant uppercase tracking-wider">UID / Last 4</span>
      <span class="text-xs font-semibold text-right max-w-[60%]">${uidDisplay}</span>
    </div>`;
    const qrDataHtml = !qrAvail
      ? `<div class="flex items-center gap-3 p-4 rounded-xl" style="background:#facc1511;border:1px solid #facc1533">
           <span style="font-size:1.5rem">⚠️</span>
           <div>
             <div class="text-sm font-bold" style="color:#facc15">QR Code Not Detected</div>
             <div class="text-xs text-on-surface-variant mt-1">${consistency.qr_error || 'Could not find or decode a QR code in this image. This makes the card suspicious.'}</div>
           </div>
         </div>`
      : `<div class="text-[10px] font-bold text-on-surface-variant/50 uppercase tracking-widest mb-3">
           Format: ${qrFmt} &nbsp;·&nbsp; ${qrMatch}% field match
         </div>
         ${qrRow('Name',           'name',   null)}
         ${qrRow('Date of Birth',  'dob',    null)}
         ${uidRow}
         ${qrRow('Gender',         'gender', null)}
         ${qrRow('Address',        'address',null)}
         ${(qrParsed && qrParsed.photo_b64) ? `
         <div class="mt-4">
           <div class="text-xs text-on-surface-variant uppercase tracking-wider mb-2">Photo (from QR)</div>
           <img src="data:image/jpeg;base64,${qrParsed.photo_b64}"
                class="w-24 h-28 object-cover rounded-lg border border-white/10 cursor-zoom-in"
                onclick="_openLightbox(this.src)" title="Click to enlarge" />
         </div>` : `
         <div class="mt-3 text-[9px] text-on-surface-variant/40 italic">
           📷 No photo embedded — physical cards omit it (only eAadhaar digital QR contains photo)
         </div>`}
         <p class="text-[9px] text-on-surface-variant/40 mt-3 italic">✓ = matches OCR &nbsp; ✗ = mismatch with OCR &nbsp;·&nbsp; Secure QR does not store full UID (UIDAI privacy policy)</p>`;    const annotatedHtml = annotated
      ? `<img src="data:image/jpeg;base64,${annotated}" class="w-full h-auto rounded-lg" style="display:block;" />`
      : '<div class="text-on-surface-variant text-xs text-center p-4">No annotated image</div>';

    // build the right-column results HTML
    const resultsHtml = `
      <div class="stagger-load flex flex-col gap-6 w-full">
        <div class="border-2 ${verdictBorder} relative overflow-hidden glass-card rounded-xl p-6" style="background:var(--card-bg, #1c1f29); animation-delay: 0.1s;">
          <div class="absolute inset-0 opacity-[0.15] mix-blend-overlay ${verdictGlow}"></div>
          <div class="absolute top-0 right-0 p-4 relative z-10">
            <div class="w-12 h-12 rounded-lg flex items-center justify-center glow-effect" style="background:${verdictColor}22; box-shadow: 0 0 20px ${verdictColor}40">
              <span style="font-size:2rem;color:${verdictColor}">${verdictIcon}</span>
            </div>
          </div>
          <h2 class="text-4xl font-black tracking-tighter mb-1 uppercase relative z-10" style="color:${verdictColor}; text-shadow: 0 0 30px ${verdictColor}80">${verdictIcon} ${verdict.toUpperCase()}</h2>
          <p class="text-on-surface-variant text-sm font-medium relative z-10">${reasons[0] || ''}</p>
        </div>

        <div class="grid grid-cols-2 gap-4" style="animation-delay: 0.2s;">
          <div class="glass-card p-4 rounded-xl relative overflow-hidden">
            <div class="flex justify-between items-end mb-2 relative z-10">
              <span class="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">Fraud Score</span>
              <span class="text-xl font-black" style="color:${verdictColor}">${fraudScore}%</span>
            </div>
            <div class="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden relative z-10">
              <div class="h-full rounded-full transition-all duration-1000 ease-out" style="width:0; background:${verdictColor}; box-shadow:0 0 10px ${verdictColor}" data-final-width="${fraudScore}%"></div>
            </div>
          </div>
          <div class="glass-card p-4 rounded-xl relative overflow-hidden">
            <div class="flex justify-between items-end mb-2 relative z-10">
              <span class="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">Confidence</span>
              <span class="text-xl font-black text-[#8083ff]">${confidence}%</span>
            </div>
            <div class="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden relative z-10">
              <div class="h-full rounded-full transition-all duration-1000 ease-out" style="width:0; background:#8083ff; box-shadow:0 0 10px #8083ff" data-final-width="${confidence}%"></div>
            </div>
          </div>
        </div>

        <div class="glass-card rounded-xl overflow-hidden" style="animation-delay: 0.3s;">
        <div class="px-6 py-4 border-b border-white/5">
          <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant">Verification Checks</h3>
        </div>
        <div class="divide-y divide-white/5">
          <div class="px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${vColor}22;color:${vColor}">${vBadge}</div>
              <div><div class="text-sm font-medium">Verhoeff Checksum</div><div class="text-xs text-on-surface-variant">${vLabel}</div></div>
            </div>
          </div>
          <div class="px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${qrColor}22;color:${qrColor}">${qrBadge}</div>
              <div><div class="text-sm font-medium">QR Consistency</div><div class="text-xs text-on-surface-variant">${qrLabel}</div></div>
            </div>
          </div>
          <div class="px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${tColor}22;color:${tColor}">${tBadge}</div>
              <div><div class="text-sm font-medium">Tampering Detection</div><div class="text-xs text-on-surface-variant">${tText}</div></div>
            </div>
          </div>
          ${photoRow}
        </div>
      </div>

      <div class="glass-card rounded-xl overflow-hidden p-6" style="animation-delay: 0.4s; background:var(--card-bg, #1c1f29)">
        <div class="flex justify-between items-center mb-4">
          <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant">Digital Forensics</h3>
          <span class="text-xs font-bold" style="color:${flagColor}">${flags}/3 flags triggered</span>
        </div>
        <div class="grid grid-cols-3 gap-4">
          <div class="border border-white/5 bg-surface-container-highest p-4 rounded-xl text-center">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-2">ELA Score</div>
            <div class="text-lg font-black" style="color:${elaFlag?'#f87171':'#4ade80'}">${elaVal}/255</div>
            <div class="text-[9px] text-on-surface-variant/60 mt-1">${elaFlag?'⚠ Re-compression':'Normal'}</div>
          </div>
          <div class="border border-white/5 bg-surface-container-highest p-4 rounded-xl text-center">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-2">Noise CV</div>
            <div class="text-lg font-black" style="color:${noiseFlag?'#f87171':'#4ade80'}">${noiseVal}</div>
            <div class="text-[9px] text-on-surface-variant/60 mt-1">${noiseFlag?'⚠ Uneven':'Uniform'}</div>
          </div>
          <div class="border border-white/5 bg-surface-container-highest p-4 rounded-xl text-center">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-2">Sharpness CV</div>
            <div class="text-lg font-black" style="color:${sharpFlag?'#f87171':'#4ade80'}">${sharpVal}</div>
            <div class="text-[9px] text-on-surface-variant/60 mt-1">${sharpFlag?'⚠ Mismatch':'Consistent'}</div>
          </div>
        </div>
      </div>

      <div class="glass-card p-6 rounded-xl" style="animation-delay: 0.5s; background:var(--card-bg, #1c1f29)">
        <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant mb-4">QR Code Data</h3>
        ${qrDataHtml}
      </div>

      <div class="glass-card p-6 rounded-xl" style="animation-delay: 0.6s; background:var(--card-bg, #1c1f29)">
        <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant mb-4">Key Findings</h3>
        <ul class="space-y-3">${findingsHtml}</ul>
      </div>

      <div class="glass-card p-4 rounded-xl" style="animation-delay: 0.7s; background:var(--card-bg, #1c1f29)">
        <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant mb-3">Detected Regions</h3>
        <div class="rounded-lg overflow-hidden cursor-zoom-in relative group" onclick="_openLightbox(this.querySelector('img').src)">
          ${annotatedHtml}
          <div class="absolute inset-0 bg-[#8083ff]/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col items-center justify-center">
            <span style="font-size:2rem;color:white">🔍</span>
          </div>
        </div>
        <p class="text-[10px] text-on-surface-variant/50 text-center mt-2">Click to enlarge</p>
      </div>
      </div>
    `;

    // find or create the right column and replace its content
    let rightCol = document.getElementById('results-panel');
    if (!rightCol) {
      // first run — find the idle placeholder and replace it
      const idle = document.querySelector('.xl\\\\:col-span-7, .lg\\\\:col-span-7');
      if (idle) {
        idle.id = 'results-panel';
        rightCol = idle;
      }
    }
    if (rightCol) {
      rightCol.innerHTML = resultsHtml;
      
      // trigger progress bar animations
      setTimeout(() => {
        const bars = rightCol.querySelectorAll('[data-final-width]');
        bars.forEach(b => {
          b.style.width = b.getAttribute('data-final-width');
        });
      }, 50);
    }
  }
})();
</script>
"""


if __name__ == "__main__":
    app.run(debug=False, port=5000)
