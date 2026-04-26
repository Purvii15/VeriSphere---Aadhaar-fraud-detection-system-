"""
Flask web server for Aadhaar Fraud Detection — VeriSphere
Serves the frontend and exposes /analyze and /chat API endpoints.
"""
import os
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

# Load .env file if present
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

from aadhaar_pipeline.pipeline import run_pipeline
from aadhaar_pipeline.detector import load_model
from aadhaar_pipeline.tampering import load_tampering_model

# ── AI setup — Gemini primary, Google Gemma (HF) fallback ─────────────────────
from google import genai as google_genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
_gemini_client = None

def get_gemini():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        _gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

def _call_gemma_hf(prompt: str) -> str:
    """Call a free HF model via Serverless Inference API (no billing, no token needed)."""
    from huggingface_hub import InferenceClient
    # Use chat_completion API — more reliable across HF providers
    client = InferenceClient(
        model="meta-llama/Llama-3.2-1B-Instruct",
        token=HF_TOKEN or None
    )
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content or "No response generated."

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

<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>VeriSphere | Aadhaar Fraud Detection</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%238083ff' d='M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z'/%3E%3C/svg%3E"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<script id="tailwind-config">
      tailwind.config = {
        darkMode: "class",
        theme: {
          extend: {
            colors: {
              "on-surface-variant": "#c7c4d7",
              "surface-dim": "#10131c",
              "on-primary-container": "#0d0096",
              "tertiary-container": "#707787",
              "on-tertiary-fixed": "#151c29",
              "tertiary-fixed": "#dce2f5",
              "on-error-container": "#ffdad6",
              "primary": "#c0c1ff",
              "error": "#ffb4ab",
              "primary-fixed-dim": "#c0c1ff",
              "on-primary-fixed-variant": "#2f2ebe",
              "on-error": "#690005",
              "on-secondary-fixed-variant": "#5516be",
              "secondary-fixed": "#e9ddff",
              "tertiary-fixed-dim": "#c0c6d9",
              "secondary": "#d0bcff",
              "on-secondary": "#3c0091",
              "surface-container-low": "#181b25",
              "surface-container-high": "#272a34",
              "secondary-container": "#571bc1",
              "on-primary-fixed": "#07006c",
              "tertiary": "#c0c6d9",
              "surface-container-lowest": "#0b0e17",
              "surface-bright": "#363943",
              "primary-fixed": "#e1e0ff",
              "outline": "#908fa0",
              "on-secondary-fixed": "#23005c",
              "on-secondary-container": "#c4abff",
              "inverse-primary": "#494bd6",
              "inverse-on-surface": "#2d303a",
              "inverse-surface": "#e0e2ef",
              "background": "#10131c",
              "surface-container-highest": "#32343f",
              "secondary-fixed-dim": "#d0bcff",
              "on-surface": "#e0e2ef",
              "on-primary": "#1000a9",
              "error-container": "#93000a",
              "on-tertiary-fixed-variant": "#404756",
              "on-background": "#e0e2ef",
              "outline-variant": "#464554",
              "surface-variant": "#32343f",
              "surface-tint": "#c0c1ff",
              "on-tertiary": "#2a303f",
              "on-tertiary-container": "#010511",
              "surface-container": "#1c1f29",
              "primary-container": "#8083ff",
              "surface": "#10131c"
            },
            fontFamily: {
              "headline": ["Inter"],
              "body": ["Inter"],
              "label": ["Inter"]
            },
            borderRadius: {"DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px"},
          },
        },
      }
    </script>
<style>
      body {
        background-color: #10131c;
        color: #e0e2ef;
        font-family: 'Inter', sans-serif;
      }
      .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
      }
      .glass-card {
        background-color: #1c1f29;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.04);
      }
      .glow-border:focus-within {
        border-color: #8083ff;
        box-shadow: 0 0 15px rgba(128, 131, 255, 0.2);
      }
      /* Coming soon tooltip */
      [data-soon] { position: relative; }
      [data-soon]::after {
        content: 'Coming soon';
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        background: #1c1f29;
        color: #c7c4d7;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.08);
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.15s;
        z-index: 100;
      }
      [data-soon]:hover::after { opacity: 1; }
    </style>
</head>
<body class="antialiased" style="overflow:hidden;height:100vh;display:flex;flex-direction:column;">
<!-- TopNavBar -->
<header class="bg-[#10131c] dark:bg-[#10131c] docked full-width top-0 z-50 flex justify-between items-center w-full px-6 py-4 border-b border-white/[0.06] shadow-[0_0_40px_rgba(99,102,241,0.08)] font-['Inter'] antialiased tracking-tight">
<div class="flex items-center gap-4">
<span class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-[#8083ff] to-[#571bc1]">VeriSphere</span>
</div>
<div class="hidden md:flex items-center gap-8">
<a class="text-[#8083ff] border-b-2 border-[#8083ff] pb-1 text-sm font-medium" href="#">Dashboard</a>
<a class="text-[#c7c4d7] hover:text-[#8083ff] transition-colors text-sm font-medium" href="/qr-decode">QR Decode</a>
<a class="text-[#c7c4d7] hover:text-[#8083ff] transition-colors text-sm font-medium" href="#" data-soon>Forensics</a>
<a class="text-[#c7c4d7] hover:text-[#8083ff] transition-colors text-sm font-medium" href="#" data-soon>Archive</a>
<a class="text-[#c7c4d7] hover:text-[#8083ff] transition-colors text-sm font-medium" href="#" data-soon>Network</a>
</div>
<div class="flex items-center gap-4">
<button class="material-symbols-outlined text-[#c7c4d7] hover:bg-[#1c1f29] transition-all duration-300 p-2 rounded-lg active:scale-95 duration-200" data-soon>security</button>
<button class="material-symbols-outlined text-[#c7c4d7] hover:bg-[#1c1f29] transition-all duration-300 p-2 rounded-lg active:scale-95 duration-200" data-soon>history</button>
<button class="material-symbols-outlined text-[#c7c4d7] hover:bg-[#1c1f29] transition-all duration-300 p-2 rounded-lg active:scale-95 duration-200" data-soon>settings</button>
<div class="h-8 w-8 rounded-full overflow-hidden border border-white/10 ml-2">
<img alt="User Profile" class="w-full h-full object-cover" data-alt="close-up portrait of a professional male system administrator with a focused expression in a dimly lit high-tech environment" src="https://lh3.googleusercontent.com/aida-public/AB6AXuB_6_fSfUddRuXweDm5R77FHmELXJBkbucKqKseiwrNeJ1-U1lENxnhVcOri7nBVeaqqfAtW4IpTnuPkNZj8dIM2rP0WOWwWTRaocgEcjDOixi3iFzkBsjNYpchZ7_htSziCraBWGYCgT9uQ-HR5_zX8GqO19Db14Z6kIXXGw7UNxw13EU4cJ37UB_xVKhLhG-4N-nEzCJS3TlQHvACBunBg4zDMAT-Imw4SE-Li0ahLKuBTmT7O-klFv8OgYWy9_eKD9x0TonVPpOg"/>
</div>
</div>
</header>
<main class="flex flex-1 overflow-hidden" style="height:calc(100vh - 72px)">
<!-- SideNavBar (Hidden on small screens) -->
<aside class="bg-[#181b25] dark:bg-[#181b25] h-full w-64 border-r border-white/[0.06] flex flex-col py-8 hidden lg:flex" style="flex-shrink:0;overflow-y:auto;">
<div class="px-6 mb-8">
<div class="text-lg font-black text-[#8083ff] tracking-tighter">VeriSphere</div>
<div class="text-[10px] text-[#c7c4d7] uppercase tracking-widest font-bold opacity-60 mt-1">Aadhaar Verification</div>
</div>
<nav class="flex-1">
<div class="bg-gradient-to-r from-[#8083ff]/10 to-transparent text-[#8083ff] border-l-4 border-[#8083ff] px-6 py-3 flex items-center gap-3 font-['Inter'] text-sm font-medium tracking-wide">
<span class="material-symbols-outlined">grid_view</span>
<span>Dashboard</span>
</div>
<a href="/qr-decode" class="text-[#c7c4d7] px-6 py-3 hover:bg-white/[0.02] hover:text-[#8083ff] transition-all flex items-center gap-3 font-['Inter'] text-sm font-medium tracking-wide">
<span class="material-symbols-outlined">qr_code_scanner</span>
<span>QR Decode</span>
</a>
<div class="text-[#c7c4d7] px-6 py-3 hover:bg-white/[0.02] hover:text-[#8083ff] transition-all flex items-center gap-3 font-['Inter'] text-sm font-medium tracking-wide cursor-pointer" data-soon>
<span class="material-symbols-outlined">smart_card_reader</span>
<span>Forensics</span>
</div>
<div class="text-[#c7c4d7] px-6 py-3 hover:bg-white/[0.02] hover:text-[#8083ff] transition-all flex items-center gap-3 font-['Inter'] text-sm font-medium tracking-wide cursor-pointer" data-soon>
<span class="material-symbols-outlined">inventory_2</span>
<span>Archive</span>
</div>
<div class="text-[#c7c4d7] px-6 py-3 hover:bg-white/[0.02] hover:text-[#8083ff] transition-all flex items-center gap-3 font-['Inter'] text-sm font-medium tracking-wide cursor-pointer" data-soon>
<span class="material-symbols-outlined">hub</span>
<span>Network</span>
</div>
<div class="text-[#c7c4d7] px-6 py-3 hover:bg-white/[0.02] hover:text-[#8083ff] transition-all flex items-center gap-3 font-['Inter'] text-sm font-medium tracking-wide cursor-pointer" data-soon>
<span class="material-symbols-outlined">shield</span>
<span>Security</span>
</div>
</nav>
<div class="px-6 mt-auto">
<button onclick="window.location.reload()" class="w-full bg-gradient-to-r from-[#8083ff] to-[#571bc1] text-white font-bold py-3 rounded-xl active:scale-95 transition-all shadow-[0_0_20px_rgba(128,131,255,0.2)] cursor-pointer">
                    New Analysis
                </button>
<button onclick="toggleChat()" class="w-full mt-3 border border-[#8083ff]/40 text-[#8083ff] font-bold py-3 rounded-xl active:scale-95 transition-all hover:bg-[#8083ff]/10 cursor-pointer text-sm">
                    ✨ Ask Gemini AI
                </button>
</div>
</aside>
<!-- Main Content Canvas -->
<section class="flex-1 overflow-y-auto bg-[#10131c] p-4">
<div class="w-full grid grid-cols-1 xl:grid-cols-12 gap-6 h-full">
<!-- Left Column (Upload Panel) -->
<div class="xl:col-span-5 flex flex-col gap-4">
<div class="flex justify-between items-end pb-4 border-b border-white/[0.06]">
<div>
<div class="flex items-center gap-2 mb-1">
<span class="text-2xl">🛡️</span>
<h1 class="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-[#8083ff] to-[#571bc1] tracking-tight">Aadhaar Fraud Detection</h1>
</div>
<p class="text-xs text-on-surface-variant font-medium uppercase tracking-widest opacity-60">System Ready for Scan</p>
</div>
<div class="text-[10px] text-on-surface-variant font-mono text-right hidden md:block">
                            YOLOv8 · EasyOCR · Verhoeff · QR · Forensics
                        </div>
</div>
<!-- Upload Zone -->
<div class="glass-card rounded-xl p-8 border-2 border-dashed border-[#8083ff]/30 flex flex-col items-center justify-center text-center group hover:border-[#8083ff]/60 transition-all cursor-pointer">
<div class="w-16 h-16 rounded-full bg-surface-container-highest flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
<span class="material-symbols-outlined text-[#8083ff] text-3xl">cloud_upload</span>
</div>
<h3 class="text-lg font-bold mb-2">Drop your Aadhaar Card here</h3>
<p class="text-sm text-on-surface-variant mb-4">Supports JPG, PNG, or WEBP (Max 10MB)</p>
<div class="flex gap-2">
<span class="px-3 py-1 bg-surface-container-low text-[10px] font-bold tracking-tighter rounded border border-white/5 uppercase">Front Side</span>
<span class="px-3 py-1 bg-surface-container-low text-[10px] font-bold tracking-tighter rounded border border-white/5 uppercase">Back Side</span>
</div>
</div>
<!-- Image Preview Placeholder -->
<div class="glass-card rounded-xl aspect-[1.58/1] flex items-center justify-center relative group">
<div class="absolute inset-0 bg-gradient-to-br from-primary-container/5 to-transparent"></div>
<div class="flex flex-col items-center opacity-40 group-hover:opacity-60 transition-opacity">
<span class="material-symbols-outlined text-5xl mb-2">image</span>
<p class="text-xs uppercase tracking-[0.2em] font-bold">Waiting for preview</p>
</div>
<!-- Faint grid lines overlay -->
<div class="absolute inset-0 opacity-[0.03] pointer-events-none" style="background-image: linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px); background-size: 20px 20px;"></div>
</div>
<!-- Analyze Button -->
<button class="w-full py-5 rounded-xl bg-surface-container-highest text-on-surface-variant font-black text-lg flex items-center justify-center gap-3 cursor-not-allowed opacity-50 border border-white/5" disabled="">
<span class="material-symbols-outlined">search</span>
                        Analyze Card
                    </button>
<!-- Info Card: Pipeline -->
<div class="glass-card rounded-xl p-6">
<h4 class="text-xs font-black uppercase tracking-widest text-on-surface-variant mb-6 border-b border-white/5 pb-2">Analysis Pipeline</h4>
<div class="space-y-6">
<div class="flex gap-4">
<div class="w-1.5 h-12 bg-[#8083ff] rounded-full shadow-[0_0_8px_#8083ff]"></div>
<div>
<p class="text-xs font-bold text-[#8083ff] uppercase mb-1 tracking-wider">YOLOv8 Detection</p>
<p class="text-xs text-on-surface-variant leading-relaxed">Spatial analysis to locate ID card boundaries and orientation.</p>
</div>
</div>
<div class="flex gap-4">
<div class="w-1.5 h-12 bg-[#60a5fa] rounded-full shadow-[0_0_8px_#60a5fa]"></div>
<div>
<p class="text-xs font-bold text-[#60a5fa] uppercase mb-1 tracking-wider">EasyOCR Text Extraction</p>
<p class="text-xs text-on-surface-variant leading-relaxed">Neural network extraction of demographic data and card ID.</p>
</div>
</div>
<div class="flex gap-4">
<div class="w-1.5 h-12 bg-[#4ade80] rounded-full shadow-[0_0_8px_#4ade80]"></div>
<div>
<p class="text-xs font-bold text-[#4ade80] uppercase mb-1 tracking-wider">Verhoeff Algorithm</p>
<p class="text-xs text-on-surface-variant leading-relaxed">Mathematical checksum validation of the 12-digit Aadhaar number.</p>
</div>
</div>
<div class="flex gap-4">
<div class="w-1.5 h-12 bg-[#facc15] rounded-full shadow-[0_0_8px_#facc15]"></div>
<div>
<p class="text-xs font-bold text-[#facc15] uppercase mb-1 tracking-wider">QR Cross-Check</p>
<p class="text-xs text-on-surface-variant leading-relaxed">Comparison between OCR extracted data and embedded QR code data.</p>
</div>
</div>
<div class="flex gap-4">
<div class="w-1.5 h-12 bg-[#f87171] rounded-full shadow-[0_0_8px_#f87171]"></div>
<div>
<p class="text-xs font-bold text-[#f87171] uppercase mb-1 tracking-wider">Deep Forensics</p>
<p class="text-xs text-on-surface-variant leading-relaxed">Pixel-level scrutiny for manipulation, cloned photos, or font mismatches.</p>
</div>
</div>
</div>
</div>
</div>
<!-- Right Column (Results Panel - Initial State) -->
<div class="xl:col-span-7 flex flex-col" style="min-height:calc(100vh - 160px)">
<div class="glass-card rounded-xl flex-1 flex flex-col items-center justify-center p-12 text-center border-white/[0.04]">
<div class="relative mb-8">
<!-- Background glow for the icon -->
<div class="absolute inset-0 bg-[#8083ff]/10 blur-[60px] rounded-full"></div>
<span class="material-symbols-outlined text-[120px] text-[#8083ff]/20 relative z-10" style="font-variation-settings: 'wght' 200;">shield_with_heart</span>
</div>
<h2 class="text-3xl font-black mb-4 tracking-tight">System Idle</h2>
<p class="text-on-surface-variant max-w-md mx-auto leading-relaxed mb-10">
                            Upload a card image to begin the multi-stage cryptographic and forensic verification process. Our AI will analyze security features and metadata in real-time.
                        </p>
<div class="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-2xl">
<div class="bg-surface-container-low p-6 rounded-xl border border-white/5 flex flex-col items-center">
<span class="material-symbols-outlined text-[#8083ff] mb-2">speed</span>
<span class="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant opacity-60">Avg Time</span>
<span class="text-lg font-black mt-1">1.2s</span>
</div>
<div class="bg-surface-container-low p-6 rounded-xl border border-white/5 flex flex-col items-center">
<span class="material-symbols-outlined text-[#8083ff] mb-2">verified</span>
<span class="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant opacity-60">Confidence</span>
<span class="text-lg font-black mt-1">99.8%</span>
</div>
<div class="bg-surface-container-low p-6 rounded-xl border border-white/5 flex flex-col items-center">
<span class="material-symbols-outlined text-[#8083ff] mb-2">query_stats</span>
<span class="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant opacity-60">Database</span>
<span class="text-lg font-black mt-1">Local</span>
</div>
</div>
</div>
<!-- Footer Metadata -->
<div class="mt-6 flex justify-between items-center px-4">
<div class="flex items-center gap-2">
<div class="w-2 h-2 rounded-full bg-[#4ade80] animate-pulse shadow-[0_0_8px_#4ade80]"></div>
<span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Core Engine Online</span>
</div>
<span class="text-[10px] font-mono text-on-surface-variant opacity-40">SOVEREIGN_SENTINEL_BUILD_88.42</span>
</div>
</div>
</div>
</section>
</main>
<!-- BottomNavBar (Mobile Only) -->
<nav class="md:hidden fixed bottom-0 left-0 right-0 bg-[#10131c] px-6 py-4 flex justify-between items-center border-t border-white/10 z-50">
<button class="flex flex-col items-center gap-1 text-[#8083ff]">
<span class="material-symbols-outlined">grid_view</span>
<span class="text-[10px] font-bold uppercase tracking-tighter">Dash</span>
</button>
<button class="flex flex-col items-center gap-1 text-[#c7c4d7]">
<span class="material-symbols-outlined">smart_card_reader</span>
<span class="text-[10px] font-bold uppercase tracking-tighter">Scan</span>
</button>
<div class="relative -top-6">
<button class="bg-gradient-to-r from-[#8083ff] to-[#571bc1] w-14 h-14 rounded-full flex items-center justify-center shadow-[0_0_20px_rgba(128,131,255,0.4)] border-4 border-[#10131c]">
<span class="material-symbols-outlined text-white text-3xl">add</span>
</button>
</div>
<button class="flex flex-col items-center gap-1 text-[#c7c4d7]">
<span class="material-symbols-outlined">inventory_2</span>
<span class="text-[10px] font-bold uppercase tracking-tighter">Vault</span>
</button>
<button class="flex flex-col items-center gap-1 text-[#c7c4d7]">
<span class="material-symbols-outlined">settings</span>
<span class="text-[10px] font-bold uppercase tracking-tighter">Set</span>
</button>
</nav>

<!-- ✨ Gemini AI Chat Widget -->
<div id="_chat_fab" onclick="toggleChat()"
  style="position:fixed;bottom:24px;right:24px;z-index:1000;width:56px;height:56px;border-radius:50%;
         background:linear-gradient(135deg,#8083ff,#571bc1);cursor:pointer;
         display:flex;align-items:center;justify-content:center;
         box-shadow:0 4px 24px rgba(128,131,255,0.5);transition:transform 0.2s"
  onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
  <span style="font-size:1.5rem">✨</span>
</div>

<div id="_chat_panel"
  style="display:none;position:fixed;bottom:92px;right:24px;z-index:999;width:360px;
         background:#10131c;border:1px solid rgba(128,131,255,0.3);border-radius:16px;
         box-shadow:0 8px 40px rgba(0,0,0,0.6);overflow:hidden;flex-direction:column">
  <!-- Header -->
  <div style="background:linear-gradient(135deg,#8083ff22,#571bc122);padding:14px 16px;
              border-bottom:1px solid rgba(128,131,255,0.2);display:flex;align-items:center;justify-content:space-between">
    <div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:1.3rem">✨</span>
      <div>
        <div style="font-weight:800;font-size:14px;color:#e0e2ef">VeriSphere AI</div>
        <div style="font-size:10px;color:#8083ff;font-weight:600">Powered by Gemini 1.5 Flash</div>
      </div>
    </div>
    <button onclick="toggleChat()" style="background:none;border:none;color:#9ca3af;cursor:pointer;font-size:18px;line-height:1">✕</button>
  </div>
  <!-- Messages -->
  <div id="_chat_msgs"
    style="height:300px;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth">
    <div style="background:rgba(128,131,255,0.1);border-radius:12px 12px 12px 4px;padding:10px 14px;max-width:85%">
      <p style="font-size:13px;color:#e0e2ef;margin:0">Hi! I'm your Gemini AI assistant. Ask me anything about fraud detection results, Aadhaar verification, or enrollment statistics.</p>
    </div>
  </div>
  <!-- Input -->
  <div style="padding:12px;border-top:1px solid rgba(255,255,255,0.06);display:flex;gap:8px">
    <input id="_chat_input" type="text" placeholder="Ask about fraud results..."
      style="flex:1;background:#1c1f29;border:1px solid rgba(128,131,255,0.3);border-radius:10px;
             padding:9px 12px;font-size:13px;color:#e0e2ef;outline:none"
      onkeydown="if(event.key==='Enter')sendChat()" />
    <button onclick="sendChat()"
      style="background:linear-gradient(135deg,#8083ff,#571bc1);border:none;border-radius:10px;
             padding:9px 14px;color:white;font-weight:700;cursor:pointer;font-size:13px">
      Send
    </button>
  </div>
</div>

<script>
function toggleChat() {
  const p = document.getElementById('_chat_panel');
  p.style.display = p.style.display === 'none' ? 'flex' : 'none';
  if (p.style.display === 'flex') document.getElementById('_chat_input').focus();
}

async function sendChat() {
  const input = document.getElementById('_chat_input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  const msgs = document.getElementById('_chat_msgs');

  // User bubble
  msgs.innerHTML += `
    <div style="background:rgba(128,131,255,0.2);border-radius:12px 12px 4px 12px;padding:10px 14px;max-width:85%;align-self:flex-end;margin-left:auto">
      <p style="font-size:13px;color:#e0e2ef;margin:0">${msg}</p>
    </div>`;

  // Typing indicator
  const typingId = '_typing_' + Date.now();
  msgs.innerHTML += `
    <div id="${typingId}" style="background:rgba(128,131,255,0.08);border-radius:12px 12px 12px 4px;padding:10px 14px;max-width:60%">
      <p style="font-size:13px;color:#9ca3af;margin:0">✨ Thinking...</p>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    document.getElementById(typingId)?.remove();

    const reply = data.reply || data.error || 'Something went wrong.';
    const modelTag = data.model ? `<span style="font-size:9px;color:#8083ff;font-weight:700;display:block;margin-bottom:4px">✨ ${data.model}</span>` : '';
    msgs.innerHTML += `
      <div style="background:rgba(128,131,255,0.08);border-radius:12px 12px 12px 4px;padding:10px 14px;max-width:85%">
        ${modelTag}
        <p style="font-size:13px;color:#e0e2ef;margin:0;white-space:pre-wrap">${reply}</p>
      </div>`;
  } catch(e) {
    document.getElementById(typingId)?.remove();
    msgs.innerHTML += `
      <div style="background:rgba(248,113,113,0.1);border-radius:12px;padding:10px 14px;max-width:85%">
        <p style="font-size:13px;color:#f87171;margin:0">Error: ${e}</p>
      </div>`;
  }
  msgs.scrollTop = msgs.scrollHeight;
}
</script>
</body></html>"""


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


@app.route("/api/chat", methods=["POST"])
def chat():
    """AI assistant — Gemini primary, Google Gemma (HF) fallback."""
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    context  = data.get("context", "")

    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    system = (
        "You are VeriSphere AI, an expert assistant for an Aadhaar card fraud detection system. "
        "Help users understand fraud detection results, Verhoeff checksum, QR cross-verification, "
        "tampering detection, and photo matching. "
        + (f"Last analysis context: {context}. " if context else "") +
        "Be concise and accurate. Politely redirect off-topic questions."
    )
    full_prompt = f"{system}\n\nUser: {user_msg}\nAssistant:"

    # ── 1. Try Gemini ──────────────────────────────────────────────────────────
    client = get_gemini()
    if client:
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=full_prompt
            )
            return jsonify({"reply": resp.text, "model": "Gemini 2.0 Flash"})
        except Exception as e:
            err = str(e)
            # Fall through to HF on quota/rate errors
            if "429" not in err and "RESOURCE_EXHAUSTED" not in err:
                return jsonify({"error": err}), 500
            print(f"Gemini quota hit, falling back to Gemma HF: {err[:80]}")

    # ── 2. Fallback — Google Gemma 2 via HF Inference API ─────────────────────
    if not HF_TOKEN and not GEMINI_API_KEY:
        return jsonify({"error": "No AI API key configured. Set GEMINI_API_KEY or HF_TOKEN."}), 503
    try:
        reply = _call_gemma_hf(full_prompt)
        return jsonify({"reply": reply, "model": "Llama 3.2 (HF Free)"})
    except Exception as e:
        return jsonify({"error": f"Both Gemini and Gemma failed: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    app.run(debug=False, host="0.0.0.0", port=port)
