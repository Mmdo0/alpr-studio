from pathlib import Path
import urllib.request

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

LIBS = {
    "tf.min.js": [
        "https://cdnjs.cloudflare.com/ajax/libs/tensorflow/3.21.0/tf.min.js",
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.21.0/dist/tf.min.js",
        "https://unpkg.com/@tensorflow/tfjs@3.21.0/dist/tf.min.js",
    ],
    "teachablemachine-image.min.js": [
        "https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8.6/dist/teachablemachine-image.min.js",
        "https://unpkg.com/@teachablemachine/image@0.8.6/dist/teachablemachine-image.min.js",
    ],
}

for filename, urls in LIBS.items():
    target = STATIC_DIR / filename
    downloaded = False
    for url in urls:
        try:
            print(f"در حال دانلود {filename} از {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = resp.read()
            if len(data) < 5000 or data.lstrip().startswith(b"<"):
                continue  # رد کردن صفحات خطای احتمالی HTML
            target.write_bytes(data)
            print(f"ذخیره شد: {target} ({len(data) // 1024} KB)")
            downloaded = True
            break
        except Exception as e:
            print(f"ناموفق ({e})")
    if not downloaded:
        print(f"خطا: دانلود خودکار {filename} انجام نشد؛ از روش دستی (مرورگر) استفاده کن.")