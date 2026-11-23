import json
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLES_DIR = BASE_DIR / "samples"

pt_path = MODEL_DIR / "iran_char_yolo.pt"
if not pt_path.exists():
    raise FileNotFoundError(f"فایل مدل یافت نشد: {pt_path}")

print("1. در حال تبدیل iran_char_yolo.pt به فرمت ONNX...")
model = YOLO(str(pt_path))
# خروجی ONNX سازگار با مرورگر (بدون نیاز به onnxsim)
model.export(format="onnx", imgsz=416, opset=12, simplify=False)

# ذخیره نام دقیق کلاس‌های مدل برای استفاده در جاوااسکریپت
classes_path = MODEL_DIR / "yolo_classes.json"
classes_path.write_text(json.dumps(model.names, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"کلاس‌ها ذخیره شدند: {classes_path}")

print("2. در حال ساخت لیست خودکار عکس‌های پوشه samples...")
exts = {".jpg", ".jpeg", ".png", ".webp"}
items = []
if SAMPLES_DIR.exists():
    for path in sorted(SAMPLES_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            rel = path.relative_to(SAMPLES_DIR).as_posix()
            parent = path.parent.name.lower()
            category = parent.upper() if parent in ("iran", "us", "eu") else "OTHER"
            items.append({
                "name": path.stem,
                "url": f"./samples/{rel}",
                "category": category,
            })

manifest_path = SAMPLES_DIR / "manifest.json"
manifest_path.write_text(json.dumps({"samples": items}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"تعداد {len(items)} نمونه پلاک در {manifest_path} ثبت شد.")
print("همه چیز آماده است!")