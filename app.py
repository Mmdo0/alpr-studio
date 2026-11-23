import base64
import json
import math
import re
import time
import traceback
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    YOLO = None
    ULTRALYTICS_OK = False

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "model"

STATIC_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SAMPLES_DIR = BASE_DIR / "samples"
for sub in ("iran", "us", "eu"):
    (SAMPLES_DIR / sub).mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ALPR Pro Platform (PyTorch + YOLOv8)")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else "cpu"

PLATE_DETECTOR_PATH = MODEL_DIR / "plate_detector.pt"
IRAN_CHAR_YOLO_PATH = MODEL_DIR / "iran_char_yolo.pt"

PLATE_DETECTOR = None
IRAN_CHAR_MODEL = None
READERS: dict[str, easyocr.Reader] = {}

if ULTRALYTICS_OK:
    if PLATE_DETECTOR_PATH.exists():
        PLATE_DETECTOR = YOLO(str(PLATE_DETECTOR_PATH))
    if IRAN_CHAR_YOLO_PATH.exists():
        IRAN_CHAR_MODEL = YOLO(str(IRAN_CHAR_YOLO_PATH))

DTYPE_BYTES = {
    "float32": 4, "float16": 2, "int32": 4,
    "uint16": 2, "uint8": 1, "bool": 1, "complex64": 8,
}

# نگاشت جامع نام کلاس‌های مدل‌های مختلف YOLO (انگلیسی و فارسی) به کاراکتر استاندارد
YOLO_LABEL_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "alef": "الف", "aleph": "الف", "a": "الف", "الف": "الف",
    "b": "ب", "be": "ب", "beh": "ب", "ب": "ب",
    "p": "پ", "pe": "پ", "peh": "پ", "پ": "پ",
    "t": "ت", "te": "ت", "teh": "ت", "taxi": "ت", "ت": "ت",
    "se": "ث", "seh": "ث", "the": "ث", "ث": "ث",
    "j": "ج", "jim": "ج", "jeem": "ج", "ج": "ج",
    "d": "د", "dal": "د", "د": "د",
    "z": "ز", "ze": "ز", "zeh": "ز", "ز": "ز",
    "zh": "ژ", "zhe": "ژ", "malol": "ژ", "malolin": "ژ", "wheelchair": "ژ", "disabled": "ژ", "ژ": "ژ",
    "s": "س", "sin": "س", "seen": "س", "س": "س",
    "sh": "ش", "shin": "ش", "sheen": "ش", "ش": "ش",
    "sad": "ص", "saad": "ص", "ص": "ص",
    "ta": "ط", "taa": "ط", "t_": "ط", "ط": "ط",
    "ein": "ع", "ain": "ع", "eyn": "ع", "ayn": "ع", "ع": "ع",
    "f": "ف", "fe": "ف", "feh": "ف", "ف": "ف",
    "q": "ق", "ghaf": "ق", "qaf": "ق", "gh": "ق", "ق": "ق",
    "k": "ک", "kaf": "ک", "ک": "ک", "ك": "ک",
    "g": "گ", "gaf": "گ", "گ": "گ",
    "l": "ل", "lam": "ل", "ل": "ل",
    "m": "م", "mim": "م", "meem": "م", "م": "م",
    "n": "ن", "non": "ن", "nun": "ن", "noon": "ن", "ن": "ن",
    "v": "و", "w": "و", "vav": "و", "waw": "و", "و": "و",
    "h": "ه", "he": "ه", "heh": "ه", "ه": "ه", "هـ": "ه", "ة": "ه",
    "y": "ی", "ye": "ی", "yeh": "ی", "ی": "ی", "ي": "ی",
    "diplomat": "D", "service": "S",
}
IGNORED_YOLO_LABELS = {"plate", "pelak", "license_plate", "car", "iran", "ir", "flag", "blue"}

# کدهای استانی پلاک ایران
IRAN_PROVINCES = {
    "11": "تهران", "22": "تهران", "33": "تهران", "44": "تهران", "55": "تهران",
    "66": "تهران", "77": "تهران", "88": "تهران", "99": "تهران",
    "10": "تهران", "20": "تهران", "30": "تهران", "40": "تهران", "21": "تهران (حومه)",
    "68": "البرز (کرج)", "78": "البرز (کرج)", "38": "البرز",
    "13": "اصفهان", "23": "اصفهان", "43": "اصفهان", "53": "اصفهان", "67": "اصفهان",
    "12": "خراسان رضوی (مشهد)", "32": "خراسان رضوی", "42": "خراسان رضوی", "36": "خراسان رضوی", "74": "خراسان رضوی",
    "15": "آذربایجان شرقی (تبریز)", "25": "آذربایجان شرقی", "35": "آذربایجان شرقی",
    "63": "فارس (شیراز)", "73": "فارس", "83": "فارس", "93": "فارس",
    "14": "خوزستان (اهواز)", "24": "خوزستان", "34": "خوزستان",
    "46": "گیلان (رشت)", "56": "گیلان", "76": "گیلان",
    "62": "مازندران (ساری)", "72": "مازندران", "82": "مازندران", "92": "مازندران",
    "18": "همدان", "28": "همدان",
    "19": "کرمانشاه", "29": "کرمانشاه",
    "17": "آذربایجان غربی (ارومیه)", "27": "آذربایجان غربی", "37": "آذربایجان غربی",
    "45": "کرمان", "65": "کرمان", "75": "کرمان",
    "47": "مرکزی (اراک)", "57": "مرکزی",
    "79": "قزوین", "89": "قزوین",
    "16": "قم",
    "54": "یزد", "64": "یزد",
    "59": "گلستان (گرگان)", "69": "گلستان",
    "87": "زنجان", "97": "زنجان",
    "51": "کردستان (سنندج)", "61": "کردستان",
    "31": "لرستان (خرم‌آباد)", "41": "لرستان",
    "48": "بوشهر", "58": "بوشهر",
    "84": "هرمزگان (بندرعباس)", "94": "هرمزگان",
    "85": "سیستان و بلوچستان", "95": "سیستان و بلوچستان",
    "91": "اردبیل",
    "86": "سمنان", "96": "سمنان",
    "71": "چهارمحال و بختیاری", "81": "چهارمحال و بختیاری",
    "26": "خراسان شمالی (بجنورد)",
    "52": "خراسان جنوبی (بیرجند)",
    "49": "کهگیلویه و بویراحمد",
    "98": "ایلام",
}

IRAN_PLATE_META = {
    "ع": ("عمومی / حمل‌ونقل", "yellow"),
    "ت": ("تاکسی", "yellow"),
    "ک": ("ماشین‌آلات کشاورزی", "yellow"),
    "الف": ("دولتی", "red"),
    "پ": ("نیروی انتظامی", "green"),
    "ث": ("سپاه پاسداران", "green"),
    "ز": ("وزارت دفاع", "blue"),
    "ف": ("ستاد کل نیروهای مسلح", "blue"),
    "ش": ("ارتش", "tan"),
    "ژ": ("جانبازان و معلولین", "white"),
    "D": ("دیپلمات", "blue"),
    "S": ("سرویس سفارت", "blue"),
}

IRAN_LETTERS = [
    "الف", "ب", "پ", "ت", "ث", "ج", "د", "ز", "ژ", "س", "ش",
    "ص", "ط", "ع", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ه", "ی",
]
LETTERS_PATTERN = "|".join(IRAN_LETTERS)
LTR_PLATE_REGEX = re.compile(rf"^(\d{{2}})({LETTERS_PATTERN})(\d{{3}})(\d{{2}})$")

CHAR_MAP = str.maketrans({
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
})


def get_ocr_reader(lang_key: str) -> easyocr.Reader:
    if lang_key not in READERS:
        langs = ["fa"] if lang_key == "fa" else ["en"]
        READERS[lang_key] = easyocr.Reader(langs, gpu=USE_GPU, verbose=False)
    return READERS[lang_key]


def normalize_yolo_char(raw_name: str) -> str | None:
    key = str(raw_name).strip().lower()
    if key in IGNORED_YOLO_LABELS:
        return None
    if key in YOLO_LABEL_MAP:
        return YOLO_LABEL_MAP[key]
    raw_clean = str(raw_name).strip()
    if raw_clean in YOLO_LABEL_MAP:
        return YOLO_LABEL_MAP[raw_clean]
    if len(raw_clean) == 1 and raw_clean.isdigit():
        return raw_clean
    return None


def parse_iranian_sequence(chars: list[dict]) -> dict:
    """تبدیل لیست کاراکترهای مرتب‌شده (چپ به راست) به ساختار کامل پلاک ایرانی"""
    tokens = [c["char"] for c in chars]
    joined = "".join(tokens)

    p1 = letter = p2 = city = ""
    pattern_ok = False

    m = LTR_PLATE_REGEX.match(joined)
    if m:
        p1, letter, p2, city = m.groups()
        pattern_ok = True
    else:
        # تلاش برای تفکیک هوشمند حتی اگر یک کاراکتر جا افتاده باشد
        letter_idx = next((i for i, t in enumerate(tokens) if not t.isdigit()), -1)
        if letter_idx != -1:
            letter = tokens[letter_idx]
            left_digits = "".join(t for t in tokens[:letter_idx] if t.isdigit())
            right_digits = "".join(t for t in tokens[letter_idx + 1:] if t.isdigit())
            p1 = left_digits[-2:] if len(left_digits) >= 2 else left_digits
            if len(right_digits) >= 5:
                p2 = right_digits[:3]
                city = right_digits[3:5]
                pattern_ok = len(p1) == 2 and len(p2) == 3 and len(city) == 2
            else:
                p2 = right_digits[:3]
                city = right_digits[3:]
        else:
            digits = "".join(t for t in tokens if t.isdigit())
            p1 = digits[:2]
            p2 = digits[2:5]
            city = digits[5:7]

    plate_type, theme = IRAN_PLATE_META.get(letter, ("خودروی شخصی" if letter else "نامشخص", "white"))
    province = IRAN_PROVINCES.get(city, "نامشخص") if city else "نامشخص"

    if p1 or letter or p2:
        display = f"{p1} {letter} {p2}".strip()
        if city:
            display = f"{display} | ایران {city}"
    else:
        display = joined

    return {
        "formatted": display,
        "parts": {"p1": p1, "letter": letter, "p2": p2, "city": city},
        "province": province,
        "plate_type": plate_type,
        "theme": theme,
        "pattern_ok": pattern_ok,
    }


def run_iran_yolo_chars(img: np.ndarray) -> dict:
    """تشخیص تک‌تک کاراکترهای پلاک ایرانی با YOLOv8 روی PyTorch"""
    results = IRAN_CHAR_MODEL.predict(
        source=img,
        conf=0.25,
        iou=0.45,
        agnostic_nms=True,
        device=DEVICE,
        verbose=False,
    )
    res = results[0]
    names = res.names
    detected = []

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, clss):
            raw_label = names[int(cls_id)]
            ch = normalize_yolo_char(raw_label)
            if not ch:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            detected.append({
                "char": ch,
                "conf": round(float(conf), 3),
                "xc": (x1 + x2) / 2.0,
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

    # مرتب‌سازی کاراکترها از چپ به راست
    detected.sort(key=lambda d: d["xc"])

    # اگر بیش از ۸ کاراکتر پیدا شد (مثلاً نویز)، ۸ مورد با بیشترین اطمینان نگه داشته و دوباره مرتب می‌شوند
    if len(detected) > 8:
        detected = sorted(detected, key=lambda d: d["conf"], reverse=True)[:8]
        detected.sort(key=lambda d: d["xc"])

    avg_conf = sum(d["conf"] for d in detected) / len(detected) if detected else 0.0
    parsed = parse_iranian_sequence(detected)

    return {
        "status": "ok" if detected else "no_text",
        "engine": "YOLOv8 Character Detector (PyTorch)",
        "text": parsed["formatted"],
        "parts": parsed["parts"],
        "province": parsed["province"],
        "plate_type": parsed["plate_type"],
        "theme": parsed["theme"],
        "pattern_ok": parsed["pattern_ok"],
        "confidence": round(avg_conf, 3),
        "chars": [
            {"char": d["char"], "conf": d["conf"], "box": d["box"]}
            for d in detected
        ],
        "message": (
            "تطابق کامل الگوی پلاک ایرانی"
            if parsed["pattern_ok"]
            else ("کاراکترها استخراج شدند (الگو ناقص است)" if detected else "کاراکتری تشخیص داده نشد")
        ),
    }


def run_easyocr_fallback(img: np.ndarray, region: str) -> dict:
    """موتور PyTorch EasyOCR برای پلاک‌های لاتین (US/EU) یا حالت موقت ایران در نبود فایل YOLO"""
    h, w = img.shape[:2]
    scale = 1.0
    proc = img
    if h < 140:
        scale = 150.0 / float(h)
        proc = cv2.resize(img, (max(1, int(w * scale)), 150), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    is_iran = region == "IRAN"
    reader = get_ocr_reader("fa" if is_iran else "en")

    with torch.no_grad():
        results = reader.readtext(enhanced)

    if not results:
        return {
            "status": "no_text",
            "engine": "PyTorch EasyOCR" + (" (موقت - بدون YOLO)" if is_iran else " (Latin)"),
            "text": "",
            "parts": {},
            "province": "نامشخص",
            "plate_type": "نامشخص",
            "theme": "white" if is_iran else region.lower(),
            "pattern_ok": False,
            "confidence": 0.0,
            "chars": [],
            "message": "متنی روی پلاک خوانده نشد.",
        }

    parsed_boxes = []
    for item in results:
        if len(item) < 3:
            continue
        pts = np.asarray(item[0], dtype=np.float64).reshape(-1, 2) / scale
        txt = str(item[1]).strip()
        conf = float(item[2])
        if not txt:
            continue
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        parsed_boxes.append({
            "char": txt,
            "conf": round(conf, 3),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "h": y2 - y1,
            "x": x1,
        })

    if not parsed_boxes:
        return {
            "status": "no_text", "engine": "PyTorch EasyOCR", "text": "",
            "parts": {}, "province": "نامشخص", "plate_type": "نامشخص",
            "theme": region.lower(), "pattern_ok": False, "confidence": 0.0,
            "chars": [], "message": "متن معتبری یافت نشد.",
        }

    # حذف نوشته‌های خیلی ریز حاشیه (مثل نام ایالت یا I.R.)
    max_h = max(b["h"] for b in parsed_boxes)
    main_boxes = [b for b in parsed_boxes if b["h"] >= max_h * 0.42] or parsed_boxes
    main_boxes.sort(key=lambda b: b["x"])

    avg_conf = sum(b["conf"] for b in main_boxes) / len(main_boxes)
    raw_join = " ".join(b["char"] for b in main_boxes)

    if is_iran:
        clean_fa = re.sub(r"ایران|ايران|I\.?R\.?|IR", "", raw_join.translate(CHAR_MAP), flags=re.IGNORECASE)
        clean_fa = re.sub(r"[^\dآ-ی\s]", "", clean_fa)
        clean_fa = re.sub(r"\s+", " ", clean_fa).strip()
        return {
            "status": "ok" if clean_fa else "no_text",
            "engine": "EasyOCR (موقت — فایل iran_char_yolo.pt در پوشه model نیست)",
            "text": clean_fa,
            "parts": {},
            "province": "نیازمند مدل YOLO کاراکتر",
            "plate_type": "پلاک ایرانی",
            "theme": "white",
            "pattern_ok": False,
            "confidence": round(avg_conf, 3),
            "chars": [{"char": b["char"], "conf": b["conf"], "box": b["box"]} for b in main_boxes],
            "message": "برای دقت بالای ۹۵٪، فایل iran_char_yolo.pt را در پوشه model قرار بده.",
        }

    clean_en = re.sub(r"[^A-Za-z0-9\-\s]", "", raw_join).upper()
    clean_en = re.sub(r"\s+", " ", clean_en).strip()
    return {
        "status": "ok" if clean_en else "no_text",
        "engine": "PyTorch OCR (Latin)",
        "text": clean_en,
        "parts": {"code": clean_en},
        "province": "ایالات متحده" if region == "US" else "اتحادیه اروپا",
        "plate_type": "پلاک بین‌المللی",
        "theme": "us" if region == "US" else "eu",
        "pattern_ok": bool(clean_en),
        "confidence": round(avg_conf, 3),
        "chars": [{"char": b["char"], "conf": b["conf"], "box": b["box"]} for b in main_boxes],
        "message": "استخراج کاراکترهای لاتین انجام شد.",
    }


@app.get("/")
def serve_home():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="فایل static/index.html پیدا نشد.")
    return FileResponse(index_path)


@app.get("/api/status")
def system_status():
    tm_ready = all(
        (MODEL_DIR / fname).exists() and (MODEL_DIR / fname).stat().st_size > 0
        for fname in ("model.json", "weights.bin", "metadata.json")
    )
    gpu_name = torch.cuda.get_device_name(0) if USE_GPU else "CPU"
    return {
        "torch_version": torch.__version__,
        "cuda_available": USE_GPU,
        "device_name": gpu_name,
        "ultralytics_installed": ULTRALYTICS_OK,
        "tm_ready": tm_ready,
        "plate_detector_ready": PLATE_DETECTOR is not None,
        "iran_char_yolo_ready": IRAN_CHAR_MODEL is not None,
    }


@app.get("/api/tm-bundle")
def get_tm_bundle():
    model_json_path = MODEL_DIR / "model.json"
    metadata_path = MODEL_DIR / "metadata.json"
    if not model_json_path.exists() or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="فایل‌های مدل در پوشه model پیدا نشدند.")

    model_json = json.loads(model_json_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    weight_chunks = bytearray()
    expected_bytes = 0
    for group in model_json.get("weightsManifest", []):
        for rel_path in group.get("paths", []):
            w_path = MODEL_DIR / Path(rel_path).name
            if not w_path.exists():
                raise HTTPException(status_code=404, detail=f"فایل '{w_path.name}' پیدا نشد.")
            data = w_path.read_bytes()
            if len(data) == 0:
                raise HTTPException(status_code=500, detail=f"فایل '{w_path.name}' خالی است.")
            weight_chunks.extend(data)
        for w_spec in group.get("weights", []):
            quant = w_spec.get("quantization")
            dtype = quant.get("dtype") if isinstance(quant, dict) else w_spec.get("dtype", "float32")
            expected_bytes += math.prod(w_spec.get("shape", [])) * DTYPE_BYTES.get(dtype, 4)

    if len(weight_chunks) < expected_bytes:
        raise HTTPException(status_code=500, detail="فایل weights.bin ناقص است.")

    return JSONResponse(
        content={
            "modelJSON": model_json,
            "metadata": metadata,
            "weightDataBase64": base64.b64encode(bytes(weight_chunks)).decode("ascii"),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/detect-plate")
async def detect_plate_endpoint(file: UploadFile = File(...)):
    """یافتن و برش خودکار پلاک از تصویر کامل ماشین (در صورت وجود plate_detector.pt)"""
    t0 = time.perf_counter()
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="تصویر معتبر نیست.")

    h, w = img.shape[:2]
    detected = False
    bbox = None
    plate_conf = 0.0
    crop = img

    if PLATE_DETECTOR is not None:
        try:
            res = PLATE_DETECTOR.predict(source=img, conf=0.35, device=DEVICE, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                best_idx = int(res.boxes.conf.argmax().item())
                x1, y1, x2, y2 = [float(v) for v in res.boxes.xyxy[best_idx].cpu().numpy()]
                plate_conf = round(float(res.boxes.conf[best_idx].item()), 3)
                # کمی حاشیه اضافه برای حفظ لبه‌ها
                pad_x = (x2 - x1) * 0.04
                pad_y = (y2 - y1) * 0.06
                lx = max(0, int(x1 - pad_x))
                ty = max(0, int(y1 - pad_y))
                rx = min(w, int(x2 + pad_x))
                by = min(h, int(y2 + pad_y))
                if (rx - lx) >= 30 and (by - ty) >= 12:
                    crop = img[ty:by, lx:rx]
                    bbox = [lx, ty, rx, by]
                    detected = True
        except Exception:
            traceback.print_exc()

    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise HTTPException(status_code=500, detail="خطا در پردازش تصویر برش‌خورده.")

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "detected": detected,
        "bbox": bbox,
        "plate_conf": plate_conf,
        "crop_data_url": f"data:image/jpeg;base64,{b64}",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/api/analyze")
async def analyze_plate(
    file: UploadFile = File(...),
    region: str = Form("UNKNOWN"),
):
    t0 = time.perf_counter()
    region_upper = region.strip().upper()
    if region_upper == "UNKNOWN":
        return {
            "status": "unknown",
            "engine": "Blocked",
            "text": "",
            "parts": {},
            "province": "—",
            "plate_type": "ناشناخته",
            "theme": "unknown",
            "pattern_ok": False,
            "confidence": 0.0,
            "chars": [],
            "elapsed_ms": 0.0,
            "message": "پلاک ناشناخته تشخیص داده شد؛ پردازش متوقف شد.",
        }

    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="تصویر معتبر نیست.")

    try:
        is_iran = "IRAN" in region_upper
        if is_iran and IRAN_CHAR_MODEL is not None:
            result = run_iran_yolo_chars(img)
        else:
            region_key = "IRAN" if is_iran else ("US" if "US" in region_upper else "EU")
            result = run_easyocr_fallback(img, region_key)
    except Exception as exc:
        traceback.print_exc()
        result = {
            "status": "error",
            "engine": "Error",
            "text": "",
            "parts": {},
            "province": "—",
            "plate_type": "—",
            "theme": "unknown",
            "pattern_ok": False,
            "confidence": 0.0,
            "chars": [],
            "message": f"خطا در پردازش: {exc}",
        }

    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result

@app.get("/api/samples")
def list_sample_plates():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    items = []
    if SAMPLES_DIR.exists():
        for path in sorted(SAMPLES_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                rel = path.relative_to(SAMPLES_DIR).as_posix()
                parent = path.parent.name.lower()
                if parent in ("iran", "us", "eu"):
                    category = parent.upper()
                else:
                    fname = path.name.lower()
                    if fname.startswith(("ir", "iran")):
                        category = "IRAN"
                    elif fname.startswith("us"):
                        category = "US"
                    elif fname.startswith("eu"):
                        category = "EU"
                    else:
                        category = "OTHER"
                items.append({
                    "name": path.stem,
                    "url": f"/samples/{rel}",
                    "category": category,
                })
    return {"samples": items}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)