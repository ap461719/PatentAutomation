"""
End-to-end pipeline:
A) GPT (text-only) → extract candidate vocabulary from claims
1) OCR numerals + bounding boxes (Tesseract)
2) Make tight + context crops per numeral
3) VLM (GPT-4o) per numeral → choose best name from candidate vocab (or 'unknown')
4) Save structured JSON with id, name, confidence, evidence, bbox

"""

import os, io, re, cv2, json, base64, argparse
import numpy as np
from PIL import Image
import pytesseract
from dataclasses import dataclass
from typing import List, Dict, Optional

from utils import client



NUMERAL_RE = re.compile(r"^\d+[a-z]?$", re.IGNORECASE)
CONFUSIONS = {"S":"5","s":"5","O":"0","o":"0","I":"1","l":"1","L":"1"}

def correct_confusion(tok: str) -> str:
    if tok in CONFUSIONS:
        return CONFUSIONS[tok]
    if tok == "36":  # frequent in suffix-heavy patents (3c)
        return "3c"
    return tok

def clamp(x1,y1,x2,y2,W,H):
    return [max(0,x1),max(0,y1),min(W,x2),min(H,y2)]

def pad_xyxy(box, pad, W, H):
    x1,y1,x2,y2 = box
    return clamp(x1-pad, y1-pad, x2+pad, y2+pad, W, H)

def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def read_text(path: Optional[str]) -> str:
    if not path: return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------
# A) GPT pre-pass: claims → candidate vocab
# ---------------------------

VOCAB_SYSTEM = (
    "You extract short, canonical, de-duplicated technical noun phrases from patent claims. "
    "Return strict JSON with a single key 'vocabulary' listing 5-40 phrases. "
    "Phrases should be 1-4 words (e.g., 'signal terminal', 'die pad', 'wiring pattern'). "
    "Avoid verbs, whole sentences, and pronouns. Use lower case."
)

def extract_vocab_with_gpt(claims_text: str) -> List[str]:
    if not claims_text.strip():
        # Fallback if no claims provided
        return [
            "semiconductor component","main body","terminal","signal terminal","ground terminal",
            "unused terminal","circuit board","wiring pattern","first trace","second trace",
            "third land","first land","second land","connecting trace","connecting member",
            "insulating film","resist film","silk film","metallic body","thermal via",
            "heat-radiating member"
        ]

    user = [
        {"type":"text","text": "Claims:\n" + claims_text},
        {"type":"text","text": (
            "Task: Extract candidate component names as short noun phrases. "
            "Return JSON only like: {\"vocabulary\": [\"signal terminal\", \"die pad\", ...]}"
        )}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # text-only is fine; choose a cheap reasoning model
        messages=[{"role":"system","content": VOCAB_SYSTEM},
                  {"role":"user","content": user}],
        temperature=0.0
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        vocab = [s.strip().lower() for s in data.get("vocabulary", []) if 1 <= len(s.strip()) <= 64]
        vocab = sorted(list(dict.fromkeys(vocab)))  # dedupe, stable
        # tiny safety: keep reasonable size
        return vocab[:60]
    except Exception:
        # Fallback minimal vocab if parsing fails
        return ["terminal","signal terminal","ground terminal","unused terminal",
                "wiring pattern","first land","second land","third land","trace","die pad",
                "insulating film","thermal via","heat-radiating member","main body","circuit board"]

# ---------------------------
# 1) OCR numerals + boxes
# ---------------------------

@dataclass
class Hit:
    id: str
    box: List[int]  # [x1,y1,x2,y2]

def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((1,1),np.uint8))
    return clean

def ocr_numerals(image: Image.Image) -> List[Hit]:
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    prep = preprocess_for_ocr(bgr)
    H, W = prep.shape[:2]

    cfg = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(prep, lang="eng", config=cfg,
                                     output_type=pytesseract.Output.DICT)
    hits: List[Hit] = []
    for text, x, y, w, h in zip(
        data["text"], data["left"], data["top"], data["width"], data["height"]
    ):
        tok = (text or "").strip()
        if not tok:
            continue
        if len(tok) <= 2:
            tok = correct_confusion(tok)
        if not NUMERAL_RE.match(tok):
            continue
        x1,y1,x2,y2 = clamp(x, y, x+w, y+h, W, H)
        hits.append(Hit(id=tok.lower(), box=[x1,y1,x2,y2]))

    # keep largest box per id (tesseract may duplicate)
    best: Dict[str, Hit] = {}
    for h in hits:
        if h.id not in best:
            best[h.id] = h
        else:
            def area(b): return max(0,b[2]-b[0]) * max(0,b[3]-b[1])
            if area(h.box) > area(best[h.id].box):
                best[h.id] = h
    return list(best.values())

# ---------------------------
# 2) Crops
# ---------------------------

def make_crops(pil: Image.Image, box_xyxy: List[int], tight_pad=28, wide_pad=160):
    W,H = pil.size
    tight = pad_xyxy(box_xyxy, tight_pad, W, H)
    wide  = pad_xyxy(box_xyxy, wide_pad,  W, H)
    return pil.crop(tuple(tight)), pil.crop(tuple(wide))

# ---------------------------
# 3) VLM per numeral
# ---------------------------

VLM_SYSTEM = "You are a precise assistant for patent figures. Return JSON only."

def ask_vlm_for_name(num_id: str,
                     tight_url: str,
                     wide_url: str,
                     vocab: List[str],
                     claims_text: str) -> Dict:
    instr = (
        "Select the best matching component name for the label shown in 'id' from the given vocabulary. "
        "Prefer exact, short technical noun phrases. If none fits, return 'unknown'. "
        f"Return JSON: {{\"id\":\"{num_id}\",\"name\":\"<candidate or unknown>\",\"confidence\":0..1,\"evidence\":\"<short reason>\"}}"
    )
    blocks = []
    if claims_text.strip():
        blocks.append({"type":"text","text": "Claims context:\n" + claims_text[:6000]})
    blocks.extend([
        {"type":"text","text": f"id: {num_id}"},
        {"type":"text","text": "Vocabulary: " + json.dumps(vocab)},
        {"type":"image_url","image_url":{"url": tight_url, "detail":"high"}},
        {"type":"image_url","image_url":{"url": wide_url,  "detail":"high"}},
        {"type":"text","text": instr},
    ])

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content": VLM_SYSTEM},
                  {"role":"user","content": blocks}],
        temperature=0.0
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"id": num_id, "name": "unknown", "confidence": 0.0, "evidence": "parse_error"}

# ---------------------------
# 4) Main
# ---------------------------

def run(image_path: str, claims_path: Optional[str], out_path: str, conf_thresh: float):
    claims = read_text(claims_path)
    # A) get vocabulary once (text-only GPT)
    vocab = extract_vocab_with_gpt(claims)

    # Load image
    pil = Image.open(image_path).convert("RGB")
    W,H = pil.size

    # 1) OCR numerals
    hits = ocr_numerals(pil)
    if not hits:
        result = {"figure_id": os.path.basename(image_path), "components": [], "note": "no_numerals_found"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # 2-3) per-numeral crops + VLM
    components = []
    for h in sorted(hits, key=lambda z: z.id):
        tight, wide = make_crops(pil, h.box)
        tight_url = pil_to_data_url(tight)
        wide_url  = pil_to_data_url(wide)

        ans = ask_vlm_for_name(h.id, tight_url, wide_url, vocab, claims)
        # attach bbox; threshold low confidence to 'unknown'
        ans["bbox_xyxy"] = h.box
        if float(ans.get("confidence", 0.0)) < conf_thresh:
            ans["name"] = "unknown"
        components.append(ans)

    result = {"figure_id": os.path.basename(image_path), "components": components,
              "vocab_used": vocab}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to patent figure (PNG/JPG)")
    ap.add_argument("--claims", default="", help="Path to claims .txt (optional but recommended)")
    ap.add_argument("--out", default="components.json", help="Output JSON path")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for accepting names")
    args = ap.parse_args()
    run(args.image, args.claims, args.out, args.conf)




"""When we started, your code was giving the entire figure and claims to GPT-4o and asking it to list all components. 
Accuracy was low because the model had to do too many hard tasks at once: read tiny numerals, find arrows, link them to shapes,
 and invent names. From the PatFig paper, we learned that large vision-language models do not directly output clean component lists 
   they still rely on OCR and text alignment to work well. The DesignCLIP and classification paper showed us another important point: 
   these models perform better when you recast the problem into smaller classification or VQA-style questions instead of open-ended captioning. 
   That insight shifted our pipeline design. Instead of “figure → list,” we now do OCR first, which reliably extracts the numeral labels and their bounding boxes. 
   But OCR only finds the text itself, not the component, so we pad and crop around each numeral to capture the arrow and local geometry. 
   Each crop, plus the numeral ID, is then passed to a VLM (like GPT-4o Vision). To keep naming consistent and reduce hallucination, 
   we do a quick text-only GPT pre-step on the claims to extract a clean candidate vocabulary (short noun phrases like “signal terminal,” 
   “ground terminal,” etc.). The VLM’s job becomes much easier: given a crop + numeral + vocabulary, simply pick the best matching term (or say “unknown”). 
   This combination of OCR anchoring, per-numeral crops, claims-derived vocabulary, and structured JSON outputs is what led us to the final code snippet 
 a far more accurate and auditable way to generate component lists than the original whole-image prompting"""