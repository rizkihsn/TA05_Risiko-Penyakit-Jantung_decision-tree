"""CardioTree: aplikasi Flask untuk klasifikasi risiko penyakit jantung dengan Decision Tree
(Model B, pruned max_depth=3). Selain prediksi, aplikasi menampilkan alur keputusan pohon."""
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

def load_model(path):
    m = joblib.load(path)
    # Patch kompatibilitas antar-versi scikit-learn (misal _fit_dtype vs _fill_dtype pada SimpleImputer)
    try:
        prep = m.named_steps.get("prep")
        if prep and hasattr(prep, "transformers_"):
            for _, transformer, _ in prep.transformers_:
                if hasattr(transformer, "_fit_dtype") and not hasattr(transformer, "_fill_dtype"):
                    transformer._fill_dtype = transformer._fit_dtype
                elif hasattr(transformer, "_fill_dtype") and not hasattr(transformer, "_fit_dtype"):
                    transformer._fit_dtype = transformer._fill_dtype
    except Exception:
        pass
    return m


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model_decision_tree.joblib")
model = load_model(MODEL_PATH)
PREP = model.named_steps["prep"]
TREE = model.named_steps["clf"]
FEATURE_NAMES = [n.split("__", 1)[1] for n in PREP.get_feature_names_out()]

# Angka hasil evaluasi pada data uji (lihat laporan/notebook)
MODEL_INFO = {"akurasi": "85,5%", "auc": "0,915", "pasien": 918, "daun": 8, "kedalaman": 3}

# Definisi kolom. Nama kolom harus sama persis dengan kolom saat pelatihan.
FIELDS = {
    "Age": {"kind": "number", "label": "Usia", "short": "Usia", "unit": "tahun", "min": 1, "max": 120,
            "step": 1, "placeholder": "54", "hint": "1 sampai 120 tahun"},
    "Sex": {"kind": "choice", "label": "Jenis kelamin", "short": "Jenis kelamin",
            "options": [("M", "Laki-laki"), ("F", "Perempuan")]},
    "ChestPainType": {"kind": "choice", "label": "Tipe nyeri dada", "short": "Tipe nyeri dada",
                      "options": [("ATA", "Angina atipikal (ATA)"), ("NAP", "Nyeri non-angina (NAP)"),
                                  ("ASY", "Tanpa gejala (ASY)"), ("TA", "Angina tipikal (TA)")]},
    "RestingBP": {"kind": "number", "label": "Tekanan darah saat istirahat", "short": "Tekanan darah",
                  "unit": "mm Hg", "min": 0, "max": 250, "step": 1, "placeholder": "130",
                  "hint": "Isi 0 jika tidak diketahui"},
    "Cholesterol": {"kind": "number", "label": "Kolesterol serum", "short": "Kolesterol", "unit": "mg/dl",
                    "min": 0, "max": 700, "step": 1, "placeholder": "220", "hint": "Isi 0 jika tidak diketahui",
                    "unknown_button": True},
    "FastingBS": {"kind": "choice", "label": "Gula darah puasa di atas 120 mg/dl", "short": "Gula darah puasa",
                  "options": [("0", "Tidak"), ("1", "Ya")]},
    "MaxHR": {"kind": "number", "label": "Detak jantung maksimum", "short": "Detak jantung maksimum",
              "unit": "bpm", "min": 40, "max": 230, "step": 1, "placeholder": "138",
              "hint": "Yang dicapai saat uji olahraga"},
    "RestingECG": {"kind": "choice", "wide": True, "label": "EKG saat istirahat", "short": "EKG istirahat",
                   "options": [("Normal", "Normal"), ("ST", "Kelainan ST-T"),
                               ("LVH", "Hipertrofi ventrikel kiri (LVH)")]},
    "ExerciseAngina": {"kind": "choice", "label": "Nyeri dada saat olahraga", "short": "Nyeri dada saat olahraga",
                       "options": [("N", "Tidak"), ("Y", "Ya")]},
    "Oldpeak": {"kind": "number", "label": "Depresi segmen ST (Oldpeak)", "short": "Oldpeak", "unit": "",
                "min": -3, "max": 8, "step": 0.1, "placeholder": "0,6",
                "hint": "Akibat olahraga. 0 berarti tidak ada depresi"},
    "ST_Slope": {"kind": "choice", "wide": True, "label": "Kemiringan segmen ST", "short": "Kemiringan segmen ST",
                 "options": [("Up", "Naik (Up)"), ("Flat", "Datar (Flat)"), ("Down", "Turun (Down)")]},
}
SECTIONS = [
    {"title": "Data pasien", "desc": "Identitas dan keluhan utama.",
     "fields": ["Age", "Sex", "ChestPainType"]},
    {"title": "Tanda vital dan hasil darah", "desc": "Diukur saat pasien beristirahat, kecuali detak jantung maksimum.",
     "fields": ["RestingBP", "Cholesterol", "MaxHR", "FastingBS"]},
    {"title": "EKG dan uji olahraga", "desc": "Hasil rekam jantung dan respons terhadap olahraga.",
     "fields": ["RestingECG", "ExerciseAngina", "Oldpeak", "ST_Slope"]},
]
FEATURE_ORDER = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
                 "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
CHOICES = {k: v for k, v in FIELDS.items() if v["kind"] == "choice"}


def num_str(x, digits=1):
    """Format angka gaya Indonesia (koma desimal), tanpa nol di belakang."""
    s = f"{float(x):.{digits}f}".rstrip("0").rstrip(".")
    return (s or "0").replace(".", ",")


def parse_form(form):
    """Validasi input. Kembalikan (DataFrame satu baris, daftar error [{field, msg}])."""
    errors, row = [], {}
    for name, cfg in FIELDS.items():
        if cfg["kind"] == "number":
            raw = (form.get(name) or "").strip().replace(",", ".")
            try:
                val = float(raw)
            except ValueError:
                errors.append({"field": name, "msg": f"{cfg['label']}: isi dengan angka."})
                continue
            if not (cfg["min"] <= val <= cfg["max"]):
                errors.append({"field": name,
                               "msg": f"{cfg['label']}: isi angka antara {cfg['min']} dan {cfg['max']}."})
                continue
            row[name] = val
        else:
            val = form.get(name)
            if val not in [o[0] for o in cfg["options"]]:
                errors.append({"field": name, "msg": f"{cfg['label']}: pilih salah satu."})
                continue
            row[name] = int(val) if name == "FastingBS" else val
    if errors:
        return None, errors
    # Nilai 0 pada Cholesterol/RestingBP dianggap hilang (sama seperti saat pelatihan)
    for c in ("Cholesterol", "RestingBP"):
        if row[c] == 0:
            row[c] = np.nan
    return pd.DataFrame([row])[FEATURE_ORDER], []


def describe_step(feat, thr, val, goes_left):
    """Ubah satu simpul keputusan menjadi kalimat yang mudah dibaca."""
    for col, cfg in CHOICES.items():
        prefix = col + "_"
        if feat.startswith(prefix):
            code = feat[len(prefix):]
            texts = dict(cfg["options"])
            if len(cfg["options"]) == 2:  # biner: sebut pilihan yang berlaku
                chosen = code if not goes_left else next(c for c, _ in cfg["options"] if c != code)
                return {"text": f"{cfg['short']}: {texts[chosen]}", "note": ""}
            if goes_left:
                return {"text": f"{cfg['short']} bukan {texts[code]}", "note": ""}
            return {"text": f"{cfg['short']}: {texts[code]}", "note": ""}
    cfg = FIELDS[feat]
    op = "≤" if goes_left else ">"
    unit = f" {cfg['unit']}" if cfg.get("unit") else ""
    if feat == "FastingBS":
        text = "Gula darah puasa ≤ 120 mg/dl" if goes_left else "Gula darah puasa > 120 mg/dl"
        return {"text": text, "note": ""}
    return {"text": f"{cfg['short']} {op} {num_str(thr, 2)}{unit}", "note": f"nilai Anda {num_str(val, 1)}{unit}"}


def explain(X):
    """Telusuri jalur akar -> daun untuk satu pasien. Kembalikan (langkah, info daun)."""
    Xt = PREP.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)
    leaf = int(TREE.apply(Xt)[0])
    steps = []
    for node in TREE.decision_path(Xt).indices:
        if int(node) == leaf:
            continue
        f = int(TREE.tree_.feature[node])
        thr = float(TREE.tree_.threshold[node])
        val = float(Xt[0, f])
        steps.append(describe_step(FEATURE_NAMES[f], thr, val, val <= thr))
    v = TREE.tree_.value[leaf][0]
    frac = v / v.sum()
    return steps, {"n": int(TREE.tree_.n_node_samples[leaf]), "frac_disease": float(frac[1])}


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, values = None, [], {}
    if request.method == "POST":
        values = request.form.to_dict()
        X, errors = parse_form(request.form)
        if X is not None:
            try:
                proba = model.predict_proba(X)[0]
                pred = int(model.predict(X)[0])
                steps, leaf = explain(X)
                p1 = float(proba[1]) * 100
                share = leaf["frac_disease"] if pred == 1 else 1 - leaf["frac_disease"]
                result = {
                    "pred": pred,
                    "label": "Berisiko penyakit jantung" if pred == 1 else "Tidak terindikasi penyakit jantung",
                    "p_disease": round(p1, 1), "p_disease_str": num_str(p1, 1),
                    "p_normal_str": num_str(100 - p1, 1),
                    "steps": steps, "leaf_n": leaf["n"], "leaf_share_str": num_str(share * 100, 0),
                    "leaf_class": "berpenyakit jantung" if pred == 1 else "normal",
                    "low_support": leaf["n"] < 20,
                    "imputed": [FIELDS[c]["short"].lower() for c in ("RestingBP", "Cholesterol") if pd.isna(X.loc[0, c])],
                }
            except Exception as e:
                app.logger.exception("Kesalahan saat kalkulasi prediksi:")
                errors.append({"field": "", "msg": f"Terjadi kesalahan saat memproses prediksi: {e}"})
    return render_template("index.html", fields=FIELDS, sections=SECTIONS, info=MODEL_INFO,
                           result=result, errors=errors, errfields=[e["field"] for e in errors], values=values)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
