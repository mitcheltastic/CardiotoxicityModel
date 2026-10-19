"""Cardivex full-pipeline simulation, isolated from the RF prediction code.

Design rule: nothing in this module may raise into app.py. Every public
function returns a dict with "ok": True/False. If R is missing, the C model
fails to compile, or a simulation crashes, only /simulate reports an error;
/predict and /explain are never affected.

Single-cell (fixed-parameter) CiPA AP simulation with the FDA IKr-dynamic ORd
model, ported from FDA/CiPA Lab_Specific_Validation_Calibration_2020
(chantest_AP_simulation). Each job simulates 1, 2, 3 and 4x Cmax (same as the
training data) and returns the mean of the 11 biomarkers as ratios to the
drug-free control, in the RF bundle's feature order.

Input modes (supervisor specification):
  1. {"drug": "vandetanib"}                         everything from the database
  2. {"cmax_nM": .., "herg": {..}, "ic50": {..}}     everything from the user
  3. {"drug": "..", "herg": {..}, "ic50": {..}}      Cmax from database, rest from user
"""
import hmac
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(tempfile.gettempdir(), "cardivex_sim")  # writable copy, compiled at start-up
FEATURES = ["qNet", "dvdtmax", "vmax", "vrest", "APD50", "APD90",
            "max_dv", "camax", "carest", "CaTD50", "CaTD90"]
HERG_KEYS = ["Kmax", "Ku", "n", "halfmax", "Vhalf"]
IC50_KEYS = ["INa_IC50", "INa_h", "INaL_IC50", "INaL_h", "ICaL_IC50", "ICaL_h"]
# Training data: biomarkers averaged over simulations at 1, 2, 3 and 4x Cmax
# (confirmed by the supervisor, 19 Oct 2026). The model input must match.
DOSES = [float(d) for d in os.environ.get("SIM_DOSES", "1,2,3,4").split(",")]
PARALLEL = int(os.environ.get("SIM_PARALLEL", "2"))  # 2 vCPU on HF cpu-basic
ENGINE_VERSION = "cipa-ordherg-fixed-1.1-mean1to4"

_state = {"ready": False, "error": "initialising"}
_init_lock = threading.Lock()
_sim_lock = threading.Lock()
_drug_db = None
_control = None


def _load_json(name):
    with open(os.path.join(SRC_DIR, name)) as f:
        return json.load(f)


def _init():
    """Copy the engine to a writable dir and compile the C model. Runs once, in the background."""
    global _drug_db, _control
    with _init_lock:
        if _state["ready"]:
            return
        try:
            _drug_db = _load_json("drug_db.json")
            _control = _load_json("control.json")
            if shutil.which("Rscript") is None:
                raise RuntimeError("Rscript not found (packages.txt not installed?)")
            if os.path.exists(WORK_DIR):
                shutil.rmtree(WORK_DIR)
            shutil.copytree(SRC_DIR, WORK_DIR, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
            proc = subprocess.run(["R", "CMD", "SHLIB", "newordherg_qNet.c"],
                                  cwd=os.path.join(WORK_DIR, "models"),
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(f"model compile failed: {proc.stderr[-500:]}")
            _state.update(ready=True, error=None)
            print("[sim] simulation engine ready", flush=True)
        except Exception as e:  # never propagate
            _state.update(ready=False, error=str(e)[:500])
            print(f"[sim] simulation engine unavailable: {_state['error']}", flush=True)


def start_background_init():
    try:
        threading.Thread(target=_init, daemon=True).start()
    except Exception as e:
        _state.update(ready=False, error=str(e)[:500])


def _check_key(key):
    expected = os.environ.get("SIM_API_KEY", "")
    return bool(expected) and bool(key) and hmac.compare_digest(str(key), expected)


def _run_engine(payload):
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
        with open(inp, "w") as f:
            json.dump(payload, f)
        cmd = ["Rscript", os.path.join(WORK_DIR, "sim_engine.R"), inp, out]
        if shutil.which("nice"):
            cmd = ["nice", "-n", "10"] + cmd  # lower priority so /predict keeps the CPU
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"engine failed: {proc.stderr[-500:]}")
        with open(out) as f:
            return json.load(f)


def _resolve(req):
    drug = (req.get("drug") or "").strip().lower()
    base = _drug_db.get(drug) if drug else None
    if drug and base is None:
        raise ValueError(f"unknown drug '{drug}'")
    cmax = req.get("cmax_nM", base["cmax_nM"] if base else None)
    herg = {**(base["herg"] if base else {}), **(req.get("herg") or {})}
    ic50 = {**(base["ic50"] if base else {}), **(req.get("ic50") or {})}
    doses = req.get("doses") or DOSES
    doses = [float(d) for d in doses]
    missing = [k for k in HERG_KEYS if herg.get(k) is None]
    if cmax is None:
        missing.append("cmax_nM")
    if missing:
        raise ValueError(f"missing parameters: {', '.join(missing)}")
    cmax = float(cmax)
    if not (cmax > 0) or not doses or len(doses) > 8 or not all(0 < d <= 100 for d in doses):
        raise ValueError("cmax_nM must be > 0 and doses must be 1 to 8 values in (0, 100]")
    for k in IC50_KEYS:
        v = ic50.get(k)
        if v is not None and float(v) <= 0:
            raise ValueError(f"{k} must be > 0 (leave it empty if the drug does not block this channel)")
    return {"drug": drug or None, "cmax_nM": cmax, "doses": doses,
            "herg": {k: float(herg[k]) for k in HERG_KEYS},
            "ic50": {k: float(ic50[k]) for k in IC50_KEYS if ic50.get(k) is not None}}


def simulate(request_json, api_key):
    """Gradio handler. Always returns a dict, never raises."""
    try:
        if not _check_key(api_key):
            return {"ok": False, "error": "unauthorized"}
        if not _state["ready"]:
            return {"ok": False, "error": f"simulation engine not ready: {_state['error']}"}
        try:
            req = json.loads(request_json) if request_json else {}
            params = _resolve(req)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return {"ok": False, "error": str(e)}
        payloads = [{"cmax_nM": params["cmax_nM"], "dose": d,
                     "herg": params["herg"], "ic50": params["ic50"]} for d in params["doses"]]
        with _sim_lock:
            with ThreadPoolExecutor(max_workers=max(1, PARALLEL)) as pool:
                results = list(pool.map(_run_engine, payloads))
        ref = _control["metrics"]
        per_dose, raw_mean, excluded = [], {}, {}
        for d, res in zip(params["doses"], results):
            r = {k: (res["metrics"].get(k) / abs(ref[k])
                     if res["metrics"].get(k) is not None and ref.get(k) else None) for k in FEATURES}
            per_dose.append({"dose": d, "conc_nM": res["conc_nM"], "kept_beat": res["beat"],
                             "ratios": [r[k] for k in FEATURES], "raw_metrics": res["metrics"],
                             "flags": {"EADs": res["EADs"], "noDepols": res["noDepols"],
                                       "noRepols": res["noRepols"]},
                             "elapsed_s": res["elapsed_s"]})
        for k in FEATURES:
            vals = [res["metrics"].get(k) for res in results]
            ok_vals = [v for v in vals if v is not None]
            if not ok_vals or not ref.get(k):
                return {"ok": False, "params": params, "per_dose": per_dose,
                        "error": f"simulation produced no value for {k} at any dose "
                                 "(e.g. repolarization failure); cannot build model input"}
            if len(ok_vals) < len(vals):
                excluded[k] = [d for d, v in zip(params["doses"], vals) if v is None]
            raw_mean[k] = sum(ok_vals) / len(ok_vals)
        ratios = [raw_mean[k] / abs(ref[k]) for k in FEATURES]
        top = results[-1]
        return {
            "ok": True,
            "features": FEATURES,
            "ratios": ratios,                      # model input: mean over doses
            "dose_mode": "mean_of_doses",
            "doses": params["doses"],
            "raw_metrics": raw_mean,
            "control_metrics": ref,
            "params": params,
            "conc_nM": [p["conc_nM"] for p in per_dose],
            "per_dose": per_dose,
            "excluded_doses": excluded,            # metric -> doses where it was undefined
            "flags": {k: sum(p["flags"][k] for p in per_dose) for k in ("EADs", "noDepols", "noRepols")},
            "traces_by_dose": [{"dose": d, **res["trace"]} for d, res in zip(params["doses"], results)],
            "trace": top["trace"],                 # highest dose, kept for compatibility
            "control_trace": _control["trace"],
            "elapsed_s": round(sum(p["elapsed_s"] for p in per_dose), 1),
            "engine_version": ENGINE_VERSION,
        }
    except Exception as e:
        return {"ok": False, "error": f"simulation failed: {str(e)[:500]}"}


def list_drugs(api_key):
    try:
        if not _check_key(api_key):
            return {"ok": False, "error": "unauthorized"}
        db = _drug_db if _drug_db is not None else _load_json("drug_db.json")
        return {"ok": True, "doses": DOSES, "dose_mode": "mean_of_doses", "drugs": db}
    except Exception as e:
        return {"ok": False, "error": str(e)[:500]}


def status():
    return {"ready": _state["ready"], "error": _state["error"], "engine_version": ENGINE_VERSION}
