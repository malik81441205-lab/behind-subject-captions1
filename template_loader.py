import copy, glob, json, os

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def list_templates():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TEMPLATE_DIR, "*.json")))


def deep_merge(a, b):
    """b over a, recursively (dicts merge, everything else is replaced)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(a[k], v) if k in a else copy.deepcopy(v)
        return out
    return copy.deepcopy(b)


def normalize(t):
    """Fill every text style (roles) from the base style so the renderer never has to guess."""
    t = copy.deepcopy(t)
    styles = t.setdefault("styles", {})
    base = deep_merge({"font": t["font"], "scale": 1.0, "color": t["colors"]["text"], "gradient": None,
                       "pill": None, "own_line": False, "uppercase": t.get("uppercase", False),
                       "tracking": 0.0, "effects": {}}, styles.get("base", {}))
    styles["base"] = base
    kw = deep_merge(base, {"color": t["colors"].get("keyword", base["color"])})
    kw = deep_merge(kw, t.get("keyword", {}))
    styles["keyword"] = deep_merge(kw, styles.get("keyword", {}))
    for k, v in list(styles.items()):
        if k not in ("base", "keyword"):
            styles[k] = deep_merge(base, v)
    return t


def load_template(name, override=None):
    path = name if name.endswith(".json") else os.path.join(TEMPLATE_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"Template '{name}' not found. Available: {', '.join(list_templates())}")
    raw = json.load(open(path))
    if override:  # user's custom settings win over the template defaults
        raw = deep_merge(raw, override)
    for key in ("font", "colors", "size", "layout"):
        if key not in raw:
            raise SystemExit(f"Template '{name}' is missing required key '{key}'")
    return normalize(raw)
