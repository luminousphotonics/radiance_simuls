from flask import Flask, request, render_template_string, send_from_directory, url_for, Response
from math import sqrt, ceil, floor
import sys, smtplib, socket
import json
import stripe
from matplotlib import pyplot as plt
import numpy as np
import base64
from io import BytesIO
from collections import Counter
import os
from dotenv import load_dotenv
from flask import send_from_directory, redirect
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from pathlib import Path
from flask import abort as _abort, jsonify, Response, abort, request, url_for
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from radsim.env import PROJECT_ROOT
from extensions import db, migrate, login_manager
from datetime import datetime

load_dotenv()  # this will load .env into os.environ

# Main Flask app for the site.
# static_folder=None because you already serve static assets explicitly.
app = Flask(__name__, static_folder=None)

# --- Core config for database + auth ---------------------------------

# Use DATABASE_URL if present (Render / prod), else a local SQLite file.
app.config.setdefault(
    "SQLALCHEMY_DATABASE_URI",
    os.environ.get("DATABASE_URL")
    or f"sqlite:///{os.path.join(PROJECT_ROOT, 'forum.db')}",
)

app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)

# Needed for sessions, Flask-Login, and Flask-WTF.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.secret_key = app.config["SECRET_KEY"]  # optional, but explicit

# Avatar uploads
app.config["AVATAR_UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "uploads", "avatars")
os.makedirs(app.config["AVATAR_UPLOAD_FOLDER"], exist_ok=True)
app.config["SIGNATURE_UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "uploads", "signatures")
os.makedirs(app.config["SIGNATURE_UPLOAD_FOLDER"], exist_ok=True)

# Optional: limit upload size to 2 MB
app.config.setdefault("MAX_CONTENT_LENGTH", 2 * 1024 * 1024)

# Initialize extensions with this app instance.
db.init_app(app)
migrate.init_app(app, db)
login_manager.init_app(app)

# Temporary login view; we’ll create this route later.
login_manager.login_view = "forum.login"

# Register the forum blueprint at /forum.
from forum import forum_bp  # noqa: E402

app.register_blueprint(forum_bp, url_prefix="/forum")


# --- Lead capture model (consult + newsletter) ---------------------------------


class Lead(db.Model):
    __tablename__ = "leads"
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(32), nullable=False)  # consult | subscribe
    name = db.Column(db.String(255))
    email = db.Column(db.String(255), nullable=False, index=True)
    phone = db.Column(db.String(64))
    size = db.Column(db.String(128))
    crop = db.Column(db.String(128))
    message = db.Column(db.Text)
    source = db.Column(db.String(64))
    ip = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_row(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name or "",
            "email": self.email or "",
            "phone": self.phone or "",
            "size": self.size or "",
            "crop": self.crop or "",
            "message": self.message or "",
            "source": self.source or "",
            "ip": self.ip or "",
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        }


with app.app_context():
    db.create_all()

# Stripe configuration (TEST MODE). Set STRIPE_SECRET_KEY and STRIPE_PUBLISHABLE_KEY in your env or .env


inf = sys.maxsize
# Known solutions for layers l=1 to l=8 (corresponding to n=3 to n=10)
known_solutions = {
    1: [('reverse_L', 'o3'), ('reverse_L', 'o1')],
    2: [('linear3', 'o1'), ('linear3', 'o2'), ('linear3', 'o1'), ('linear3', 'o2')],
    3: [('L', 'o2'), ('L', 'o3'), ('L', 'o4'), ('L', 'o1')],
    4: [('linear3', 'o1'), ('linear3', 'o2'), ('L', 'o3'), ('reverse_L', 'o1'), ('linear3', 'o2'), ('linear3', 'o1')],
    5: [('L', 'o2'), ('linear3', 'o2'), ('linear3', 'o1'), ('linear3', 'o1'), ('linear4', 'o2'), ('reverse_L', 'o2'), ('linear3', 'o1')],
    6: [('L', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear3', 'o1'), ('linear4', 'o2'), ('linear3', 'o2'), ('linear3', 'o1'), ('linear3', 'o1')],
    7: [('linear4', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear4', 'o1'), ('linear4', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear4', 'o1')],
    8: [('linear4', 'o2'), ('linear4', 'o2'), ('L', 'o3'), ('linear4', 'o1'), ('reverse_L', 'o1'), ('linear4', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear4', 'o1')]
}
prices = {
    'L': 308,
    'reverse_L': 308,
    'linear4': 308,
    'linear3': 231,
    'linear2': 154, # Assumed price for 2-COB linear module
    'centerpiece': 385
}

def summarize_pricing(layout_data):
    """
    Given layout_data from compute_layout(return_data=True), return a breakdown of modules and costs.
    """
    counts = Counter()
    for mtype, _, _ in layout_data.get("module_groups", []):
        counts[mtype] += 1

    items = []
    subtotal = 0
    for mtype, count in sorted(counts.items()):
        price = prices.get(mtype, 0)
        line = price * count
        subtotal += line
        items.append({
            "name": mtype,
            "count": count,
            "unit_price": price,
            "line_total": line,
        })
    return items, subtotal

def _to_data_url(fig) -> str:
    """Serialize a matplotlib figure to a base64 data URL."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _render_surface_and_scatter(csv_path: Path) -> tuple[str | None, str | None, dict]:
    """Create surface and scatter data URLs from a PPFD CSV and return grid stats."""
    if not csv_path.exists():
        return None, None, {}
    try:
        import csv as _csv
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        return None, None, {}

    vals = {}
    xs_set, ys_set = set(), set()
    for r in rows:
        try:
            x = float(r["x"]); y = float(r["y"]); v = float(r["ppfd"])
            vals[(y, x)] = v
            xs_set.add(x); ys_set.add(y)
        except Exception:
            continue
    if not vals:
        return None, None, {}

    ux, uy = sorted(xs_set), sorted(ys_set)
    arr = np.array([[vals.get((y, x), 0.0) for x in ux] for y in uy], dtype=float)

    X, Y = np.meshgrid(ux, uy, indexing="xy")
    surface_url = scatter_url = None
    stats = {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "uniformity": float(np.min(arr) / max(np.mean(arr), 1e-9)),
    }
    # Widen Z range so surfaces/scatter appear less spiky; center around mean
    z_pad = max(200.0, 0.35 * stats["mean"])
    z_min = max(0.0, stats["mean"] - z_pad)
    z_max = stats["mean"] + z_pad

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(6, 4), dpi=160)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, arr, cmap="viridis", linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("PPFD")
        ax.set_zlim(z_min, z_max)
        surface_url = _to_data_url(fig)
    except Exception:
        surface_url = None

    try:
        fig = plt.figure(figsize=(6, 4), dpi=160)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X.flatten(), Y.flatten(), arr.flatten(), c=arr.flatten(), cmap="plasma", s=14, vmin=z_min, vmax=z_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("PPFD")
        ax.set_zlim(z_min, z_max)
        scatter_url = _to_data_url(fig)
    except Exception:
        scatter_url = None

    return surface_url, scatter_url, stats

def fetch_radiance_report(length_ft: float, width_ft: float, target_ppfd: float = 900.0):
    """
    Best-effort bridge into radsim.solver to produce real stats/media for the report.
    Returns dict with stats + URLs or None on failure.
    """
    try:
        from radsim import solver as rsolver
        reflectance = "MYLAR90"
        pre = rsolver._precompute_cell(length_ft, width_ft, reflectance, write_radiance_files=True)
        if not pre.get("ok"):
            return None

        cell_key = pre.get("cell_key")
        solve = rsolver._run_ppfd_solve(cell_key, target_ppfd, 10, 100, "fast")
        solve_stats = solve.get("solve") or {}

        def rel_to_url(rel: str | None) -> str | None:
            return None if not rel else f"/files/{rel.lstrip('/')}"

        heatmap_url = rel_to_url(solve.get("png_rel"))
        csv_rel = solve.get("csv_rel")
        surface_url = scatter_url = None
        grid_stats = {}
        if csv_rel:
            csv_path = Path(rsolver.PROJECT_ROOT) / csv_rel
            surface_url, scatter_url, grid_stats = _render_surface_and_scatter(csv_path)

        avg = solve_stats.get("mean") or grid_stats.get("mean")
        min_ppfd = grid_stats.get("min")
        max_ppfd = grid_stats.get("max")
        uni = solve_stats.get("dou_percent")
        if uni is None and avg and min_ppfd:
            uni = (min_ppfd / avg) * 100.0
        return {
            "cell_key": cell_key,
            "heatmap_url": heatmap_url,
            "csv_url": rel_to_url(csv_rel),
            "json_url": rel_to_url(solve.get("json_rel")),
            "stats": {
                "mean": avg,
                "std": solve_stats.get("std"),
                "cv": solve_stats.get("cv"),
                "dou_percent": uni,
                "min": min_ppfd,
                "max": max_ppfd,
                "uniformity": grid_stats.get("uniformity"),
                "target": target_ppfd,
            },
            "surface_url": surface_url,
            "scatter_url": scatter_url,
        }
    except Exception as e:
        print("Radiance report generation failed:", e)
        return None

# app_email.py (or inside app.py)
import os, datetime as dt
from flask import request, jsonify
from dotenv import load_dotenv
load_dotenv()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

try:
    import requests
except Exception as e:
    requests = None
    print("WARN: python-requests not importable:", e)

@app.post("/api/consult")
def api_consult():
    data = request.get_json(silent=True) or {}

    # Honeypot: if bots fill it, pretend success but do nothing
    if data.get("hp"):
        return jsonify(ok=True), 200

    name   = (data.get("name") or "").strip()
    email  = (data.get("email") or "").strip()
    phone  = (data.get("phone") or "").strip()
    size   = (data.get("size") or data.get("grow_space") or "").strip()
    crop   = (data.get("crop") or "").strip()
    qs     = (data.get("questions") or data.get("message") or "").strip()
    source = (data.get("source") or "consult").strip()
    ip_addr= request.headers.get('X-Forwarded-For') or request.remote_addr

    if not name or "@" not in email:
        return jsonify(ok=False, error="Missing or invalid name/email"), 400

    # Persist locally so we are not blocked by Brevo.
    try:
        lead = Lead(
            kind="consult",
            name=name,
            email=email,
            phone=phone,
            size=size,
            crop=crop,
            message=qs,
            source=source,
            ip=ip_addr,
        )
        db.session.add(lead)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("ERROR: failed to store consult lead:", repr(e))
        return jsonify(ok=False, error="Could not store submission."), 500

    # Best-effort Brevo (do not block on failure)
    api_key = os.getenv("BREVO_API_KEY")
    sender  = os.getenv("BREVO_SENDER", "austin@luminousphotonics.com")
    brevo_error = None
    if api_key and requests is not None:
        body = "\n".join([
            f"[Consultation] {name} — {size or 'N/A'}",
            f"Name: {name}",
            f"Email: {email}",
            f"Phone: {phone or 'N/A'}",
            f"Grow Space Size: {size or 'N/A'}",
            f"Crop Type: {crop or 'N/A'}",
            "",
            "Questions:",
            (qs or "—"),
            "",
            f"Submitted: {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            f"IP: {ip_addr}",
        ])
        try:
            resp = requests.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={
                    "api-key": api_key,
                    "accept": "application/json",
                    "content-type": "application/json",
                },
                json={
                    "sender": {"name": "Luminous Photonics", "email": sender},
                    "to": [
                        {"email": "austin@luminousphotonics.com"},
                        {"email": "annemarie.rouse@gmail.com"},
                    ],
                    "replyTo": {"email": email, "name": name},
                    "subject": f"[Consultation] {name} — {size or 'Grow details'}",
                    "textContent": body,
                },
                timeout=15,
            )
            print("Brevo response:", resp.status_code, resp.text[:500])
            if resp.status_code >= 300:
                brevo_error = f"Brevo {resp.status_code}"
        except Exception as e:
            brevo_error = f"Brevo error: {e}"
            print("ERROR: Brevo call failed:", repr(e))

    return jsonify(ok=True, stored=True, brevo_error=brevo_error), 200


@app.post("/api/subscribe")
def api_subscribe():
    data = request.get_json(silent=True) or {}

    # Honeypot field
    if data.get("hp"):
        return jsonify(ok=True), 200

    email = (data.get("email") or "").strip()
    name = (data.get("name") or "").strip()
    source = (data.get("source") or "newsletter").strip()
    ip_addr = request.headers.get('X-Forwarded-For') or request.remote_addr

    if not email or "@" not in email:
        return jsonify(ok=False, error="Missing or invalid email"), 400

    # Persist locally first
    try:
        lead = Lead(
            kind="subscribe",
            name=name,
            email=email,
            source=source,
            ip=ip_addr,
        )
        db.session.add(lead)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("ERROR: failed to store subscribe lead:", repr(e))
        return jsonify(ok=False, error="Could not store submission."), 500

    # Best-effort Brevo notification
    api_key = os.getenv("BREVO_API_KEY")
    sender = os.getenv("BREVO_SENDER", "austin@luminousphotonics.com")
    brevo_error = None
    if api_key and requests is not None:
        body = "\n".join(
            [
                f"[Newsletter] {email}",
                f"Name: {name or 'N/A'}",
                f"Email: {email}",
                f"Source: {source}",
                "",
                f"Submitted: {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}",
                f"IP: {ip_addr}",
            ]
        )
        try:
            resp = requests.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={
                    "api-key": api_key,
                    "accept": "application/json",
                    "content-type": "application/json",
                },
                json={
                    "sender": {"name": "Luminous Photonics", "email": sender},
                    "to": [{"email": "austin@luminousphotonics.com"}],
                    "replyTo": {"email": email, "name": name or email},
                    "subject": f"[Newsletter] {email}",
                    "textContent": body,
                },
                timeout=15,
            )
            print("Brevo subscribe response:", resp.status_code, resp.text[:500])
            if resp.status_code >= 300:
                brevo_error = f"Brevo {resp.status_code}"
        except Exception as e:
            brevo_error = f"Brevo error: {e}"
            print("ERROR: Brevo subscribe call failed:", repr(e))

    return jsonify(ok=True, stored=True, brevo_error=brevo_error), 200


@app.get("/leads/<token>")
def view_leads(token: str):
    secret = os.getenv("LEADS_ADMIN_TOKEN") or ""
    if not secret or token != secret:
        abort(404)

    leads = Lead.query.order_by(Lead.created_at.desc()).limit(500).all()
    rows = [l.to_row() for l in leads]
    return render_template_string(
        """
        <!doctype html>
        <title>Leads</title>
        <style>
          body{font-family:Arial,sans-serif;background:#0b0b0b;color:#eee;padding:20px;}
          table{width:100%;border-collapse:collapse;margin-top:10px;}
          th,td{border:1px solid #333;padding:8px;text-align:left;font-size:14px;}
          th{background:#1a1a1a;color:#ffd700;position:sticky;top:0;}
          tr:nth-child(even){background:#111;}
          tr:nth-child(odd){background:#0c0c0c;}
          code{color:#9be7ff;}
        </style>
        <h2>Leads (latest first)</h2>
        <p>Total: {{rows|length}}</p>
        <table>
          <tr>
            <th>ID</th><th>Kind</th><th>Name</th><th>Email</th><th>Phone</th><th>Size</th><th>Crop</th><th>Message</th><th>Source</th><th>IP</th><th>Created (UTC)</th>
          </tr>
          {% for r in rows %}
          <tr>
            <td>{{r.id}}</td>
            <td>{{r.kind}}</td>
            <td>{{r.name}}</td>
            <td><code>{{r.email}}</code></td>
            <td>{{r.phone}}</td>
            <td>{{r.size}}</td>
            <td>{{r.crop}}</td>
            <td style="max-width:240px;white-space:pre-wrap;">{{r.message}}</td>
            <td>{{r.source}}</td>
            <td>{{r.ip}}</td>
            <td>{{r.created_at}}</td>
          </tr>
          {% endfor %}
        </table>
        """,
        rows=rows,
    )

@app.get("/files/<path:rel>")
def files_root(rel: str):
    safe_rel = os.path.normpath(rel).lstrip("/")
    full = os.path.join(PROJECT_ROOT, safe_rel)

    pr = os.path.realpath(PROJECT_ROOT)
    fp = os.path.realpath(full)
    if not (fp == pr or fp.startswith(pr + os.sep)):
        abort(403)
    if not os.path.isfile(fp):
        abort(404)

    dirname, filename = os.path.split(full)
    resp = send_from_directory(dirname, filename)
    if filename.lower().endswith(".html"):
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    return resp
  
@app.after_request
def add_csp(resp):
    # allow same-origin frames only
    resp.headers.setdefault("Content-Security-Policy", "frame-ancestors 'self'")
    return resp


# --- JSON 405 everywhere (so you never see HTML error pages) ---
@app.errorhandler(405)
def _json_405(_e):
    return jsonify(ok=False, error="Method Not Allowed"), 405

# --- Catch-all OPTIONS to prevent preflight 405s on unknown routes ---
@app.route("/<path:_any>", methods=["OPTIONS"])
def _any_options(_any):
    return Response("", 204)

# --- Passthroughs for competitor endpoints (root -> mounted /solver/...) ---
@app.route("/competitor/verify", methods=["GET", "POST", "OPTIONS"])
def _competitor_verify_passthrough():
    if request.method == "OPTIONS":
        return Response("", 204)
    # 307 preserves method/body for POST
    return redirect("/solver/competitor/verify", code=307)

@app.route("/competitor/precompute", methods=["GET", "POST", "OPTIONS"])
def _competitor_precompute_passthrough():
    if request.method == "OPTIONS":
        return Response("", 204)
    return redirect("/solver/competitor/precompute", code=307)

def get_min_fixtures(r, corner_set, known_l=None):
    if known_l is not None and known_l in known_solutions:
        return len(known_solutions[known_l]), known_solutions[known_l]
   
    max_size = 4
    extended = r + max_size
    DP = [inf] * (extended + 1)
    prev = [None] * (extended + 1)
    DP[r] = 0
    for pos in range(r - 1, -1, -1):
        if pos + 2 <= extended:
            has_corner_middle = any((pos + j) % r in corner_set for j in range(1, 2))
            if not has_corner_middle:
                new_val = 1 + DP[pos + 2]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (2, 'linear2', 'o1')
        if pos + 3 <= extended:
            has_corner_middle = any((pos + j) % r in corner_set for j in range(1, 3))
            if not has_corner_middle:
                new_val = 1 + DP[pos + 3]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (3, 'linear3', 'o1')
        if pos + 4 <= extended:
            has_corner_middle = any((pos + j) % r in corner_set for j in range(1, 4))
            if not has_corner_middle:
                new_val = 1 + DP[pos + 4]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (4, 'linear4', 'o1')
        if pos + 4 <= extended:
            c = (pos + 1) % r
            m = (pos + 2) % r
            if c in corner_set and m not in corner_set:
                new_val = 1 + DP[pos + 4]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (4, 'L', 'o1')
        if pos + 4 <= extended:
            c = (pos + 2) % r
            m = (pos + 1) % r
            if c in corner_set and m not in corner_set:
                new_val = 1 + DP[pos + 4]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (4, 'reverse_L', 'o1')
   
    if DP[0] == inf:
        return inf, []
   
    modules = []
    pos = 0
    while pos < r and prev[pos] is not None:
        size, mtype, orient = prev[pos]
        modules.append((mtype, orient))
        pos += size
    return DP[0], modules
def get_n_from_dimensions(d_ft):
    if d_ft <= 4:
        return 2
    return floor((d_ft - 4) / 2) + 2
def get_ring_positions(l):
    d = l + 1
    positions = []
    for i in range(0, d + 1):
        positions.append((i, d - i))
    for j in range(1, d + 1):
        positions.append((d - j, -j))
    for k in range(1, d + 1):
        positions.append((-k, -d + k))
    for p in range(1, d + 1):
        positions.append((-d + p, p))
    positions = list(dict.fromkeys(positions))
    return positions
def get_ring_positions_rect(k, offset):
    u_max = offset + k
    v_max = k
    left = []
    u = -u_max
    for v in range(-v_max, v_max + 1):
        if (u + v) % 2 == 0:
            left.append((u, v))
    len_left = len(left)
    top = []
    v = v_max
    for u in range(-u_max + 1, u_max + 1):
        if (u + v) % 2 == 0:
            tup = (u, v)
            if tup not in [p for p in left]:
                top.append(tup)
    len_top = len(top)
    right = []
    u = u_max
    for v in range(v_max - 1, -v_max - 1, -1):
        if (u + v) % 2 == 0:
            tup = (u, v)
            if tup not in [p for p in left + top]:
                right.append(tup)
    len_right = len(right)
    bottom = []
    v = -v_max
    for u in range(u_max - 1, -u_max - 1, -1):
        if (u + v) % 2 == 0:
            tup = (u, v)
            if tup not in [p for p in left + top + right]:
                bottom.append(tup)
    len_bottom = len(bottom)
    ring_pos = left + top + right + bottom
    return ring_pos, len_left, len_top, len_right, len_bottom
def get_central_positions(offset):
    positions = []
    for u in range(-offset, offset + 1):
        if (u + 0) % 2 == 0:
            positions.append((u, 0))
    return positions
def get_central_modules(offset):
    central_cobs = get_central_positions(offset)
    num_c = len(central_cobs)
    number4 = num_c // 4
    rem = num_c % 4
    number3 = 0
    number2 = 0
    if rem == 1:
        if number4 >= 1:
            number4 -= 1
            number3 = 1
            number2 = 1
        else:
            number3 = 1
            number2 = num_c - 3
    elif rem == 2:
        if number4 >= 1:
            number4 -= 1
            number3 = 2
        else:
            number2 = 1
    elif rem == 3:
        number3 = 1
    central_modules = []
    pos = 0
    for _ in range(number4):
        cobs = central_cobs[pos:pos + 4]
        central_modules.append(('linear4', 'o2', cobs))
        pos += 4
    for _ in range(number3):
        cobs = central_cobs[pos:pos + 3]
        central_modules.append(('linear3', 'o2', cobs))
        pos += 3
    for _ in range(number2):
        cobs = central_cobs[pos:pos + 2]
        central_modules.append(('linear2', 'o2', cobs))
        pos += 2
    return central_modules
def get_module_cobs(r, modules, ring_pos):
    module_groups = []
    pos = 0
    used = set()
    for mtype, orient in modules:
        if mtype == 'linear2':
            indices = [pos % r, (pos + 1) % r]
            if all(i not in used for i in indices):
                cobs = [ring_pos[i] for i in indices]
                module_groups.append((mtype, orient, cobs))
                used.update(indices)
                pos += 2
        elif mtype == 'linear3':
            indices = [pos % r, (pos + 1) % r, (pos + 2) % r]
            if all(i not in used for i in indices):
                cobs = [ring_pos[i] for i in indices]
                module_groups.append((mtype, orient, cobs))
                used.update(indices)
                pos += 3
        elif mtype in ['linear4', 'L', 'reverse_L']:
            indices = [pos % r, (pos + 1) % r, (pos + 2) % r, (pos + 3) % r]
            if all(i not in used for i in indices):
                cobs = [ring_pos[i] for i in indices]
                module_groups.append((mtype, orient, cobs))
                used.update(indices)
                pos += 4
    return module_groups
def get_extension_groups(base_n, num_steps, shift_x=0.0, shift_y=0.0):
    extension_groups = []
    for i in range(1, num_steps + 1):
        k = base_n + i
        lower = ceil(i / 2)
        upper = floor((2 * base_n + i) / 2)
        num_cobs = upper - lower + 1 if upper >= lower else 0
        if num_cobs < 2:
            continue
        rem = num_cobs % 4
        number4 = num_cobs // 4
        number3 = 0
        number2 = 0
        if rem == 1:
            if number4 >= 1:
                number4 -= 1
                number3 = 1
                number2 = 1
            else:
                number3 = 1
                number2 = num_cobs - 3
        elif rem == 2:
            if number4 >= 1:
                number4 -= 1
                number3 = 2
            else:
                number2 = 1
        elif rem == 3:
            number3 = 1
        modules_for_column = []
        for _ in range(number4):
            modules_for_column.append(('linear4', 'o1'))
        for _ in range(number3):
            modules_for_column.append(('linear3', 'o1'))
        for _ in range(number2):
            modules_for_column.append(('linear2', 'o1'))
        cobs = [(float(x) + shift_x, float(k - x) + shift_y) for x in range(lower, upper + 1)]
        pos = 0
        for mtype, orient in modules_for_column:
            size = int(mtype[-1])
            group_cobs = cobs[pos:pos + size]
            extension_groups.append((mtype, orient, group_cobs))
            pos += size
    return extension_groups

def generate_visuals_general(all_positions, module_groups, center, rings, cv_enabled, tutorial, wrap_flat_group=False):
    from math import sqrt
    import numpy as np
    import matplotlib.pyplot as plt

    def _square(px, py, size=10, fill="white"):
        half = size / 2
        return f'<rect x="{px - half:.2f}" y="{py - half:.2f}" width="{size}" height="{size}" fill="{fill}" />\n'

    scale = 22 / sqrt(2)
    plot_points = [(scale * (x + y), scale * (x - y)) for x, y in all_positions]

    min_x = min(p[0] for p in plot_points) - 100 if plot_points else -100
    min_y = min(p[1] for p in plot_points) - 100 if plot_points else -100
    max_x = max(p[0] for p in plot_points) + 100 if plot_points else 100
    max_y = max(p[1] for p in plot_points) + 100 if plot_points else 100
    width = max_x - min_x
    height = max_y - min_y

    svg = f'<svg id="layout-svg" viewBox="{min_x} {min_y} {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    if tutorial:
        svg += (
            '<defs>'
            '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">'
            '<polygon points="0 0, 10 3.5, 0 7" fill="#FFD700"/>'
            '</marker>'
            '</defs>\n'
        )

    colors = ['#' + ''.join(f'{int(c*255):02x}' for c in plt.cm.tab20(i)[:3]) for i in np.linspace(0, 1, 20)]
    color_idx = 0

    if tutorial and center and rings:
        # Center group
        if center:
            initial_style = ' style="opacity:0; transform: translateX(-2000px);"' if tutorial else ''
            svg += f'<g id="ring0"{initial_style}>\n'
            mtype, orient, cobs = center[0]
            rotated = [(scale * (a + b), scale * (a - b)) for a, b in cobs]
            for px, py in rotated:
                svg += _square(px, py, size=10, fill="white")
            color = colors[color_idx % len(colors)]
            color_idx += 1
            cx, cy = rotated[0]
            for rx, ry in rotated[1:]:
                svg += f'<line x1="{cx}" y1="{cy}" x2="{rx}" y2="{ry}" stroke="{color}" stroke-width="2" />\n'
            svg += '</g>\n'

        # Ring groups
        for ring_idx, ring in enumerate(rings):
            initial_style = ' style="opacity:0; transform: translateX(-2000px);"' if tutorial else ''
            svg += f'<g id="ring{ring_idx+1}"{initial_style}>\n'
            for mtype, orient, cobs in ring:
                rotated = [(scale * (a + b), scale * (a - b)) for a, b in cobs]
                for px, py in rotated:
                    svg += _square(px, py, size=10, fill="white")
                color = colors[color_idx % len(colors)]
                color_idx += 1
                if 'linear' in mtype:
                    if len(rotated) > 1:
                        path = ' '.join(f'{x},{y}' for x, y in rotated)
                        svg += f'<polyline points="{path}" stroke="{color}" stroke-width="2" fill="none" />\n'
                else:  # L, reverse_L
                    if mtype == 'L':
                        long = rotated[1:4]
                        short = rotated[0:2]
                    else:
                        long = rotated[0:3]
                        short = rotated[2:4]
                    l_path = ' '.join(f'{x},{y}' for x, y in long)
                    s_path = ' '.join(f'{x},{y}' for x, y in short)
                    svg += f'<polyline points="{l_path}" stroke="{color}" stroke-width="2" fill="none" />\n'
                    svg += f'<polyline points="{s_path}" stroke="{color}" stroke-width="2" fill="none" />\n'
            svg += '</g>\n'
    else:
        flat_content = ''
        for px, py in plot_points:
            flat_content += _square(px, py, size=10, fill="white")
        for mtype, orient, cobs in module_groups:
            rotated = [(scale * (a + b), scale * (a - b)) for a, b in cobs]
            color = colors[color_idx % len(colors)]
            if mtype == 'centerpiece':
                cx, cy = rotated[0]
                for rx, ry in rotated[1:]:
                    flat_content += f'<line x1="{cx}" y1="{cy}" x2="{rx}" y2="{ry}" stroke="{color}" stroke-width="2" />\n'
            elif 'linear' in mtype:
                if len(cobs) > 1:
                    path = ' '.join(f'{x},{y}' for x, y in rotated)
                    flat_content += f'<polyline points="{path}" stroke="{color}" stroke-width="2" fill="none" />\n'
            else:  # L, reverse_L
                if mtype == 'L':
                    long = rotated[1:4]
                    short = rotated[0:2]
                else:
                    long = rotated[0:3]
                    short = rotated[2:4]
                l_path = ' '.join(f'{x},{y}' for x, y in long)
                s_path = ' '.join(f'{x},{y}' for x, y in short)
                flat_content += f'<polyline points="{l_path}" stroke="{color}" stroke-width="2" fill="none" />\n'
                flat_content += f'<polyline points="{s_path}" stroke="{color}" stroke-width="2" fill="none" />\n'
            color_idx += 1
        if wrap_flat_group:
            svg += '<g id="layout-group" style="opacity:0; transform: translateX(-2000px);">\n'
            svg += flat_content
            svg += '</g>\n'
        else:
            svg += flat_content

    # Animated coverage probe (kept as circle to avoid offset issues)
    gaps = []
    if cv_enabled:
        unique_py = sorted(set(py for _, py in plot_points), reverse=True)
        for i in range(len(unique_py) - 1):
            py1 = unique_py[i]
            py2 = unique_py[i + 1]
            mid_py = (py1 + py2) / 2
            row1_px = [px for px, py in plot_points if py == py1]
            row2_px = [px for px, py in plot_points if py == py2]
            all_px = sorted(set(row1_px + row2_px))
            mid_pxs = [(all_px[j] + all_px[j+1]) / 2 for j in range(len(all_px) - 1)]
            if i % 2 == 0:
                row_gaps = [(mx, mid_py) for mx in sorted(mid_pxs)]
            else:
                row_gaps = [(mx, mid_py) for mx in sorted(mid_pxs, reverse=True)]
            gaps.extend(row_gaps)
        if gaps:
            reverse_gaps = gaps[::-1]
            path_points = gaps + reverse_gaps[1:]
            values = [path_points[0]] + [p for p in path_points[1:] for _ in (p, p)]
            values_str = ';'.join(f'{x:.2f},{y:.2f}' for x, y in values)
            num_moves = len(path_points) - 1
            move_dur = 0.5
            pause_dur = 1.0
            total_dur = num_moves * move_dur + num_moves * pause_dur
            key_times = [0.0]
            cum_time = 0.0
            for _ in range(num_moves):
                cum_time += move_dur
                key_times.append(cum_time / total_dur)
                cum_time += pause_dur
                key_times.append(cum_time / total_dur)
            key_times_str = ';'.join(f'{t:.4f}' for t in key_times)
            svg += '<circle r="6" fill="#ff0077" stroke="#000" stroke-width="1">\n'
            svg += f'<animateMotion dur="{total_dur}s" repeatCount="indefinite" calcMode="linear" keyTimes="{key_times_str}" values="{values_str}" />\n'
            svg += '</circle>\n'

    svg += '</svg>'
    return svg

def compute_layout(length_ft, width_ft, cv_enabled, tutorial=False, wrap_flat_group=False, return_data=False):
    """
    Return (svg_markup, total_cost[, layout_data])
    layout_data = {
      "all_positions": [(x,y), ...],             # XY before rotate/scale
      "module_groups": [ [mtype, orient, cobs], ... ],   # cobs: [(x,y),...]
      "center": center,                           # same shape as you already use
      "rings": rings                              # list of ring groups
    }
    """
    min_dim = min(length_ft, width_ft)
    max_dim = max(length_ft, width_ft)
    base_n = get_n_from_dimensions(min_dim)
    c = 2.0
    unit = min_dim + c
    min_rect_long = min_dim + 4
    max_s = floor((max_dim + c) / unit)
    s = max_s
    found = False
    has_rect = False
    rect_long = 0
    rem = 0
    while s >= 0 and not found:
        if s == 0:
            rect_long = max_dim
            found = True
            has_rect = True
        else:
            pure_length = min_dim * s + c * (s - 1)
            rem = max_dim - pure_length
            if rem == 0:
                found = True
                has_rect = False
            elif rem > c and (rem - c) >= min_rect_long:
                rect_long = rem - c
                found = True
                has_rect = True
            else:
                s -= 1
    shift_step = base_n + 1
    layer_modules = []
    for l in range(1, base_n):
        min_fix, modules = get_min_fixtures(4 * (l + 1), set([0, l + 1, 2 * (l + 1), 3 * (l + 1)]), l)
        layer_modules.append(modules)
    module_groups = []
    center = []
    rings = []
    for j in range(s):
        sx = j * shift_step
        sy = j * shift_step
        if base_n >= 2:
            centerpiece = [(sx, sy), (1 + sx, sy), (sx, 1 + sy), (-1 + sx, sy), (sx, -1 + sy)]
            if tutorial:
                center = [('centerpiece', 'o1', centerpiece)]
            else:
                module_groups.append(('centerpiece', 'o1', centerpiece))
        for l in range(1, base_n):
            ring_pos = get_ring_positions(l)
            ring_pos_shifted = [(xx + sx, yy + sy) for xx, yy in ring_pos]
            ring_groups = get_module_cobs(4 * (l + 1), layer_modules[l - 1], ring_pos_shifted)
            if tutorial:
                rings.append(ring_groups)
            else:
                module_groups.extend(ring_groups)
    for m in range(s - 1):
        csx = m * shift_step
        csy = m * shift_step
        connector_groups = get_extension_groups(base_n, 1, csx, csy)
        module_groups.extend(connector_groups)
    if has_rect:
        a = get_n_from_dimensions(rect_long)
        offset = a - base_n
        if offset % 2 == 1:
            a -= 1
            offset -= 1
        if s > 0:
            last_connector_u = base_n + 1 + (s - 1) * 2 * (base_n + 1)
            last_csx = (s - 1) * shift_step
            last_csy = (s - 1) * shift_step
            last_connector_groups = get_extension_groups(base_n, 1, last_csx, last_csy)
            module_groups.extend(last_connector_groups)
            shift_u = last_connector_u + 1 + (offset + base_n)
        else:
            shift_u = 0
        shift_v = 0
        rect_modules = get_central_modules(offset)
        for k in range(1, base_n + 1):
            ring_pos, len_left, len_top, len_right, len_bottom = get_ring_positions_rect(k, offset)
            r = len(ring_pos)
            corner_set = set([0, len_left - 1, len_left + len_top - 1, len_left + len_top + len_right - 1])
            min_fix, modules = get_min_fixtures(r, corner_set)
            ring_groups = get_module_cobs(r, modules, ring_pos)
            rect_modules.extend(ring_groups)
        for i in range(len(rect_modules)):
            mtype, orient, local_cobs = rect_modules[i]
            transformed_cobs = [((u + shift_u + v + shift_v) / 2, (u + shift_u - (v + shift_v)) / 2) for u, v in local_cobs]
            rect_modules[i] = (mtype, orient, transformed_cobs)
        module_groups.extend(rect_modules)
    all_positions = []
    for _, _, cobs in module_groups:
        all_positions.extend(cobs)
    all_positions = list(set(all_positions))
    if tutorial:
        module_groups = center + [m for r in rings for m in r]
    total_cost = 0
    total_modules = Counter()
    for mtype, _, _ in module_groups:
        total_modules[mtype] += 1
    for mtype, count in total_modules.items():
        total_cost += count * prices.get(mtype, 0)
    svg = generate_visuals_general(all_positions, module_groups, center, rings, cv_enabled, tutorial, wrap_flat_group)
    if return_data:
        layout_data = {
            "all_positions": all_positions,
            "module_groups": module_groups,
            "center": center,
            "rings": rings,
        }
        return svg, total_cost, layout_data
    return svg, total_cost



from flask import request, render_template_string

ASSEMBLY_TPL = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>3D Assembly</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root { --accent:#ffd700; --panel: rgba(255,255,255,.06); --border: rgba(255,255,255,.12); }
    * { box-sizing: border-box; }
    html, body { margin:0; height:100%; background:#0b0b0b; color:#fff; font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
    .topbar { width:100vw; padding:12px 16px; display:flex; align-items:center; justify-content:space-between; }
    .topbar h2 { margin:0; color:var(--accent); font-weight:800; }
    .btn { background:var(--accent); color:#000; border:0; padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:800; }
    .stage-wrap { position:relative; width:100vw; height:calc(100vh - 56px); overflow:hidden; }
    svg#stage { position:absolute; inset:0; width:100%; height:100%; display:block; background:#0e0e0e; transition: opacity .35s ease; }
    #glwrap { position:absolute; inset:0; width:100%; height:100%; opacity:0; pointer-events:none; transition: opacity .35s ease; background: {{ bg }}; }
    #glwrap.ready { opacity:1; pointer-events:auto; }
    .hud { position:fixed; left:12px; bottom:12px; display:flex; gap:10px; flex-wrap:wrap; z-index:20; }
    .tag { background:var(--panel); border:1px solid var(--border); padding:6px 10px; border-radius:999px; font-weight:600; color:#ddd; }
    .gl-ui { position: fixed; right:12px; bottom:12px; display:flex; gap:8px; z-index:25; }
    .gl-ui .btn { padding:8px 10px; border-radius:10px; }
    #gl { width:100%; height:100%; display:block; }

    /* — embed mode (for splash iframe) — */
    body.embed { margin:0 !important; background:#0b0b0b !important; overflow:hidden !important; }
    body.embed .topbar,
    body.embed .hud,
    body.embed .gl-ui { display:none !important; }
    body.embed .stage-wrap { width:100vw; height:100vh; }

  </style>
</head>
<body class="{{ 'embed' if embed else '' }}">
  <div class="topbar">
    <h2>3D Assembly</h2>
    <div><button class="btn" onclick="history.back()">Back</button></div>
  </div>

  <div class="stage-wrap">
    <!-- 2D Assembly -->
    <svg id="stage"
         viewBox="{{ viewbox }}"
         preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg"
         aria-label="Assembly Canvas">
      <g id="asm-root"></g>
    </svg>

    <!-- 3D Scene (hidden until 2D finishes) -->
    <div id="glwrap">
      <canvas id="gl"></canvas>
    </div>

    <div class="hud">
      <div class="tag">Grow space: {{ length_ft }}' × {{ width_ft }}'</div>
      <div class="tag" id="progress">Assembling… 0 / 0</div>
    </div>
    <div class="gl-ui" style="display:none" id="glui">
      <button class="btn" id="replay3d">Replay</button>
      <button class="btn" id="reset3d">Reset View</button>
    </div>
  </div>

  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
    }
  }
  </script>

  <!-- Single ES module: 2D assemble, then 3D -->
  <script type="module">
    // ===== Imports =====
    import * as THREE from 'three';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { OBJLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/OBJLoader.js';

    // ===== Shared params/data =====
    const fixtures = {{ fixtures|tojson }};
    const ROOM = { Lft: {{ length_ft }}, Wft: {{ width_ft }} };
    const params = {
      plantURL: {{ plant_url|tojson }},
      plantSpacing: {{ plant_spacing }},
      targetHeight: {{ target_height }},
      bg: {{ bg|tojson }}
    };

    // ===== 2D assembly =====
    const NS = "http://www.w3.org/2000/svg";
    const stageSvg   = document.getElementById("stage");
    const asmRoot    = document.getElementById("asm-root");
    const progressEl = document.getElementById("progress");
    const MARKER_SIZE = {{ marker_size }};
    const STROKE_W    = {{ stroke_w }};

    function sq(x,y,size=MARKER_SIZE,fill="#fff"){
      const h=size/2; const r=document.createElementNS(NS,"rect");
      r.setAttribute("x",(x-h).toFixed(2)); r.setAttribute("y",(y-h).toFixed(2));
      r.setAttribute("width",size); r.setAttribute("height",size); r.setAttribute("fill",fill);
      return r;
    }
    function poly(pts,color="#54a0ff"){
      const pl=document.createElementNS(NS,"polyline");
      pl.setAttribute("points",pts.map(p=>p.join(",")).join(" "));
      pl.setAttribute("stroke",color);
      pl.setAttribute("stroke-width",STROKE_W);
      pl.setAttribute("fill","none");
      return pl;
    }
    function line(x1,y1,x2,y2,color="#54a0ff"){
      const ln=document.createElementNS(NS,"line");
      ln.setAttribute("x1",x1);
      ln.setAttribute("y1",y1); // IMPORTANT: y1 (not y)
      ln.setAttribute("x2",x2);
      ln.setAttribute("y2",y2);
      ln.setAttribute("stroke",color);
      ln.setAttribute("stroke-width",STROKE_W);
      return ln;
    }

    function groupFixture(f){
      const g=document.createElementNS(NS,"g");
      f.targets.forEach(([x,y])=>g.appendChild(sq(x,y,MARKER_SIZE,"#fff")));
      if(f.mtype==="centerpiece" && f.targets.length>=2){
        const [cx,cy]=f.targets[0];
        for(let i=1;i<f.targets.length;i++){
          const [rx,ry]=f.targets[i];
          g.appendChild(line(cx,cy,rx,ry,f.color));
        }
      } else if (f.targets.length>1){
        g.appendChild(poly(f.targets,f.color));
      }
      return g;
    }

    function bounds(fixt){
      let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
      fixt.forEach(f=>f.targets.forEach(([x,y])=>{
        if(x<minx)minx=x; if(x>maxx)maxx=x; if(y<miny)miny=y; if(y>maxy)maxy=y;
      }));
      return {w:maxx-minx,h:maxy-miny,cx:(minx+maxx)/2,cy:(miny+maxy)/2};
    }
    function spawnPos(i, vb){
      const [vx,vy,vw,vh]=vb; const r=Math.hypot(vw,vh)*0.6; const th=(i*137.508)*Math.PI/180;
      const cx=vx+vw/2, cy=vy+vh/2; return {x:cx+r*Math.cos(th), y:cy+r*Math.sin(th)};
    }
    function centroid(pts){
      let sx=0,sy=0; for(const [x,y] of pts){sx+=x;sy+=y}
      const n=Math.max(pts.length,1); return {x:sx/n,y:sy/n};
    }
    function tween(g, from, to, s0, s1, dur=900, delay=0, ease=(t)=>t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2){
      let start=null;
      function step(ts){
        if(!start) start=ts;
        const t=Math.min(1,(ts-start)/dur), k=ease(t);
        const x=from.x+(to.x-from.x)*k, y=from.y+(to.y-from.y)*k, s=s0+(s1-s0)*k;
        g.setAttribute("transform",`translate(${x.toFixed(2)}, ${y.toFixed(2)}) scale(${s})`);
        if(t<1) requestAnimationFrame(step);
      }
      setTimeout(()=>requestAnimationFrame(step), delay);
    }

    // Guard: only hand off to 3D ONCE, after last tween
    let launched3D = false;
    const launch3DOnce = ()=>{ if (launched3D) return; launched3D = true; start3D(); };

    (function assemble2D(){
      progressEl.textContent = `Assembling… 0 / ${fixtures.length}`;
      const b = bounds(fixtures);
      const vb = stageSvg.getAttribute("viewBox").split(" ").map(Number);
      const [,,vb_w,vb_h]=vb;
      const pxW = stageSvg.clientWidth||window.innerWidth, pxH = stageSvg.clientHeight||window.innerHeight;
      const u2pxW = (u)=> (u/vb_w)*pxW, u2pxH = (u)=> (u/vb_h)*pxH;
      const FILL=0.94;
      let scale = Math.min((pxW*FILL)/Math.max(1,u2pxW(b.w)), (pxH*FILL)/Math.max(1,u2pxH(b.h)));
      scale = Math.max(0.9, Math.min(scale, 6.5));
      asmRoot.setAttribute("transform", `translate(${(b.cx*(1-scale)).toFixed(2)}, ${(b.cy*(1-scale)).toFixed(2)}) scale(${scale})`);

      if (!fixtures.length) { setTimeout(launch3DOnce, 100); return; }

      const BASE=900, STEP=240, TWEEN=800;
      fixtures.forEach((f,i)=>{
        const g=groupFixture(f); asmRoot.appendChild(g);
        const sp=spawnPos(i, vb), cent=centroid(f.targets);
        const tx=sp.x-cent.x, ty=sp.y-cent.y;
        g.setAttribute("transform",`translate(${tx.toFixed(2)}, ${ty.toFixed(2)}) scale(0.85)`);
        const delay=BASE+i*STEP;
        tween(g,{x:tx,y:ty},{x:0,y:0},0.85,1,TWEEN,delay);

        setTimeout(()=>{
          const done=Math.min(i+1,fixtures.length);
          progressEl.textContent = done===fixtures.length
            ? `Assembly complete ✓ (${done})`
            : `Assembling… ${done} / ${fixtures.length}`;
        }, delay+TWEEN+20);
      });

      // One definitive handoff after all tweens complete
      const totalDelay = BASE + (fixtures.length-1)*STEP + TWEEN + 380;
      setTimeout(launch3DOnce, totalDelay);
      // Tell parent (splash) that 2D assembly is done
      try{ setTimeout(()=>window.parent?.postMessage({type:'lp-assembly-phase', phase:'2d-complete'}, '*'), totalDelay + 12); }catch(_){}

    })();

    // ===== 3D viewer =====
    const glwrap    = document.getElementById('glwrap');
    const glui      = document.getElementById('glui');
    const replayBtn = document.getElementById('replay3d');
    const wireBtn   = document.getElementById('wire3d');
    const resetBtn  = document.getElementById('reset3d');

    // ===== 3D state =====
    let renderer, scene, camera, controls;
    let plantContainer, plantInst, fixtureGroup, canopyGroup;
    let fixtureLight, lightTarget;
    let wireframe = false, rafId = null;

    const easeOutCubic  = t => 1 - Math.pow(1 - t, 3);
    const easeInOutQuad = t => (t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2);

    function setupRenderer(){
      const canvas = document.getElementById('gl');
      renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(glwrap.clientWidth, glwrap.clientHeight, true);
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.domElement.style.width  = '100%';
      renderer.domElement.style.height = '100%';

      scene = new THREE.Scene();
      scene.background = new THREE.Color(params.bg);

      camera = new THREE.PerspectiveCamera(55, glwrap.clientWidth/glwrap.clientHeight, 0.01, 2000);
      camera.position.set(2.4, 1.6, 2.4);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true; controls.dampingFactor = 0.06;

      const hemi = new THREE.HemisphereLight(0xffffff, 0x181a20, 0.7);
      scene.add(hemi);

      const dir  = new THREE.DirectionalLight(0xffffff, 0.35);
      dir.position.set(3,5,2);
      dir.castShadow = true;
      dir.shadow.mapSize.set(1024,1024);
      scene.add(dir);

      const fill = new THREE.SpotLight(0xffffff, 0.55, 0, Math.PI/3, 0.5, 1.0);
      fill.castShadow = false;
      fill.position.set(2.2, 2.0, 2.2);
      fill.target.position.set(0, 0.4, 0);
      scene.add(fill, fill.target);

      lightTarget = new THREE.Object3D();
      lightTarget.position.set(0, 0.0, 0);
      scene.add(lightTarget);

      const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(60,60),
        new THREE.MeshStandardMaterial({ color: 0x0b1220, roughness: 1.0 })
      );
      ground.rotation.x = -Math.PI/2;
      ground.position.y = -0.001;
      ground.receiveShadow = true;
      scene.add(ground);

      const grid = new THREE.GridHelper(60, 60, 0x29435c, 0x142231);
      grid.position.y = 0;
      scene.add(grid);

      window.addEventListener('resize', ()=>{
        const w = glwrap.clientWidth, h = glwrap.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, true);
      });
    }

    function setWireframe(root, on){
      if(!root) return;
      root.traverse(n=>{
        if(n.isMesh){
          (Array.isArray(n.material)?n.material:[n.material]).forEach(m=>{ if(m) m.wireframe = on; });
        }
      });
    }

    function frameByFootprint(widthM, depthM, heightM, offset = 1.6){
      const w = glwrap.clientWidth, h = glwrap.clientHeight;
      renderer.setSize(w, h, true);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();

      const radius = Math.max(widthM, depthM) * 0.5;
      const center = new THREE.Vector3(0, heightM, 0);
      const dist   = (offset * radius) / Math.tan((camera.fov * Math.PI) / 360);
      const camPos = center.clone().add(new THREE.Vector3(1, 0.8, 1).normalize().multiplyScalar(dist));

      camera.position.copy(camPos);
      controls.target.copy(center);
      controls.update();
    }

    // ===== Units mapping from true layout (fixes 14×36 squeeze/links) =====
    const FT_TO_M = 0.3048;
    function layoutBounds(fixt){
      let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
      fixt.forEach(f => f.targets.forEach(([x,y])=>{
        if(x<minx)minx=x; if(x>maxx)maxx=x; if(y<miny)miny=y; if(y>maxy)maxy=y;
      }));
      const w = Math.max(1e-6, maxx-minx);
      const h = Math.max(1e-6, maxy-miny);
      return { minx, maxx, miny, maxy, w, h, cx:(minx+maxx)/2, cy:(miny+maxy)/2 };
    }
    const LAY = layoutBounds(fixtures);

    // pick mapping that keeps scale most isotropic (auto axis swap if needed)
    const sX_len = (ROOM.Lft * FT_TO_M) / LAY.w; // SVG X -> room length
    const sZ_wid = (ROOM.Wft * FT_TO_M) / LAY.h;
    const sX_wid = (ROOM.Wft * FT_TO_M) / LAY.w; // SVG X -> room width
    const sZ_len = (ROOM.Lft * FT_TO_M) / LAY.h;
    const useSwap = Math.abs(sX_wid - sZ_len) < Math.abs(sX_len - sZ_wid);

    const metersPerUnitX = useSwap ? sX_wid : sX_len;
    const metersPerUnitZ = useSwap ? sZ_len : sZ_wid;
    const ux = u => (u - LAY.cx) * metersPerUnitX;  // SVG x -> world X
    const uz = v => (v - LAY.cy) * metersPerUnitZ;  // SVG y -> world Z
    const widthM = LAY.w * metersPerUnitX;          // world footprint
    const depthM = LAY.h * metersPerUnitZ;

    // ===== Fixture build (panels + per-fixture links) =====
    function makeFixtureFromLayout(fixt){
      const g = new THREE.Group();
      const panelG = new THREE.BoxGeometry(0.18, 0.03, 0.18);
      const panelM = new THREE.MeshStandardMaterial({ color: 0xcfd8dc, roughness: 0.6, metalness: 0.1 });
      const barM   = new THREE.MeshStandardMaterial({ color: 0x90a4ae, roughness: 0.7 });

      fixt.forEach(f=>{
        f.targets.forEach(([x,y])=>{
          const m = new THREE.Mesh(panelG, panelM);
          m.position.set(ux(x), 0.0, uz(y));
          m.castShadow = m.receiveShadow = true;
          g.add(m);
        });
        if (f.mtype === 'centerpiece' && f.targets.length >= 2){
          const [cx, cy] = f.targets[0];
          const xC = ux(cx), zC = uz(cy);
          for (let i=1;i<f.targets.length;i++){
            const [rx,ry]=f.targets[i]; const xR=ux(rx), zR=uz(ry);
            const dx=xR-xC, dz=zR-zC, len=Math.hypot(dx,dz);
            const bar = new THREE.Mesh(new THREE.BoxGeometry(0.02,0.02,Math.max(0.02,len)), barM);
            bar.position.set((xC+xR)/2, 0.005, (zC+zR)/2);
            bar.rotation.y = Math.atan2(dx, dz);
            bar.castShadow = bar.receiveShadow = true; g.add(bar);
          }
        } else {
          for (let i=0;i<f.targets.length-1;i++){
            const [ax,ay]=f.targets[i], [bx,by]=f.targets[i+1];
            const x1=ux(ax), z1=uz(ay), x2=ux(bx), z2=uz(by);
            const dx=x2-x1, dz=z2-z1, len=Math.hypot(dx,dz);
            const bar = new THREE.Mesh(new THREE.BoxGeometry(0.02,0.02,Math.max(0.02,len)), barM);
            bar.position.set((x1+x2)/2, 0.005, (z1+z2)/2);
            bar.rotation.y = Math.atan2(dx, dz);
            bar.castShadow = bar.receiveShadow = true; g.add(bar);
          }
        }
      });
      return g;
    }

    // ===== Plants (fewer but bigger) =====
    async function loadPlantMesh(url){
      const root = await new OBJLoader().loadAsync(url);
      let mesh=null; root.traverse(n=>{ if(n.isMesh && !mesh) mesh=n; });
      if(!mesh){
        mesh = new THREE.Mesh(
          new THREE.ConeGeometry(0.07, 0.15, 12),
          new THREE.MeshStandardMaterial({ color: 0x6fbf73, roughness: 0.85 })
        );
      }
      const g = mesh.geometry.clone(); g.computeBoundingBox();
      const bb = g.boundingBox;
      const center = new THREE.Vector3((bb.min.x+bb.max.x)/2, bb.min.y, (bb.min.z+bb.max.z)/2);
      g.translate(-center.x, -center.y, -center.z);
      const mat = ('metalness' in mesh.material) ? mesh.material : new THREE.MeshStandardMaterial({ color: 0x6fbf73, roughness: 0.85 });
      const normalized = new THREE.Mesh(g, mat); normalized.castShadow = normalized.receiveShadow = true;
      const size = new THREE.Vector3().subVectors(bb.max, bb.min);
      return { mesh: normalized, size };
    }

    function buildPlantGrid(
      sampleMesh,
      sampleSize,
      widthM, depthM, spacingM,
      { density = 0.45, sizeBoost = 1.20, maxInstances = 2200 } = {}
    ){
      const spacingEff = spacingM / Math.max(0.05, density);
      const cols  = Math.max(1, Math.round(widthM  / spacingEff));
      const rows  = Math.max(1, Math.round(depthM / spacingEff));
      const total = rows * cols;
      const stride = total > maxInstances ? Math.ceil(Math.sqrt(total / maxInstances)) : 1;

      const effCols = Math.ceil(cols / stride);
      const effRows = Math.ceil(rows / stride);

      const w = (effCols - 1) * spacingEff * stride;
      const d = (effRows - 1) * spacingEff * stride;
      const startX = -w * 0.5, startZ = -d * 0.5;

      const targetXZ  = (spacingEff * stride) * (0.60 * sizeBoost);
      const currentXZ = Math.max(1e-6, Math.max(sampleSize.x, sampleSize.z));
      const scale     = targetXZ / currentXZ;

      const geom = sampleMesh.geometry.clone();
      geom.scale(scale, scale, scale);
      const mat  = sampleMesh.material.clone();

      const count = effRows * effCols;
      const inst  = new THREE.InstancedMesh(geom, mat, count);
      inst.castShadow = inst.receiveShadow = true;

      const tmp = new THREE.Object3D(); let i=0;
      for(let r=0;r<effRows;r++){
        for(let c=0;c<effCols;c++){
          const jx = (Math.random()-0.5) * spacingEff * 0.08;
          const jz = (Math.random()-0.5) * spacingEff * 0.08;
          tmp.position.set(startX + c*spacingEff*stride + jx, 0, startZ + r*spacingEff*stride + jz);
          tmp.rotation.y = (Math.random()*0.25) - 0.125;
          const s = 1 + (Math.random()-0.5)*0.12;
          tmp.scale.set(s, s, s);
          tmp.updateMatrix();
          inst.setMatrixAt(i++, tmp.matrix);
        }
      }
      inst.instanceMatrix.needsUpdate = true;

      const container = new THREE.Group();
      container.add(inst);
      container.position.set(0, -0.8, 0);

      return { container, instanced: inst, rows: effRows, cols: effCols, width: Math.max(w, spacingEff), depth: Math.max(d, spacingEff), stride, spacingEff };
    }

    // ===== Light blanket (stacked sheets baked from module origins) =====
    function makeLightSheetTexture(widthM, depthM, modulePositions, {res=1024, spotRadius=0.08, gain=2.0, gamma=1.25}={}){
      const cnv = document.createElement('canvas'); cnv.width = cnv.height = res;
      const ctx = cnv.getContext('2d');
      const sx = res / widthM, sz = res / depthM;
      const pr = Math.max(2, Math.round(spotRadius * 0.5 * (sx + sz)));
      const grad = ctx.createRadialGradient(0,0,0, 0,0, pr);
      grad.addColorStop(0.0, `rgba(255,242,204,${0.22 * gain})`);
      grad.addColorStop(0.5, `rgba(255,242,204,${0.10 * gain})`);
      grad.addColorStop(1.0, `rgba(255,242,204,0.0)`);
      ctx.globalCompositeOperation = 'lighter';
      for (const p of modulePositions){
        const u = (p.x / widthM) + 0.5, v = (p.z / depthM) + 0.5;
        const x = u * res, y = v * res;
        ctx.save(); ctx.translate(x, y); ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(0, 0, pr, 0, Math.PI*2); ctx.fill(); ctx.restore();
      }
      if (gamma !== 1.0){
        const img = ctx.getImageData(0,0,res,res), d = img.data;
        for (let i=0;i<d.length;i+=4){
          d[i  ] = 255 * Math.pow(d[i  ]/255, 1/gamma);
          d[i+1] = 255 * Math.pow(d[i+1]/255, 1/gamma);
          d[i+2] = 255 * Math.pow(d[i+2]/255, 1/gamma);
        }
        ctx.putImageData(img, 0, 0);
      }
      const tex = new THREE.CanvasTexture(cnv);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.needsUpdate = true;
      return tex;
    }

    function makeLightSheet(widthM, depthM, texture){
      const geo = new THREE.PlaneGeometry(widthM, depthM);
      const mat = new THREE.MeshBasicMaterial({
        map: texture, transparent: true, opacity: 0.0,
        blending: THREE.AdditiveBlending, depthWrite: false
      });
      const sheet = new THREE.Mesh(geo, mat);
      sheet.rotation.x = -Math.PI/2;
      sheet.position.y = 0.015;
      sheet.renderOrder = 3;
      return sheet;
    }

    function makeLightBlanket(widthM, depthM, modulePositions, {
      layers = 6, height = 0.24, spotRadius = 0.10, gain = 2.2, gamma = 1.2
    } = {}){
      const g = new THREE.Group();
      for (let i = 0; i < layers; i++){
        const t = i / Math.max(layers - 1, 1);
        const layerOpacity = 0.65 * (1.0 - Math.pow(t, 1.5));
        const tex = makeLightSheetTexture(widthM, depthM, modulePositions, {
          res: 1024,
          spotRadius: spotRadius * (1.0 + t * 0.25),
          gain: gain * (0.9 + 0.1 * (1 - t)),
          gamma
        });
        const sheet = makeLightSheet(widthM, depthM, tex);
        sheet.position.y = 0.015 + t * height;
        sheet.material.opacity = 0.0;
        sheet.userData._maxOpacity = layerOpacity;
        g.add(sheet);
      }
      return g;
    }

    function setBlanketOpacity(blanketGroup, k){
      if (!blanketGroup) return;
      blanketGroup.children.forEach(m => {
        const cap = m.userData._maxOpacity ?? 0.6;
        m.material.opacity = cap * k;
      });
    }

    // ===== Main =====
    async function start3D(){
      glwrap.classList.add('ready');
      // Tell parent that 3D has started
      try{ window.parent?.postMessage({type:'lp-assembly-phase', phase:'3d-start'}, '*'); }catch(_){}

      glui.style.display = 'flex';
      stageSvg.style.opacity = 0;

      if(!scene) setupRenderer();

      // clear previous
      if (rafId){ cancelAnimationFrame(rafId); rafId = null; }
      if (plantContainer){ scene.remove(plantContainer); plantContainer = null; }
      if (canopyGroup){ scene.remove(canopyGroup); canopyGroup = null; }
      if (fixtureGroup){ scene.remove(fixtureGroup); fixtureGroup = null; }
      if (fixtureLight){ fixtureLight.parent?.remove(fixtureLight); fixtureLight = null; }

      // 1) Fixture
      fixtureGroup = makeFixtureFromLayout(fixtures);
      scene.add(fixtureGroup);

      // 2) Plants (fewer/bigger)
      const { mesh: plantMesh, size: plantSize } = await loadPlantMesh(params.plantURL);
      const grid = buildPlantGrid(plantMesh, plantSize, widthM, depthM, params.plantSpacing, {
        density: 0.45, sizeBoost: 1.20, maxInstances: 2200
      });
      plantContainer = grid.container;
      plantInst = grid.instanced;

      // 3) Canopy group
      canopyGroup = new THREE.Group();
      canopyGroup.add(plantContainer);
      scene.add(canopyGroup);

      // 4) Pose (define start/end Y before using endY)
      const rotStart = Math.PI * 0.75, rotEnd = Math.PI * 0.15;
      const startY   = params.targetHeight + 0.6;
      const endY     = params.targetHeight;

      fixtureGroup.position.set(0, startY, 0);
      fixtureGroup.rotation.set(0.0, rotStart, 0.0);
      canopyGroup.rotation.y = rotStart;

      // 5) Soft spotlight
      fixtureLight = new THREE.SpotLight(0xfff2cc, 0.0, 18, Math.PI/3.6, 0.55, 1.4);
      fixtureLight.castShadow = true;
      fixtureLight.shadow.mapSize.set(512, 512);
      fixtureLight.shadow.bias   = -0.0006;
      fixtureLight.shadow.radius = 2;
      fixtureLight.position.set(0, -0.05, 0);
      fixtureLight.target = lightTarget;
      fixtureGroup.add(fixtureLight);

      // 6) Module positions (fixture-local)
      const modulePositions = [];
      fixtures.forEach(f => f.targets.forEach(([uxVal, uyVal]) => {
        modulePositions.push(new THREE.Vector3(ux(uxVal), 0, uz(uyVal)));
      }));

      // 7) Light blanket (+ crisp surface sheet)
      const blanket = makeLightBlanket(
        widthM, depthM, modulePositions,
        { layers: 6, height: Math.min(0.28, startY * 0.45), spotRadius: Math.max(params.plantSpacing * 0.95, 0.08), gain: 2.2, gamma: 1.2 }
      );
      canopyGroup.add(blanket);

      const sheetTex = makeLightSheetTexture(
        widthM, depthM, modulePositions,
        { res: 1024, spotRadius: Math.max(params.plantSpacing*0.75, 0.06), gain: 2.0, gamma: 1.25 }
      );
      const lightSheet = makeLightSheet(widthM, depthM, sheetTex);
      canopyGroup.add(lightSheet);

      // 8) Frame AFTER endY known
      frameByFootprint(widthM, depthM, endY, 1.6);
      renderer.render(scene, camera);

      // 9) Animate
      let animDone = false;
      const renderOnce = () => renderer.render(scene, camera);

      function tick(now){
        if(!tick.t0) tick.t0 = now;
        const t = now - tick.t0;

        // plants up
        { const k = Math.min(1, Math.max(0, (t-0)/1600)); const e = easeOutCubic(k);
          plantContainer.position.y = THREE.MathUtils.lerp(-0.8, 0.0, e); }

        // fixture drop
        { const k = Math.min(1, Math.max(0, (t-600)/1400)); const e = easeInOutQuad(k);
          fixtureGroup.position.y = THREE.MathUtils.lerp(startY, endY, e); }

        // spin + align
        { const k = Math.min(1, Math.max(0, (t - 600)/1600)); const e = easeInOutQuad(k);
          const yRot = THREE.MathUtils.lerp(rotStart, rotEnd, e);
          fixtureGroup.rotation.y = yRot; canopyGroup.rotation.y  = yRot; }

        // light fade
        {
          const k = Math.min(1, Math.max(0, (t - 900)/900)); const e = easeInOutQuad(k);
          fixtureLight.intensity = THREE.MathUtils.lerp(0.0, 3.0, e);
          setBlanketOpacity(blanket, e);
          if (plantInst?.material?.emissiveIntensity !== undefined){
            plantInst.material.emissiveIntensity = THREE.MathUtils.lerp(0.0, 0.45, e);
          }
        }

        controls.update(); renderOnce();

        if (t > 2700 && !animDone){
        animDone = true;
        controls.addEventListener('change', renderOnce);
        // Tell parent that the full 3D animation is complete
        try{ window.parent?.postMessage({type:'lp-assembly-phase', phase:'3d-complete'}, '*'); }catch(_){}
        return;
        }

        rafId = requestAnimationFrame(tick);
      }
      tick.t0 = undefined;           // reset timeline for replays
      rafId = requestAnimationFrame(tick);
    }

    // Expose start3D (optional)
    window.start3D = start3D;

    // UI
    replayBtn?.addEventListener('click', ()=>{ launched3D=false; start3D(); });
    wireBtn?.addEventListener('click', ()=>{
      wireframe = !wireframe; if (scene) setWireframe(scene, wireframe);
    });
    resetBtn?.addEventListener('click', ()=>{
      if (!renderer || !scene || !camera) return;
      frameByFootprint(widthM, depthM, params.targetHeight, 1.6);
      renderer.render(scene, camera);
    });

    // Error surfacing
    window.addEventListener('error', e => console.error('3D error:', e.error || e.message));
  </script>

</body>
</html>

'''




@app.get('/assembly')
def assembly():
    from math import sqrt

    def _coerce(v, default):
        try: return float(v)
        except Exception: return float(default)

    length_ft = _coerce(request.args.get('length', 12), 12)
    width_ft  = _coerce(request.args.get('width', 12), 12)
    cv_enabled = False

    # NEW: embed flag (for splash iframe)
    embed = str(request.args.get('embed', '0')).lower() in ('1','true','yes','on')


    # Build layout (unchanged)
    svg, total_cost, layout_data = compute_layout(
        length_ft, width_ft, cv_enabled,
        tutorial=False, wrap_flat_group=False, return_data=True
    )

    BASE = 22 / sqrt(2)
    def rot(x, y): return (BASE * (x + y), (BASE * (x - y)))

    # Fixtures list (unchanged logic)
    fixtures, order = [], []
    if layout_data.get("center") or layout_data.get("rings"):
        if layout_data.get("center"): order.append(("centerpiece", layout_data["center"]))
        for idx, rg in enumerate(layout_data.get("rings", []), start=1): order.append((f"ring{idx}", rg))
    else:
        order.append(("flat", layout_data.get("module_groups", [])))

    palette = ["#f9d423","#e65c00","#1dd1a1","#54a0ff","#ff6b6b",
               "#5f27cd","#48dbfb","#00d2d3","#ff9f43","#10ac84"]
    ci = 0
    for gname, items in order:
        for mtype, orient, cobs in items:
            fixtures.append({"group": gname, "mtype": mtype, "targets": [rot(a,b) for (a,b) in cobs], "color": palette[ci % len(palette)]})
            ci += 1

    # Assembly bbox for viewBox
    all_points = [rot(x,y) for (x,y) in layout_data.get("all_positions", [])]
    if not all_points and layout_data.get("module_groups"):
        for _,_,cobs in layout_data["module_groups"]:
            all_points.extend([rot(a,b) for (a,b) in cobs])

    if all_points:
        xs = [p[0] for p in all_points]; ys = [p[1] for p in all_points]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    else:
        minx, maxx, miny, maxy = -400, 400, -300, 300

    PAD = 120
    asm_left, asm_right = minx - PAD, maxx + PAD
    asm_top, asm_bottom = miny - PAD, maxy + PAD
    asm_cx, asm_cy = (minx + maxx) / 2, (miny + maxy) / 2
    asm_w, asm_h = asm_right - asm_left, asm_bottom - asm_top
    viewbox = f"{asm_cx - asm_w/2:.0f} {asm_cy - asm_h/2:.0f} {asm_w:.0f} {asm_h:.0f}"

    bbox = dict(minx=minx, maxx=maxx, miny=miny, maxy=maxy)

    # 3D params (override via query if needed)
    plant_url     = request.args.get("plant_url", "/static/plant.obj")
    plant_spacing = _coerce(request.args.get("plant_spacing", 0.30), 0.30)   # meters
    target_height = _coerce(request.args.get("target_height", 1.20), 1.20)   # meters
    bg            = request.args.get("bg", "#0b0f14")

    return render_template_string(
        ASSEMBLY_TPL,
        fixtures=fixtures,
        length_ft=length_ft, width_ft=width_ft,
        viewbox=viewbox,
        marker_size=10, stroke_w=2,
        plant_url=plant_url, plant_spacing=plant_spacing,
        target_height=target_height, bg=bg,
        bbox=bbox,
        embed=embed,              # <-- pass to template        
    )

# --- Explicit static routes (GET/HEAD) ---
@app.get("/blog/index.html")
def blog_index_html():
    return send_from_directory(Path(app.root_path) / "blog", "index.html")

@app.get("/blog/")
@app.get("/blog")
def blog_dir_index():
    return send_from_directory(Path(app.root_path) / "blog", "index.html")

@app.get("/lights.html")
def lights_html():
    return send_from_directory(app.root_path, "lights.html")

@app.get("/lights_intro.html")
def lights_intro_html():
    return send_from_directory(app.root_path, "lights_intro.html")

@app.get("/simulations.html")
def simulations_html():
    return send_from_directory(app.root_path, "simulations.html")

@app.get("/splash.html")
def splash_html():
    return send_from_directory(app.root_path, "splash.html")


# --- Safe catch-all for public assets (place BEFORE any JSON catch-alls) ---
@app.route("/<path:relpath>", methods=["GET", "HEAD"])
def serve_public(relpath: str):
    # Do not intercept API/solver/file endpoints
    RESERVED = ("api/", "solver/", "precompute/", "files/")
    if relpath.startswith(RESERVED):
        abort(404)

    root = Path(app.root_path).resolve()
    p = (root / relpath).resolve()

    # Block path traversal
    if not str(p).startswith(str(root)):
        abort(404)

    # If exact file exists (html/css/js/img/etc), serve it
    if p.is_file():
        # Let Flask handle correct mimetype & caching
        return send_from_directory(root, relpath)

    # If a directory, try index.html
    if p.is_dir() and (p / "index.html").is_file():
        return send_from_directory(p, "index.html")

    # Try implicit ".html" (e.g. /about -> /about.html)
    if (root / f"{relpath}.html").is_file():
        return send_from_directory(root, f"{relpath}.html")

    abort(404)

# serve /pics/<file> from the project’s pics/ folder
@app.get("/pics/<path:asset>")
def pics_static(asset: str):
    pics_dir = os.path.join(app.root_path, "pics")
    file_path = os.path.join(pics_dir, asset)
    if os.path.isfile(file_path):
        return send_from_directory(pics_dir, asset)
    abort(404)

# --- Explicit static routes (GET/HEAD) ---
@app.get("/index.html")
def index_html():
    return send_from_directory(app.root_path, "index.html")

@app.get("/styles.css")
def styles_css():
    return send_from_directory(app.root_path, "styles.css")

@app.get("/scripts.js")
def scripts_js():
    return send_from_directory(app.root_path, "scripts.js")

@app.get("/favicon.ico")
def favicon():
    return send_from_directory(app.root_path, "favicon.ico")

# ---------- Root proxies to solver APIs (accept GET/POST/OPTIONS) ----------
# These let the SPA call /api/* at the root, while the real app is mounted at /solver

@app.route("/api/_precompute_sync", methods=["GET", "POST", "OPTIONS"])
def _root_api_precompute_sync():
    # Delegate to the solver's implementation
    from radsim.solver import api__precompute_sync as _solver_precompute_sync
    return _solver_precompute_sync()

@app.route("/api/run_radiance", methods=["GET", "POST", "OPTIONS"])
def _root_api_run_radiance():
    from radsim.solver import api_run_radiance as _solver_run_radiance
    return _solver_run_radiance()

# Competitor endpoints too (your SPA likely calls these at the root)
@app.route("/competitor/precompute", methods=["GET", "POST", "OPTIONS"])
def _root_competitor_precompute():
    from radsim.solver import competitor_precompute as _solver_comp_precompute
    return _solver_comp_precompute()

@app.route("/competitor/verify", methods=["GET", "POST", "OPTIONS"])
def _root_competitor_verify():
    from radsim.solver import competitor_verify as _solver_comp_verify
    return _solver_comp_verify()

@app.post("/create-checkout-session")
def create_checkout_session():
    """Create a Stripe Checkout Session (TEST MODE)."""
    if not stripe.api_key:
        return jsonify(ok=False, error="Stripe not configured (set STRIPE_SECRET_KEY)."), 500

    data = request.get_json(silent=True) or {}
    length_ft = data.get("length") or ""
    width_ft  = data.get("width") or ""
    dims_desc = f"{length_ft}ft x {width_ft}ft" if length_ft and width_ft else "Custom grow space"
    try:
        total_cost = float(str(data.get("total_cost", 0)).replace(",", ""))
    except Exception:
        total_cost = 0.0

    # If no total provided, recompute from layout
    if total_cost <= 0 and length_ft and width_ft:
        try:
            _, computed_total = compute_layout(float(length_ft), float(width_ft), False, tutorial=False, wrap_flat_group=False)
            total_cost = computed_total or 0.0
        except Exception:
            total_cost = 0.0

    # Fallback to $100 if still missing
    amount_cents = int(max(total_cost, 0.0) * 100) if total_cost > 0 else 10000

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": amount_cents,
                    "product_data": {
                        "name": "Lighting Layout System",
                        "description": f"Horticultural lighting layout for {dims_desc}",
                    },
                },
                "quantity": 1,
            }],
            success_url=url_for("checkout_success", _external=True),
            cancel_url=url_for("checkout_cancel", _external=True),
        )
        return jsonify(ok=True, sessionId=session.id)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

@app.get("/success")
def checkout_success():
    return render_template_string("""
<!DOCTYPE html>
<html><head><title>Payment Success</title>
<style>
  body{background:#000;color:#fff;font-family:'Avenir Next',sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;}
  .card{background:#0d0d0d;border:1px solid rgba(255,215,0,.45);border-radius:14px;padding:24px;max-width:480px;text-align:center;box-shadow:0 18px 36px rgba(0,0,0,.6);}
  h1{color:#FFD700;margin:0 0 12px;}
  a{color:#FFD700;text-decoration:none;}
</style>
</head><body>
  <div class="card">
    <h1>Payment Success</h1>
    <p>Thanks for completing checkout. We’ll be in touch shortly.</p>
    <p><a href="/index.html">Return to homepage</a></p>
  </div>
</body></html>
    """)

@app.get("/cancel")
def checkout_cancel():
    return render_template_string("""
<!DOCTYPE html>
<html><head><title>Payment Canceled</title>
<style>
  body{background:#000;color:#fff;font-family:'Avenir Next',sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;}
  .card{background:#0d0d0d;border:1px solid rgba(255,215,0,.45);border-radius:14px;padding:24px;max-width:480px;text-align:center;box-shadow:0 18px 36px rgba(0,0,0,.6);}
  h1{color:#FFD700;margin:0 0 12px;}
  a{color:#FFD700;text-decoration:none;}
</style>
</head><body>
  <div class="card">
    <h1>Payment Canceled</h1>
    <p>Your payment was canceled. You can return to your cart to try again.</p>
    <p><a href="/checkout">Back to cart</a></p>
  </div>
</body></html>
    """)

@app.route("/")
def home():
    return send_from_directory(".", "splash.html")

@app.route('/app', methods=['GET', 'POST'])
def generator():
    from flask import request, render_template_string

    def _fmt_ft(v):
        v = float(v)
        return str(int(v)) if v.is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")

    if request.method == 'POST':
        try:
            length_ft = request.form.get("length", type=float)
            width_ft  = request.form.get("width",  type=float)
            if length_ft is None or width_ft is None:
                raise ValueError("Missing dimensions")

            cv_enabled    = 'cv_mode' in request.form
            tutorial      = request.form.get('tutorial', 'false') == 'true'
            radiance_mode = request.form.get('radiance_mode', 'false') == 'true'

            length_label = _fmt_ft(length_ft)
            width_label  = _fmt_ft(width_ft)

            if tutorial:
                svg_square, total_cost = compute_layout(12, 12, cv_enabled, tutorial=True,  wrap_flat_group=False)
                svg_20x40, cost_40     = compute_layout(20, 40, cv_enabled, tutorial=False, wrap_flat_group=True)
                svg_20x60, cost_60     = compute_layout(20, 60, cv_enabled, tutorial=False, wrap_flat_group=True)
                svg_20x75, cost_75     = compute_layout(20, 75, cv_enabled, tutorial=False, wrap_flat_group=True)
            else:
                svg_square, total_cost = compute_layout(length_ft, width_ft, cv_enabled, tutorial=False, wrap_flat_group=False)
                svg_20x40 = svg_20x60 = svg_20x75 = ''
                cost_40 = cost_60 = cost_75 = 0

            return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Lighting Layout</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #000 0%, #1a1a1a 100%);
      color:#fff;margin:0;padding:20px;display:flex;flex-direction:column;align-items:center;min-height:100vh
    }
    .header{display:flex;justify-content:space-between;align-items:center;width:90%;max-width:1200px;margin-bottom:10px;gap:12px}
    h1{color:#ffd700;margin:0;text-shadow:2px 2px 4px rgba(0,0,0,.3)}
    .header p{margin:0;white-space:nowrap}
    .quote{background:rgba(255,255,255,.05);padding:10px 16px;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,.2);white-space:nowrap}
    .svg-container{
      position:relative;width:90%;max-width:1200px;height:80vh;background:rgba(255,255,255,.05);
      border:1px solid rgba(255,255,255,.1);border-radius:12px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.2);
      display:flex;align-items:center;justify-content:center;
    }
    #layout-svg{max-width:100%;max-height:100%}
    .controls{
      position:absolute;top:10px;right:10px;background:rgba(0,0,0,.7);padding:10px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,.2)
    }
    button{
      background:#ffd700;color:#000;border:none;padding:8px 12px;margin:5px;border-radius:8px;cursor:pointer;font-weight:700;
      transition:background-color .18s ease, box-shadow .18s ease, transform .08s ease;
    }
    button:hover{ background:#fff; box-shadow:0 10px 22px rgba(255,255,255,.18); transform:translateY(-1px) }
    button:active{ transform:translateY(0) }
    a{color:#ffd700;text-decoration:none;margin-top:20px}
    a:hover{text-decoration:underline}
    @keyframes glow{0%,100%{filter:drop-shadow(0 0 5px #FFD700);}50%{filter:drop-shadow(0 0 15px #FFD700);}}
    .glowing{animation:glow 2s infinite}
    #end-modal{
      position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;z-index:1000
    }
    .modal-content{background:#000;padding:20px;border:2px solid #ffd700;border-radius:10px;text-align:center;color:#fff}
    .modal-content button{background:#ffd700;color:#000;border:none;padding:10px 20px;margin:10px;border-radius:8px;cursor:pointer}
    .modal-content button:hover{background:#fff}

    /* --- Slim bottom info bar (matches solver) --- */
    .tour-bar{
      position:fixed;left:50%;bottom:18px;transform:translateX(-50%);
      width:min(1100px,94vw);
      background:linear-gradient(180deg, rgba(30,30,30,.96), rgba(25,25,25,.96));
      border:1px solid rgba(255,255,255,.16);border-radius:12px;padding:12px 14px;
      display:none;z-index:1000;
    }
    .tour-row{display:flex;justify-content:space-between;align-items:center;gap:10px}
    .tour-text{color:#fff;opacity:.96;font-size:15px}
    .tour-cta{display:flex;gap:8px}
    .tour-btn{background:#ffd700;color:#000;border:none;padding:8px 14px;border-radius:10px;font-weight:700;cursor:pointer}
    .tour-btn.secondary{background:transparent;color:#fff;border:1px solid rgba(255,255,255,.25)}
    .tour-btn:disabled{opacity:.6;cursor:not-allowed}
    @media (max-width:768px){ .svg-container{height:70vh} }
  </style>

  <script>
    // Pan/zoom
    document.addEventListener('DOMContentLoaded', () => {
      const svg = document.getElementById('layout-svg');
      if (!svg) return;
      let viewBox = (svg.getAttribute('viewBox') || "0 0 100 100").split(' ').map(Number);
      let [minX, minY, width, height] = viewBox;
      let zoomLevel = 1, isPanning = false, startX, startY;

      function updateViewBox(){ svg.setAttribute('viewBox', `${minX} ${minY} ${width/zoomLevel} ${height/zoomLevel}`); }
      function zoom(f){ zoomLevel *= f; updateViewBox(); }

      document.getElementById('zoom-in') ?.addEventListener('click', () => zoom(1.2));
      document.getElementById('zoom-out')?.addEventListener('click', () => zoom(0.8333));
      document.getElementById('reset')   ?.addEventListener('click', () => { zoomLevel=1; minX=viewBox[0]; minY=viewBox[1]; updateViewBox(); });

      svg.addEventListener('mousedown', e => { isPanning=true; startX=e.clientX; startY=e.clientY; });
      svg.addEventListener('mousemove', e => {
        if (!isPanning) return;
        const dx=(e.clientX-startX)*(width/zoomLevel/svg.clientWidth);
        const dy=(e.clientY-startY)*(height/zoomLevel/svg.clientHeight);
        minX-=dx; minY-=dy; startX=e.clientX; startY=e.clientY; updateViewBox();
      });
      ['mouseup','mouseleave'].forEach(ev => svg.addEventListener(ev, () => { isPanning=false; }));

      // touch
      let tStartX, tStartY, tStartMinX, tStartMinY, initialDistance, initialZoom;
      const dist = ts => Math.hypot(ts[0].clientX-ts[1].clientX, ts[0].clientY-ts[1].clientY);
      svg.addEventListener('touchstart', e => {
        if (e.touches.length === 1){
          isPanning=true; tStartX=e.touches[0].clientX; tStartY=e.touches[0].clientY; tStartMinX=minX; tStartMinY=minY;
        } else if (e.touches.length === 2){ initialDistance = dist(e.touches); initialZoom = zoomLevel; }
      }, {passive:false});
      svg.addEventListener('touchmove', e => {
        if (e.touches.length === 1 && isPanning){
          const dx=(e.touches[0].clientX-tStartX)*(width/zoomLevel/svg.clientWidth);
          const dy=(e.touches[0].clientY-tStartY)*(height/zoomLevel/svg.clientHeight);
          minX=tStartMinX-dx; minY=tStartMinY-dy; updateViewBox();
        } else if (e.touches.length === 2){
          zoomLevel = initialZoom * (dist(e.touches) / initialDistance); updateViewBox();
        }
        e.preventDefault();
      }, {passive:false});
      svg.addEventListener('touchend', () => { isPanning=false; });
    });
  </script>

  {% if tutorial %}
  <script>
    // Results-page tutorial (slim bottom bar UI)
    window.svgs  = {'square': {{ svg_square|tojson|safe }}, 'rect40': {{ svg_20x40|tojson|safe }}, 'rect60': {{ svg_20x60|tojson|safe }}, 'rect75': {{ svg_20x75|tojson|safe }}};
    window.costs = {'square': {{ total_cost }}, 'rect40': {{ cost_40 }}, 'rect60': {{ cost_60 }}, 'rect75': {{ cost_75 }}};
    window.dims  = {'square': '12x12', 'rect40': '20x40', 'rect60': '20x60', 'rect75': '20x75'};

    document.addEventListener('DOMContentLoaded', () => {
      const texts = [
        "Welcome to our modularized horticultural LED lighting system. It starts with the centerpiece that just flew onto the screen. The white squares are the SMD-LED modules and the colored lines are the fixtures they're installed to.",
        "Now adding the first ring: Two reverse L-Shaped modular fixtures. We separate the lighting system into rings to tune the intensity of each ring for maximum PPFD uniformity.",
        "Next, the second ring: Four Linear 3-SMD LED module fixtures, providing further control. Each additional ring provides more control of the uniformity.",
        "The third ring: A combination of Linear-3, L, and Reverse-L fixtures.",
        "The fourth ring: L, Linear-3s, Linear-4s, Reverse-L, and Linear-3 fixtures. A truly infinitely scalable, modularized lighting system.",
        "The lighting system works for rectangular grow space dimensions too!",
        "For a 20'x40' space, we create one big rectangular layout because adding another square wouldn't fit well. This strategy uses a straight line of lights in the center and rings surrounding the edges to cover the area efficiently.",
        "For a 20'x60' space, we can fit one square unit connected to a rectangular extension. This lets us use the efficient square design for maximum uniformity control and a rectangular section to fill the rest. A connector module sits between the square and rectangle to join them together effectively.",
        "For a 20'x75' space, we fit two square units connected together via those connector modules we mentioned in the last slide, plus a rectangular extension at the end. This strategy enables the coverage of any sized rectangular space while achieving maximum PPFD uniformity at any desired PPFD."
      ];

      let svgElem  = document.getElementById('layout-svg');
      let currentStep = 0;
      const maxSteps = 8;

      const bar   = document.getElementById('tour-bar');
      const msg   = document.getElementById('tour-msg');
      const next  = document.getElementById('tour-next');

      function showBar(text){ if (msg) msg.textContent = text; if (bar) bar.style.display = 'block'; }
      function hideBar(){ if (bar) bar.style.display = 'none'; }

      function swapLayout(key){
        if (!window.svgs[key]) return;
        svgElem.outerHTML = window.svgs[key];
        svgElem = document.getElementById('layout-svg');
        document.querySelector('.quote').innerText = `Total Cost: $${window.costs[key]}`;
        document.querySelector('.header p').innerText = `Grow space: ${window.dims[key]}'`;
      }

      function showStep(step){
        if (step > maxSteps){ hideBar(); return; }

        const layoutSteps = {5:'rect40', 6:'rect40', 7:'rect60', 8:'rect75'};
        const newKey = layoutSteps[step];
        if (newKey) swapLayout(newKey);

        const ring  = document.getElementById(`ring${step}`);
        const group = document.getElementById('layout-group');
        const anim  = ring || group;
        if (anim){
          anim.style.transition = 'transform 1s ease-out, opacity 1s ease-out';
          anim.style.transform = 'translateX(0)';
          anim.style.opacity = '1';
          setTimeout(() => anim.classList.add('glowing'), 1000);
        }

        showBar(texts[step]);
        if (next) next.textContent = (step === maxSteps) ? 'Finish' : 'Next';
      }

      next?.addEventListener('click', () => {
        const ring  = document.getElementById(`ring${currentStep}`);
        const group = document.getElementById('layout-group');
        const anim  = ring || group;
        if (anim) anim.classList.remove('glowing');

        if (next.textContent === 'Finish'){
          hideBar();
          document.getElementById('end-modal').style.display = 'flex';
          return;
        }
        currentStep = Math.min(maxSteps, currentStep + 1);
        showStep(currentStep);
      });

      document.getElementById('yes-end')?.addEventListener('click', () => { window.location.href = '/app'; });
      document.getElementById('no-end') ?.addEventListener('click', () => { document.getElementById('end-modal').style.display = 'none'; });

      setTimeout(() => showStep(0), 400);
    });
  </script>
  {% endif %}
</head>

<body>
  <div class="header">
    <h1>Lighting Layout</h1>
    <p>Grow space: {{ length_label }}' × {{ width_label }}' {% if radiance_mode %}(Radiance mode){% endif %}</p>
    <div class="quote">Total Cost: ${{ total_cost }}</div>
  </div>

  <div class="svg-container">
    {{ svg_square|safe }}
    <div class="controls">
      <button id="add-to-cart">Add to Cart</button>
      <button id="zoom-in">Zoom In</button>
      <button id="zoom-out">Zoom Out</button>
      <button id="reset">Reset</button>
      <button onclick="window.location.href='/app'">Back</button>
      <button id="view-assembly">View 3D Assembly</button>
    </div>
  </div>

  <!-- Slim bottom info bar (used only when tutorial=true) -->
  {% if tutorial %}
  <div id="tour-bar" class="tour-bar">
    <div class="tour-row">
      <div id="tour-msg" class="tour-text"></div>
      <div class="tour-cta">
        <button id="tour-next" class="tour-btn">Next</button>
      </div>
    </div>
  </div>
  {% endif %}

  <div id="end-modal">
    <div class="modal-content">
      <h2>Try Your Own</h2>
      <p>Would you like to try your own grow space dimensions?</p>
      <button id="yes-end">Yes</button>
      <button id="no-end">No</button>
    </div>
  </div>

  <a href="/app">Back to Generator</a>

  <script>
    // "View 3D Assembly" link (results page)
    document.addEventListener('DOMContentLoaded', () => {
      const L = {{ length_ft|tojson }};
      const W = {{ width_ft|tojson }};
      const totalCost = {{ total_cost|tojson }};
      let assemblyViewed = false;

      const assemblyBtn = document.getElementById('view-assembly');
      if (assemblyBtn){
        assemblyBtn.addEventListener('click', () => {
          assemblyViewed = true;
          window.location.href = `/assembly?length=${encodeURIComponent(L)}&width=${encodeURIComponent(W)}`;
        });
      }

      const addBtn = document.getElementById('add-to-cart');
      if (addBtn){
        addBtn.addEventListener('click', () => {
          const form = document.createElement('form');
          form.method = 'POST';
          form.action = '/checkout';

          const payload = {
            length: L,
            width: W,
            total_cost: totalCost,
            layout_svg: document.getElementById('layout-svg')?.outerHTML || '',
            assembly_viewed: assemblyViewed ? '1' : '0',
          };

          Object.entries(payload).forEach(([key, val]) => {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = key;
            input.value = String(val);
            form.appendChild(input);
          });

          document.body.appendChild(form);
          form.submit();
        });
      }
    });
  </script>
</body>
</html>
            ''',
            svg_square=svg_square, svg_20x40=svg_20x40, svg_20x60=svg_20x60, svg_20x75=svg_20x75,
            cost_40=cost_40, cost_60=cost_60, cost_75=cost_75, total_cost=total_cost,
            tutorial=tutorial, radiance_mode=radiance_mode,
            length_label=length_label, width_label=width_label,
            length_ft=length_ft, width_ft=width_ft)

        except ValueError:
            return "Invalid input. Please enter numeric values.", 400

    # ---------- GET (initial form) ----------
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Lighting Layout Generator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body{
      font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background:linear-gradient(135deg,#000 0%,#1a1a1a 100%);color:#fff;margin:0;padding:40px;
      display:flex;flex-direction:column;align-items:center;min-height:100vh;transition:transform .5s ease-out, opacity .5s ease-out;
    }
    body.slide-up{ transform:translateY(-100%); opacity:0 }
    h1{ color:#ffd700; text-shadow:2px 2px 4px rgba(0,0,0,.3); margin:60px 0 30px }
    form{
      background:rgba(255,255,255,.05); padding:30px; border-radius:12px; box-shadow:0 8px 32px rgba(0,0,0,.2);
      width:400px; backdrop-filter:blur(10px);
    }
    label{ display:block; margin-bottom:10px; font-weight:700 }
    input[type="number"]{
      width:100%; margin-bottom:20px; padding:12px; border:1px solid rgba(255,255,255,.1);
      border-radius:8px; background:rgba(255,255,255,.02); color:#fff; font-size:1em;
      transition:border-color .18s ease, box-shadow .18s ease;
    }
    input[type="number"]:focus{ outline:none; border-color:rgba(255,215,0,.6); box-shadow:0 0 0 3px rgba(255,215,0,.18) }
    /* Primary submit */
    #submit-btn, #howto-btn{
      background:#ffd700;color:#000;border:none;padding:12px;border-radius:8px;cursor:pointer;width:100%;
      font-size:1.1em;font-weight:700;margin-top:2px;transition:background-color .18s ease, box-shadow .18s ease, transform .08s ease;
    }
    #submit-btn:hover, #howto-btn:hover{ background:#fff; box-shadow:0 10px 22px rgba(255,255,255,.18); transform:translateY(-1px) }
    #submit-btn:active, #howto-btn:active{ transform:translateY(0) }

    #home-arrow{
      position:fixed; top:20px; left:50%; transform:translateX(-50%);
      background:#ffd700;color:#000;border:none;padding:8px 16px;border-radius:50%;
      cursor:pointer;font-size:1.5em; z-index:1000; box-shadow:0 4px 8px rgba(0,0,0,.2);
    }
    #home-arrow:hover{ background:#fff }

    /* Start-screen modal — hidden by default; same behavior as solver */
    #tutorial-modal{
      position:fixed; inset:0; background:rgba(0,0,0,.8); display:none; align-items:center; justify-content:center; z-index:1000;
    }
    .modal-content{
      background:#000; padding:20px; border:2px solid #ffd700; border-radius:10px; text-align:center; color:#fff; width:min(520px,92vw)
    }
    .modal-content button{
      background:#ffd700;color:#000;border:none;padding:10px 20px;margin:10px;border-radius:8px;cursor:pointer;font-weight:700
    }
    .modal-content button:hover{ background:#fff }
  </style>

  <script>
    document.addEventListener('DOMContentLoaded', () => {
      // Defaults
      const len = document.getElementById('length');
      const wid = document.getElementById('width');
      if (!len.value || isNaN(+len.value)) len.value = '14';
      if (!wid.value || isNaN(+wid.value)) wid.value = '14';

      // Home arrow
      document.getElementById('home-arrow')?.addEventListener('click', (e)=>{
        e.preventDefault(); document.body.classList.add('slide-up'); setTimeout(()=>{ location.href='/index.html'; }, 500);
      });

      const modal = document.getElementById('tutorial-modal');
      const showModal = () => { if (modal) modal.style.display = 'flex'; };
      const hideModal = () => { if (modal) modal.style.display = 'none'; };

      // CTA under submit: "How Does This System Work?"
      document.getElementById('howto-btn')?.addEventListener('click', (e)=>{
        e.preventDefault(); showModal();
      });

      // Auto-open if ?tour=1
      try {
        const force = new URL(location.href).searchParams.get('tour') === '1';
        if (force) showModal();
      } catch (_) {}

      // Modal buttons
      document.getElementById('skip')?.addEventListener('click', hideModal);
      document.getElementById('continue')?.addEventListener('click', ()=>{
        const form = document.getElementById('dims-form');
        if (!form) return;
        if (len) len.value = '14';
        if (wid) wid.value = '14';
        const hid = document.createElement('input');
        hid.type='hidden'; hid.name='tutorial'; hid.value='true';
        form.appendChild(hid);
        form.submit();
      });
    });
  </script>
</head>

<body>
  <button id="home-arrow">↑</button>
  <h1>Horticultural Lighting Layout Generator</h1>

  <form method="post" id="dims-form" novalidate>
    <div class="field" id="len-field">
      <label for="length">Length (ft)</label>
      <input type="number" id="length" name="length" inputmode="decimal">
    </div>

    <div class="field" id="wid-field">
      <label for="width">Width (ft)</label>
      <input type="number" id="width" name="width" inputmode="decimal">
    </div>

    <input type="submit" id="submit-btn" value="Generate Layout" />
    <!-- New: helper button under submit (same hover/feel as primary) -->
    <button id="howto-btn" type="button" aria-haspopup="dialog" aria-controls="tutorial-modal" style="margin-top:10px;">How Does This System Work?</button>
  </form>

  <!-- Start screen (hidden until opened by button or ?tour=1) -->
  <div id="tutorial-modal">
    <div class="modal-content">
      <h2>Interactive Tutorial</h2>
      <p>Follow a quick, guided walkthrough or skip.</p>
      <button id="skip">Skip</button>
      <button id="continue">Continue</button>
    </div>
  </div>
</body>
</html>
    '''

@app.route("/module_viewer")
def module_viewer():
    from math import sqrt

    raw = request.args.get("module", "centerpiece") or "centerpiece"
    module = raw.lower()

    # Integer-grid COB layouts per module
    # These are *before* the 45° diamond transform.
    if module == "centerpiece":
        mtype = "centerpiece"
        # center + 4 neighbors (this already gives the correct X after transform)
        cobs = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]

    elif module == "linear4":
        mtype = "linear4"
        # 4 emitters in a horizontal line after transform.
        # Using steps of (1,1) so the diamond transform makes them horizontal.
        cobs = [
            (-3, -3),
            (-1, -1),
            (1, 1),
            (3, 3),
        ]

    elif module == "linear3":
        mtype = "linear3"
        cobs = [
            (-2, -2),
            (0, 0),
            (2, 2),
        ]

    elif module == "linear2":
        mtype = "linear2"
        cobs = [
            (-1, -1),
            (1, 1),
        ]

    elif module in ("l", "L"):
        # L fixture: 3 emitters vertical + horizontal leg from the *top* emitter
        mtype = "L"
        # Vertical leg: steps of (1,-1) -> vertical line after transform
        # Top emitter at (2,-2), horizontal leg step (1,1) -> rightwards.
        cobs = [
            (0, 0),    # bottom of vertical
            (1, -1),   # middle
            (2, -2),   # top (corner)
            (3, -1),   # horizontal from top
        ]

    elif module in ("reverse_l", "reverse_l".lower(), "reverse_L"):
        # Reverse L: mirror horizontal leg to the other side
        mtype = "reverse_L"
        cobs = [
            (0, 0),    # bottom of vertical
            (1, -1),   # middle
            (2, -2),   # top (corner)
            (1, -3),   # horizontal from top, opposite side
        ]

    else:
        # Fallback to centerpiece
        mtype = "centerpiece"
        cobs = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]

    # Same 45° diamond transform as generate_visuals_general:
    # X = s*(x + y), Y = s*(x - y)
    scale = 22 / sqrt(2)
    rotated = [(scale * (a + b), scale * (a - b)) for a, b in cobs]

    if rotated:
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = min_y = -50
        max_x = max_y = 50

    margin = 40.0
    view_min_x = min_x - margin
    view_min_y = min_y - margin
    view_w = (max_x - min_x) + 2 * margin
    view_h = (max_y - min_y) + 2 * margin

    parts = []
    parts.append(
        f'<svg viewBox="{view_min_x:.2f} {view_min_y:.2f} {view_w:.2f} {view_h:.2f}" '
        'xmlns="http://www.w3.org/2000/svg" '
        'style="background:#000; font-family:system-ui;">'
    )

    # Emitters as white squares
    size = 10.0
    half = size / 2.0
    for px, py in rotated:
        parts.append(
            f'<rect x="{px - half:.2f}" y="{py - half:.2f}" '
            f'width="{size:.2f}" height="{size:.2f}" fill="#ffffff" />'
        )

    color = "#FFD700"

    # Connectors, mirroring your layout logic but adapted to these coordinates
    if mtype == "centerpiece":
        # Center to all arms (you said this is already perfect)
        cx, cy = rotated[0]
        for rx, ry in rotated[1:]:
            parts.append(
                f'<line x1="{cx:.2f}" y1="{cy:.2f}" '
                f'x2="{rx:.2f}" y2="{ry:.2f}" '
                f'stroke="{color}" stroke-width="2" />'
            )

    elif mtype.startswith("linear"):
        # Connect in order → straight horizontal line after transform
        if len(rotated) > 1:
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in rotated)
            parts.append(
                f'<polyline points="{path}" stroke="{color}" '
                f'stroke-width="2" fill="none" />'
            )

    elif mtype in ("L", "reverse_L"):
        # Explicitly connect [0→1→2→3] which gives:
        # 0→1→2 vertical leg, 2→3 horizontal leg from top.
        if len(rotated) >= 4:
            p0, p1, p2, p3 = rotated[0], rotated[1], rotated[2], rotated[3]
            path_leg = " ".join(
                f"{x:.2f},{y:.2f}" for (x, y) in (p0, p1, p2, p3)
            )
            parts.append(
                f'<polyline points="{path_leg}" stroke="{color}" '
                f'stroke-width="2" fill="none" />'
            )

    # Optional label
    label_x = view_min_x + 10
    label_y = view_min_y + 22
    parts.append(
        f'<text x="{label_x:.2f}" y="{label_y:.2f}" '
        'fill="#9ca3af" font-size="14">'
        f'{mtype}</text>'
    )

    parts.append("</svg>")
    svg = "\n".join(parts)
    return Response(svg, mimetype="image/svg+xml")


@app.route("/checkout", methods=["GET", "POST"])
def checkout():
    if request.method == "GET":
        return redirect("/app")

    def _coerce(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    length_ft = _coerce(request.form.get("length"), 0.0)
    width_ft = _coerce(request.form.get("width"), 0.0)
    if length_ft <= 0 or width_ft <= 0:
        return redirect("/app")

    posted_svg = request.form.get("layout_svg") or ""
    assembly_viewed = str(request.form.get("assembly_viewed", "0")).lower() in ("1", "true", "yes", "on")
    cv_enabled = False

    svg, computed_total, layout_data = compute_layout(
        length_ft, width_ft, cv_enabled,
        tutorial=False, wrap_flat_group=False, return_data=True
    )

    layout_svg = posted_svg or svg
    breakdown, subtotal = summarize_pricing(layout_data)
    total_cost = request.form.get("total_cost", type=float) or computed_total or subtotal

    area = max(length_ft * width_ft, 1e-6)
    avg_ppfd = max(550, min(1150, round(950 - area * 0.35)))
    min_ppfd = max(400, round(avg_ppfd * 0.92))
    max_ppfd = round(avg_ppfd * 1.05)
    uniformity = min(0.99, round(min_ppfd / avg_ppfd, 3))
    module_density = round(len(layout_data.get("module_groups", [])) / area, 3)

    report_metrics = {
        "avg_ppfd": None,
        "min_ppfd": None,
        "max_ppfd": None,
        "uniformity": None,
        "dou_percent": None,
        "area_sqft": round(area, 2),
        "modules": len(layout_data.get("module_groups", [])),
        "module_density": module_density,
    }

    assembly_url = f"/assembly?length={length_ft:.4f}&width={width_ft:.4f}&embed=1"
    len_label = f"{length_ft:.2f}".rstrip("0").rstrip(".")
    wid_label = f"{width_ft:.2f}".rstrip("0").rstrip(".")
    total_label = f"{total_cost:,.0f}"
    subtotal_label = f"{subtotal:,.0f}"
    total_num = total_cost

    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Checkout | Luminous Photonics</title>
  <style>
    :root { --gold:#FFD700; --bg:#000; --panel:#0d0d0d; --muted:#b7b7b7; }
    *{ box-sizing:border-box; }
    body{
      margin:0; padding:24px;
      background:radial-gradient(120% 120% at 20% 10%, rgba(255,215,0,0.14), transparent 52%), #000;
      color:#fff; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height:100vh;
    }
    header{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; margin-bottom:18px; }
    .logo{ font-weight:800; letter-spacing:0.08em; color:var(--gold); text-transform:uppercase; }
    .badge{ padding:6px 10px; border:1px solid rgba(255,255,255,0.12); border-radius:10px; color:#fff; background:rgba(255,255,255,0.06); }
    .grid{ display:grid; grid-template-columns: 2fr 1fr; gap:18px; }
    @media (max-width: 960px){ .grid{ grid-template-columns:1fr; } }
    .card{
      background:var(--panel);
      border:1px solid rgba(255,215,0,0.45);
      border-radius:14px;
      padding:16px;
      box-shadow:0 18px 36px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04) inset;
    }
    h1{ margin:0; color:var(--gold); }
    h2{ margin:0 0 8px; color:var(--gold); }
    h3{ margin:0 0 8px; color:var(--gold); font-size:1.1rem; }
    p{ color:#f5f5f5; line-height:1.6; }
    .layout-wrap{
      background:#050505;
      border:1px solid rgba(255,255,255,0.12);
      border-radius:12px;
      padding:12px;
      overflow:hidden;
    }
    .layout-wrap svg{ width:100%; height:auto; display:block; }
    .price-row{ display:flex; justify-content:space-between; align-items:center; margin:8px 0; color:#fff; }
    .muted{ color:var(--muted); font-size:0.95rem; }
    .pill{
      display:inline-flex; align-items:center; gap:6px;
      padding:8px 12px; border-radius:999px;
      background:rgba(255,215,0,0.15); color:#fff; border:1px solid rgba(255,215,0,0.45);
      font-weight:700;
    }
    .cta{
      display: inline-flex;              /* shrink to content */
      justify-content: center;
      align-items: center;
      padding: 12px 18px;
      border-radius: 12px;
      border: 1px solid transparent;
      background: var(--gold);
      color: #000;
      font-weight: 800;
      text-decoration: none;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      text-align: center;
      margin: 12px auto 0;               /* center the pill itself */
      width: fit-content;                /* key: don't stretch to 100% */
      transition: transform 0.12s ease, box-shadow 0.2s ease;
      box-shadow: 0 14px 28px rgba(255,215,0,0.28);
    }

    .cta:hover { transform: translateY(-1px); box-shadow: 0 16px 30px rgba(255,215,0,0.36); }
    .cta:active { transform: translateY(0); }

    .list{ list-style:none; padding:0; margin:0; display:grid; gap:10px; }
    .list li{ position:relative; display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.04); padding:12px 14px; border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.06); transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease; }
    .list li:hover{ border-color: rgba(255,215,0,0.45); box-shadow:0 14px 26px rgba(0,0,0,0.38); transform: translateY(-1px); }
    .cart-hover {
      position:absolute; inset:0; display:flex; justify-content:flex-end; align-items:center;
      background:linear-gradient(90deg, rgba(0,0,0,0), rgba(0,0,0,0.55));
      opacity:0; pointer-events:none; transition: opacity 0.2s ease;
      padding-right:12px;
    }
    .list li:hover .cart-hover { opacity:1; pointer-events:auto; }
    .cart-hover button {
      background: linear-gradient(135deg, #ffd700, #ffb347);
      color:#000; border:none; border-radius:20px; padding:8px 14px; font-weight:700; cursor:pointer;
      box-shadow:0 10px 20px rgba(0,0,0,0.35); transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .cart-hover button:hover { transform: translateY(-1px); box-shadow:0 12px 24px rgba(0,0,0,0.4); }
    .cart-hover button:active { transform: translateY(0); }
    .report-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:10px; }
    .report-cell{
      background:rgba(255,255,255,0.04);
      border:1px solid rgba(255,215,0,0.35);
      border-radius:10px; padding:10px;
    }
    iframe{
      width:100%; border:1px solid rgba(255,255,255,0.12); border-radius:12px; min-height:360px; background:#000;
      box-shadow:0 12px 28px rgba(0,0,0,0.45);
    }
    .inline{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  </style>
</head>
<body>
  <header>
    <div class="inline" style="gap:18px; align-items:center; width:100%; justify-content:space-between;">
      <a href="/index.html" style="display:inline-flex; align-items:center; gap:10px; text-decoration:none;">
        <img src="/pics/homeleaf.png" alt="Luminous Photonics Logo" style="width:150px; height:auto; display:block;">
      </a>
      <div style="text-align:right;">
        <div class="logo">Luminous Photonics</div>
        <h1 style="margin-top:4px;">Checkout</h1>
        <div class="inline" style="justify-content:flex-end; margin-top:6px;">
          <span class="pill">Grow space: {{ len_label }}′ × {{ wid_label }}′</span>
          <span class="pill">Total: ${{ total_label }}</span>
        </div>
      </div>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <h2>3D Assembly (live)</h2>
      <p class="muted">Interact with the assembly in 3D to verify canopy coverage.</p>
      <iframe src="{{ assembly_url }}" title="3D Assembly preview" loading="lazy"></iframe>

      <h3 style="margin-top:16px;">2D Layout Snapshot</h3>
      <p class="muted">Footprint and module positions captured from your generator run.</p>
      <div class="layout-wrap">{{ layout_svg|safe }}</div>
    </div>

    <div class="card">
      <h2>Cart Summary</h2>
      <ul class="list">
        {% for item in breakdown %}
          <li class="cart-item" data-module="{{ item.name }}">
            <div>
              <strong>{{ item.name }}</strong><br>
              <span class="muted">{{ item.count }} × ${{ item.unit_price }}</span>
            </div>
            <div><strong>${{ "{:,.0f}".format(item.line_total) }}</strong></div>
            <div class="cart-hover">
              <button type="button" class="view-module">
                View {{ item.name|capitalize }} Module
              </button>
            </div>
          </li>
        {% endfor %}
      </ul>

      <div class="price-row">
        <span class="muted">Subtotal</span>
        <span>${{ subtotal_label }}</span>
      </div>

      <div style="text-align:center;">
        <a class="cta"
          href="/checkout/complete?length={{ len_label }}&width={{ wid_label }}&total={{ total_num }}">
          Complete Checkout &gt;
        </a>
      </div>

      <p class="muted">Next: plug in payment and shipping to make this live.</p>
    </div>

  </div>

  <div class="card" style="margin-top:18px;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
      <h2 style="margin:0;">Lighting Report</h2>
      <button id="generate-report" class="cta" style="padding:10px 16px; margin:0;">Generate Lighting Report</button>
    </div>
    <p class="muted" style="margin-top:6px;">For larger grow space dimensions, load time for your lighting report could be significant.</p>
    <p id="report-status" class="muted" style="margin-top:4px;">Report not generated yet.</p>
    <div class="report-grid">
      <div class="report-cell"><strong>Area</strong><br><span class="muted" id="metric-area">{{ report_metrics.area_sqft }} sq ft</span></div>
      <div class="report-cell"><strong>Avg PPFD</strong><br><span class="muted" id="metric-avg">—</span></div>
      <div class="report-cell"><strong>Min / Max PPFD</strong><br><span class="muted" id="metric-minmax">—</span></div>
      <div class="report-cell"><strong>Uniformity (min/avg)</strong><br><span class="muted" id="metric-uni">—</span></div>
      <div class="report-cell"><strong>DOU (%)</strong><br><span class="muted" id="metric-dou">—</span></div>
      <div class="report-cell"><strong>Modules deployed</strong><br><span class="muted">{{ report_metrics.modules }} ({{ report_metrics.module_density }} modules/sq ft)</span></div>
      <div class="report-cell"><strong>Mounting height</strong><br><span class="muted">18"</span></div>
    </div>
    <div class="card" style="margin-top:12px; background:rgba(255,255,255,0.04); border-style:dashed;">
      <h3>Simulation Media</h3>
      <p class="muted">Media will appear after you generate the lighting report.</p>
      <div id="media-heatmap"></div>
      <div id="media-plots" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px,1fr)); gap:12px; margin-top:12px;"></div>
      <p id="media-csv" class="muted" style="margin-top:12px;"></p>
    </div>
  </div>

  <div id="module-modal"
      style="
        display:none;
        position:fixed;
        inset:0;
        background:rgba(0,0,0,0.80);
        z-index:9999;
        align-items:center;
        justify-content:center;
        padding:20px;
      ">
    <div style="
          background:#050505;
          border:1px solid rgba(255,215,0,0.4);
          border-radius:14px;
          width:min(900px, 94vw);
          padding:12px 12px 16px;
          position:relative;
          box-shadow:0 24px 48px rgba(0,0,0,0.5);
        ">
      <button id="module-close"
              style="
                position:absolute;
                top:8px;
                right:10px;
                background:none;
                color:#ffd700;
                border:none;
                font-size:20px;
                cursor:pointer;
              ">
        ×
      </button>
      <h3 id="module-title" style="margin:6px 0 10px;">Module</h3>
      <iframe id="module-frame"
              style="
                width:100%;
                height:520px;
                border:1px solid rgba(255,255,255,0.15);
                border-radius:10px;
                background:#000;
              "
              allowfullscreen
              loading="lazy"></iframe>
    </div>
  </div>


<script>
(() => {
  // --- Elements for module modal ---
  const modal       = document.getElementById('module-modal');
  const closeBtn    = document.getElementById('module-close');
  const moduleTitle = document.getElementById('module-title');
  const moduleFrame = document.getElementById('module-frame');

  function openModal(moduleName) {
    const name = (moduleName || 'centerpiece').toString();
    modal.style.display = 'flex';
    moduleTitle.textContent =
      'View ' + name.charAt(0).toUpperCase() + name.slice(1) + ' Module';
    moduleFrame.src = '/module_viewer?module=' + encodeURIComponent(name.toLowerCase());
  }

  function closeModal() {
    modal.style.display = 'none';
    moduleFrame.src = '';
  }

  if (closeBtn) {
    closeBtn.addEventListener('click', closeModal);
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
  }

  // --- Wire up Cart Summary buttons ---
  document.querySelectorAll('.view-module').forEach((btn) => {
    btn.addEventListener('click', () => {
      const li = btn.closest('.cart-item');
      const moduleName = li && li.dataset.module ? li.dataset.module : 'centerpiece';
      openModal(moduleName);
    });
  });

  // ==========================
  // Lighting report JS
  // ==========================
  const btn         = document.getElementById('generate-report');
  const statusEl    = document.getElementById('report-status');
  const metricAvg   = document.getElementById('metric-avg');
  const metricMinMax= document.getElementById('metric-minmax');
  const metricUni   = document.getElementById('metric-uni');
  const metricDou   = document.getElementById('metric-dou');
  const heatmapSlot = document.getElementById('media-heatmap');
  const plotsSlot   = document.getElementById('media-plots');
  const csvSlot     = document.getElementById('media-csv');

  const lenFt  = {{ length_ft|tojson }};
  const widFt  = {{ width_ft|tojson }};
  const target = 900;

  function setStatus(msg){ if(statusEl) statusEl.textContent = msg; }

  function setMedia({heatmap_url, surface_url, scatter_url, csv_url}){
    heatmapSlot.innerHTML = '';
    plotsSlot.innerHTML = '';
    csvSlot.textContent = '';
    if (heatmap_url){
      const img = document.createElement('img');
      img.src = heatmap_url;
      img.alt = 'PPFD heatmap';
      img.loading = 'lazy';
      img.style.width = '100%';
      img.style.borderRadius = '10px';
      img.style.border = '1px solid rgba(255,255,255,0.12)';
      heatmapSlot.appendChild(img);
    }
    if (surface_url){
      const wrap = document.createElement('div');
      const p = document.createElement('p'); p.className='muted'; p.textContent='PPFD Surface';
      const img = document.createElement('img');
      img.src = surface_url; img.alt='PPFD surface plot'; img.loading='lazy';
      img.style.width='100%'; img.style.borderRadius='10px'; img.style.border='1px solid rgba(255,255,255,0.12)';
      wrap.appendChild(p); wrap.appendChild(img); plotsSlot.appendChild(wrap);
    }
    if (scatter_url){
      const wrap = document.createElement('div');
      const p = document.createElement('p'); p.className='muted'; p.textContent='PPFD Scatter';
      const img = document.createElement('img');
      img.src = scatter_url; img.alt='PPFD scatter plot'; img.loading='lazy';
      img.style.width='100%'; img.style.borderRadius='10px'; img.style.border='1px solid rgba(255,255,255,0.12)';
      wrap.appendChild(p); wrap.appendChild(img); plotsSlot.appendChild(wrap);
    }
    if (csv_url){
      csvSlot.innerHTML = 'Raw data: <a href="' + csv_url + '" class="muted">PPFD CSV</a>';
    }
  }

  async function generateReport(){
    try{
      if (!btn) return;
      btn.disabled = true;
      btn.textContent = 'Generating...';
      setStatus('Running Radiance solver... this may take a bit for large rooms.');

      const resp = await fetch('/checkout/report', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({length: lenFt, width: widFt, target: target})
      });
      const data = await resp.json();
      if(!resp.ok || !data.ok){
        throw new Error(data.error || 'Unable to generate report');
      }
      if (metricAvg)    metricAvg.textContent =
        String(Math.round(data.metrics.avg_ppfd)) + ' µmol/m²/s';
      if (metricMinMax) metricMinMax.textContent =
        String(Math.round(data.metrics.min_ppfd)) + ' / ' +
        String(Math.round(data.metrics.max_ppfd)) + ' µmol/m²/s';
      if (metricUni)    metricUni.textContent =
        data.metrics.uniformity ? data.metrics.uniformity.toFixed(3) : '—';
      if (metricDou)    metricDou.textContent =
        data.metrics.dou_percent ? data.metrics.dou_percent.toFixed(2) : '—';

      setMedia({
        heatmap_url:  data.heatmap_url,
        surface_url:  data.surface_url,
        scatter_url:  data.scatter_url,
        csv_url:      data.csv_url
      });

      setStatus('Lighting report generated.');
    } catch(err){
      console.error(err);
      setStatus('Could not generate report. Please try again.');
    } finally{
      if (btn){
        btn.disabled = false;
        btn.textContent = 'Generate Lighting Report';
      }
    }
  }

  if (btn) {
    btn.addEventListener('click', generateReport);
  }
})();
</script>


</body>
</html>
    ''',
    breakdown=breakdown,
    subtotal=subtotal,
    total_cost=total_cost,
    subtotal_label=subtotal_label,
    total_label=total_label,
    total_num=total_num,
    len_label=len_label,
    wid_label=wid_label,
    length_ft=length_ft,
    width_ft=width_ft,
    layout_svg=layout_svg,
    assembly_url=assembly_url,
    assembly_viewed=assembly_viewed,
    report_metrics=report_metrics
    )

@app.post("/checkout/report")
def checkout_report():
    payload = request.get_json(silent=True) or {}
    def _c(k, d=0.0):
        try: return float(payload.get(k, d))
        except Exception: return d
    length_ft = _c("length", 0.0)
    width_ft = _c("width", 0.0)
    target_ppfd = _c("target", 900.0)
    if length_ft <= 0 or width_ft <= 0:
        return jsonify(ok=False, error="Invalid dimensions"), 400

    cv_enabled = False
    _, _, layout_data = compute_layout(length_ft, width_ft, cv_enabled, tutorial=False, wrap_flat_group=False, return_data=True)
    area = max(length_ft * width_ft, 1e-6)
    module_density = round(len(layout_data.get("module_groups", [])) / area, 3)

    solver_report = fetch_radiance_report(length_ft, width_ft, target_ppfd)
    if not solver_report:
        return jsonify(ok=False, error="Solver unavailable"), 500

    stats = solver_report.get("stats", {}) or {}
    metrics = {
        "avg_ppfd": stats.get("mean"),
        "min_ppfd": stats.get("min"),
        "max_ppfd": stats.get("max"),
        "uniformity": stats.get("uniformity"),
        "dou_percent": stats.get("dou_percent"),
        "modules": len(layout_data.get("module_groups", [])),
        "module_density": module_density,
        "area_sqft": round(area, 2),
    }

    return jsonify(ok=True,
                   metrics=metrics,
                   heatmap_url=solver_report.get("heatmap_url"),
                   surface_url=solver_report.get("surface_url"),
                   scatter_url=solver_report.get("scatter_url"),
                   csv_url=solver_report.get("csv_url"),
                   cell_key=solver_report.get("cell_key"))

@app.get("/checkout/complete")
def checkout_complete():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Complete Checkout | Luminous Photonics</title>
  <style>
    :root { --gold:#FFD700; --bg:#000; --panel:#0d0d0d; --muted:#b7b7b7; }
    *{ box-sizing:border-box; }
    body{
      margin:0; padding:24px;
      background:radial-gradient(120% 120% at 20% 10%, rgba(255,215,0,0.12), transparent 52%), #000;
      color:#fff; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height:100vh; display:flex; flex-direction:column; gap:18px;
    }
    header{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px; }
    .logo-link img{ width:150px; height:auto; display:block; }
    h1{ margin:0; color:var(--gold); }
    .card{
      background:var(--panel);
      border:1px solid rgba(255,215,0,0.45);
      border-radius:14px;
      padding:16px;
      box-shadow:0 18px 36px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04) inset;
      max-width:900px;
      width:100%;
      margin:0 auto;
    }
    form{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:12px; }
    label{ font-weight:700; color:#fff; }
    input, textarea{
      width:100%; padding:10px 12px;
      background:#050505; color:#fff;
      border:1px solid rgba(255,255,255,0.18);
      border-radius:10px;
    }
    textarea{ resize:vertical; min-height:120px; }
    .cta{
      display:inline-flex; justify-content:center; align-items:center;
      padding:12px 18px; border-radius:12px; border:1px solid transparent;
      background:var(--gold); color:#000; font-weight:800; text-decoration:none;
      letter-spacing:0.05em; text-transform:uppercase; margin-top:12px;
      transition:transform 0.12s ease, box-shadow 0.2s ease;
      box-shadow:0 14px 28px rgba(255,215,0,0.28);
    }
    .cta:hover{ transform:translateY(-1px); box-shadow:0 16px 30px rgba(255,215,0,0.36); }
    .cta:active{ transform:translateY(0); }
  </style>
</head>
<body>
  <header style="justify-content:center; text-align:center; gap:10px;">
    <a class="logo-link" href="/index.html"><img src="/pics/homeleaf.png" alt="Luminous Photonics Logo"></a>
    <h1 style="width:100%; text-align:center;">Complete Checkout</h1>
  </header>

  <div class="card" style="margin:0 auto;">
    <div class="price-row" style="justify-content:center; font-size:1.1rem;"><span class="muted" style="margin-right:8px;">Estimated Total</span><span><strong>${{ total_label }}</strong></span></div>
    <form>
      <div>
        <label for="name">Full Name</label>
        <input id="name" name="name" type="text" placeholder="Jane Doe">
      </div>
      <div>
        <label for="email">Email</label>
        <input id="email" name="email" type="email" placeholder="you@example.com">
      </div>
      <div>
        <label for="phone">Phone</label>
        <input id="phone" name="phone" type="tel" placeholder="(555) 555-5555">
      </div>
      <div>
        <label for="company">Company (optional)</label>
        <input id="company" name="company" type="text" placeholder="Farm Co.">
      </div>
      <div style="grid-column:1 / -1;">
        <label for="address">Shipping Address</label>
        <textarea id="address" name="address" placeholder="Street, City, State, ZIP"></textarea>
      </div>
      <div style="grid-column:1 / -1;">
        <label for="notes">Order Notes</label>
        <textarea id="notes" name="notes" placeholder="Anything we should know about your grow or delivery?"></textarea>
      </div>
    </form>
    <a class="cta" href="javascript:history.back()">Back to Cart</a>
  </div>

  <div class="card">
    <h2>Stripe Checkout (TEST MODE)</h2>
    <p style="color:#d5d5d5; margin:0 0 12px;">Pay securely with Stripe. This is TEST MODE — no real charges occur.</p>
    <button class="cta" id="pay-btn" type="button" style="margin-top:8px;">Pay with card</button>
  </div>

  <script src="https://js.stripe.com/v3/"></script>
  <script>
    (function(){
      const pk = {{ publishable_key|tojson }};
      const stripe = pk ? Stripe(pk) : null;
      const payBtn = document.getElementById('pay-btn');
      if (!stripe && payBtn){
        payBtn.disabled = true;
        payBtn.textContent = 'Stripe not configured';
        return;
      }
      payBtn?.addEventListener('click', async () => {
        payBtn.disabled = true;
        payBtn.textContent = 'Redirecting…';
        try{
          const params = new URLSearchParams(window.location.search);
          const rawTotal = (params.get('total') || '').replace(/,/g, '');
          const payload = {
            length: params.get('length') || '',
            width: params.get('width') || '',
            total_cost: parseFloat(rawTotal || '0') || 0
          };
          const resp = await fetch('/create-checkout-session', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify(payload)
          });
          const data = await resp.json();
          if (!resp.ok || !data.sessionId){
            throw new Error(data.error || 'Unable to create checkout session');
          }
          const { error } = await stripe.redirectToCheckout({ sessionId: data.sessionId });
          if (error) throw error;
        }catch(err){
          alert(err.message || 'Stripe checkout failed.');
          payBtn.disabled = false;
          payBtn.textContent = 'Pay with card';
        }
      });
    })();
  </script>
</body>
</html>
    ''', publishable_key=STRIPE_PUBLISHABLE_KEY)

# ---- Health / proxy / canonical host --------------------------------
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flask import request, redirect, Response

# Top-level Flask health (preferred)
@app.get("/healthz")
def healthz():
    return "ok", 200

# Optional: force apex -> www for all requests
@app.before_request
def _apex_to_www():
    host = request.headers.get("Host", "")
    if host == "luminousphotonics.com":
        path = request.full_path if request.query_string else request.path
        return redirect(f"https://www.luminousphotonics.com{path}", code=301)

# ---- Mount solver at /solver ----------------------------------------
def _create_solver_app():
    try:
        from radsim.solver import create_app as create_solver_app
        return create_solver_app()
    except ImportError:
        from radsim.solver import app as solver_app
        return solver_app

solver_app = _create_solver_app()

# WSGI-level health fallback (never 404s even if Flask routing is off)
def _health_wsgi(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"ok"]

# First mount solver and health fallback, then wrap with ProxyFix
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    "/solver": solver_app,
    "/healthz": _health_wsgi,   # absolute fallback; keeps Render happy
})
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# ---- Local dev ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # 5000 is taken locally
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=port)
