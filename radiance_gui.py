#!/usr/bin/env python3
# radiance_gui.py
# Simple, dependency-free Tk GUI for the Radiance workflow (SMD + SPYDR3).
# - One-click uniformity pipeline for SMD (runs run_uniformity.sh)
# - Optional SPYDR3 run (run_simulation_spydr3.sh)
# - Visualization (visualize_ppfd.py) with overlay selection
# - Advanced pane for solver knobs (w min/max, lambdas, chebyshev) and sim knobs (mode/OS)

import os
import math
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# Optional high-quality image scaling for logo
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = ImageTk = None

ROOT = Path(__file__).resolve().parent
FIX_X_IN = 47.0  # SPYDR3 fixture span in X
FIX_Y_IN = 43.0  # SPYDR3 fixture span in Y
DEFAULT_MARGIN_IN = 5.0
BG = "#000000"
ACCENT = "#FFD700"
FG = "#FFFFFF"
LOGO_MAX_WIDTH = 350  # px before downscale (integer subsample)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_cmd(cmd, env=None, cwd=None, on_line=None):
    """Run a command, streaming stdout/stderr; return exit code."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=cwd)
    for line in p.stdout:
        if on_line:
            on_line(line.rstrip())
    return p.wait()


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Radiance PPFD Workflow")
        self.geometry("1020x900")
        self.configure(bg=BG)

        self._load_styles()

        # Core variables
        self.mode = tk.StringVar(value="SMD")            # SMD | SPYDR3
        self.layout = tk.StringVar(value="square")       # square | rect_rect
        self.align_long_x = tk.BooleanVar(value=True)
        self.length_ft = tk.StringVar(value="12")
        self.width_ft = tk.StringVar(value="12")
        self.target_ppfd = tk.StringVar(value="1200")
        self.run_basis = tk.BooleanVar(value=False)

        # Solver knobs (advanced)
        self.w_min = tk.StringVar(value="10")
        self.w_max = tk.StringVar(value="150")
        self.lambda_s = tk.StringVar(value="0 1e-3 1e-2 1e-1 1.0 10.0")
        self.lambda_r = tk.StringVar(value="0 1e-3 1e-2 1e-1")
        self.lambda_mean = tk.StringVar(value="10")
        self.use_cheby = tk.BooleanVar(value=False)

        # Simulation knobs (shared)
        self.sim_mode = tk.StringVar(value="instant")    # instant|fast|quality|direct
        self.os = tk.StringVar(value="4")
        self.subgrid = tk.StringVar(value="1")
        self.optics = tk.StringVar(value="none")

        # Visualization
        self.overlay = tk.StringVar(value="auto")

        # SPYDR3 specifics
        self.sp_nx = tk.StringVar(value="auto")
        self.sp_ny = tk.StringVar(value="auto")
        self.sp_ppf = tk.StringVar(value="2200")
        self.sp_z_m = tk.StringVar(value="0.4572")
        self.sp_eff = tk.StringVar(value="1.0")
        self.sp_ppe = tk.StringVar(value="2.7")
        self.sp_margin_in = tk.StringVar(value=str(DEFAULT_MARGIN_IN))  # fixed 5" inset

        # Layout
        self._build_scrollable_ui()
        self._build_log()
        self._running = False

    def _load_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        base_font = ("Helvetica Neue", 11)
        style.configure(".", background=BG, foreground=FG, font=base_font)
        style.configure("Card.TFrame", background=BG)
        style.configure("Card.TLabel", background=BG, foreground=FG, font=base_font)
        style.configure("Card.TLabelframe", background=BG, foreground=FG, bordercolor=ACCENT, relief="solid")
        style.configure("Card.TLabelframe.Label", background=BG, foreground=ACCENT, font=("Helvetica Neue", 11, "bold"))
        style.configure("Card.TCheckbutton", background=BG, foreground=FG, focuscolor=ACCENT)
        style.configure("Accent.TButton", background=ACCENT, foreground=BG, padding=(10,6), borderwidth=0, font=("Helvetica Neue", 11, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#e6c200")],
                  relief=[("pressed","sunken")])
        style.configure("Accent.TRadiobutton", background=BG, foreground=FG, indicatorcolor=ACCENT, focuscolor=ACCENT)
        style.configure("Title.TLabel", background=BG, foreground=ACCENT, font=("Helvetica Neue", 18, "bold"))
        style.configure("TEntry", fieldbackground="#111111", foreground=FG, insertcolor=FG, bordercolor=ACCENT, relief="flat")
        style.configure("TCombobox", fieldbackground="#111111", background="#111111", foreground=FG, arrowcolor=ACCENT)
        style.configure("Accent.Vertical.TScrollbar", background=BG, troughcolor="#111111", bordercolor=ACCENT, arrowcolor=ACCENT)

    def _load_logo(self):
        # Attempt to load a logo image (PNG preferred). Returns PhotoImage or None.
        for name in (os.getenv("LOGO_PATH"), "luminous_logo.jpg", "logo.png", "luminous.png"):
            if not name:
                continue
            p = ROOT / name
            if p.exists():
                try:
                    if Image and ImageTk:
                        img = Image.open(p)
                        w, h = img.size
                        if w > LOGO_MAX_WIDTH:
                            new_w = LOGO_MAX_WIDTH
                            new_h = int(h * (new_w / w))
                            img = img.resize((new_w, new_h), Image.LANCZOS)
                        return ImageTk.PhotoImage(img)
                    else:
                        img = tk.PhotoImage(file=str(p))
                        if img.width() > LOGO_MAX_WIDTH:
                            factor = max(1, int(math.ceil(img.width()/LOGO_MAX_WIDTH)))
                            img = img.subsample(factor, factor)
                        return img
                except Exception:
                    continue
        return None

    # UI builders ----------------------------------------------------------- #
    def _build_scrollable_ui(self):
        # Scrollable container to avoid clipped controls on small screens
        container = ttk.Frame(self, style="Card.TFrame")
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0, bg=BG)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview, style="Accent.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.ctrl_frame = ttk.Frame(canvas, style="Card.TFrame")
        self.ctrl_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_controls(self.ctrl_frame)

    def _build_controls(self, root):
        # Header
        header = ttk.Frame(root, style="Card.TFrame"); header.pack(fill="x", padx=10, pady=(6,2))
        logo = self._load_logo()
        if logo:
            ttk.Label(header, image=logo, style="Card.TLabel").pack(side="left", padx=(0,12))
            self._logo_img = logo
        ttk.Label(header, text="Luminous Photonics – Radiance Workflow", style="Title.TLabel").pack(side="left", anchor="w")

        top = ttk.Frame(root, style="Card.TFrame"); top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Mode:", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(top, text="SMD", variable=self.mode, value="SMD", style="Accent.TRadiobutton").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(top, text="SPYDR3", variable=self.mode, value="SPYDR3", style="Accent.TRadiobutton").grid(row=0, column=2, sticky="w")

        ttk.Label(top, text="Layout", style="Card.TLabel").grid(row=0, column=3, sticky="e", padx=(20, 2))
        ttk.Combobox(top, textvariable=self.layout, width=10,
                     values=["square", "rect_rect"], state="readonly").grid(row=0, column=4, sticky="w")
        ttk.Checkbutton(top, text="Align long axis to X", variable=self.align_long_x, style="Card.TCheckbutton").grid(row=0, column=5, sticky="w", padx=(12,0))

        dims = ttk.LabelFrame(root, text="Room", style="Card.TLabelframe"); dims.pack(fill="x", padx=10, pady=6)
        ttk.Label(dims, text="Length (ft)").grid(row=0, column=0, sticky="w")
        ttk.Entry(dims, textvariable=self.length_ft, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(dims, text="Width (ft)").grid(row=0, column=2, sticky="w", padx=(12,0))
        ttk.Entry(dims, textvariable=self.width_ft, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(dims, text="Target PPFD").grid(row=0, column=4, sticky="w", padx=(18,0))
        ttk.Entry(dims, textvariable=self.target_ppfd, width=8).grid(row=0, column=5, sticky="w")
        ttk.Checkbutton(dims, text="Run basis (rebuild A)", variable=self.run_basis, style="Card.TCheckbutton").grid(row=0, column=6, sticky="w", padx=(18,0))

        vis = ttk.LabelFrame(root, text="Visualization", style="Card.TLabelframe"); vis.pack(fill="x", padx=10, pady=6)
        ttk.Label(vis, text="Overlay").grid(row=0, column=0, sticky="w")
        ttk.Combobox(vis, textvariable=self.overlay, width=10,
                     values=["auto","smd","spydr3","both","none"], state="readonly").grid(row=0, column=1, sticky="w")

        # Advanced toggler
        self.adv_frame = ttk.LabelFrame(root, text="Advanced (solver + sim)", style="Card.TLabelframe"); self.adv_visible = tk.BooleanVar(value=False)
        btn = ttk.Checkbutton(root, text="Show advanced options", variable=self.adv_visible,
                              command=self._toggle_adv)
        btn.pack(anchor="w", padx=12, pady=(4,0))
        self._build_advanced()

        # Action buttons
        btns = ttk.Frame(root, style="Card.TFrame"); btns.pack(fill="x", padx=10, pady=8)
        ttk.Button(btns, text="Run Uniformity (SMD)", command=self.run_smd, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="Run SPYDR3", command=self.run_spydr, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="Visualize", command=self.run_vis, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="Run + Visualize (SMD)", command=self.run_all_smd, style="Accent.TButton").pack(side="left", padx=4)

    def _build_advanced(self):
        f = self.adv_frame
        # Solver
        sol = ttk.Frame(f, style="Card.TFrame"); sol.pack(fill="x", padx=8, pady=6)
        ttk.Label(sol, text="W min").grid(row=0, column=0, sticky="w")
        ttk.Entry(sol, textvariable=self.w_min, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(sol, text="W max").grid(row=0, column=2, sticky="w", padx=(12,0))
        ttk.Entry(sol, textvariable=self.w_max, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(sol, text="lambda_s list").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Entry(sol, textvariable=self.lambda_s, width=28).grid(row=1, column=1, columnspan=3, sticky="w", pady=(6,0))
        ttk.Label(sol, text="lambda_r list").grid(row=2, column=0, sticky="w", pady=(6,0))
        ttk.Entry(sol, textvariable=self.lambda_r, width=28).grid(row=2, column=1, columnspan=3, sticky="w", pady=(6,0))
        ttk.Label(sol, text="lambda_mean").grid(row=3, column=0, sticky="w", pady=(6,0))
        ttk.Entry(sol, textvariable=self.lambda_mean, width=10).grid(row=3, column=1, sticky="w", pady=(6,0))
        ttk.Checkbutton(sol, text="Use Chebyshev minimax solver", variable=self.use_cheby).grid(row=3, column=2, columnspan=2, sticky="w", padx=(12,0), pady=(6,0))

        # Simulation knobs
        sim = ttk.Frame(f, style="Card.TFrame"); sim.pack(fill="x", padx=8, pady=4)
        ttk.Label(sim, text="Sim mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(sim, textvariable=self.sim_mode, values=["instant","fast","quality","direct"],
                     width=10, state="readonly").grid(row=0, column=1, sticky="w")
        ttk.Label(sim, text="Oversample (OS)").grid(row=0, column=2, sticky="w", padx=(12,0))
        ttk.Entry(sim, textvariable=self.os, width=6).grid(row=0, column=3, sticky="w")
        ttk.Label(sim, text="Subpatch grid").grid(row=0, column=4, sticky="w", padx=(12,0))
        ttk.Entry(sim, textvariable=self.subgrid, width=6).grid(row=0, column=5, sticky="w")
        ttk.Label(sim, text="Optics").grid(row=0, column=6, sticky="w", padx=(12,0))
        ttk.Combobox(sim, textvariable=self.optics, values=["none","lens"], width=8, state="readonly").grid(row=0, column=7, sticky="w")

        # SPYDR advanced
        sp = ttk.Frame(f, style="Card.TFrame"); sp.pack(fill="x", padx=8, pady=6)
        ttk.Label(sp, text="Auto NX × NY").grid(row=0, column=0, sticky="w")
        ttk.Label(sp, textvariable=self.sp_nx).grid(row=0, column=1, sticky="w")
        ttk.Label(sp, text="×").grid(row=0, column=2, sticky="w")
        ttk.Label(sp, textvariable=self.sp_ny).grid(row=0, column=3, sticky="w")
        ttk.Label(sp, text="PPF per fixture (µmol/s)").grid(row=0, column=4, sticky="w", padx=(12,0))
        ttk.Entry(sp, textvariable=self.sp_ppf, width=10).grid(row=0, column=5, sticky="w")
        ttk.Label(sp, text="Z (m)").grid(row=0, column=6, sticky="w", padx=(12,0))
        ttk.Entry(sp, textvariable=self.sp_z_m, width=8).grid(row=0, column=7, sticky="w")
        ttk.Label(sp, text="Eff scale").grid(row=0, column=8, sticky="w", padx=(12,0))
        ttk.Entry(sp, textvariable=self.sp_eff, width=8).grid(row=0, column=9, sticky="w")
        ttk.Label(sp, text="PPE (µmol/J)").grid(row=0, column=10, sticky="w", padx=(12,0))
        ttk.Entry(sp, textvariable=self.sp_ppe, width=8).grid(row=0, column=11, sticky="w")
        ttk.Label(sp, text="Margin (in, fixed)").grid(row=0, column=12, sticky="w", padx=(12,0))
        ttk.Label(sp, textvariable=self.sp_margin_in).grid(row=0, column=13, sticky="w")

    def _toggle_adv(self):
        if self.adv_visible.get():
            self.adv_frame.pack(fill="x", padx=10, pady=6)
        else:
            self.adv_frame.forget()

    def _build_log(self):
        frame = ttk.Frame(self, style="Card.TFrame")
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.log = tk.Text(frame, wrap="word", height=18, bg=BG, fg=FG, insertbackground=ACCENT)
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

    # Utility --------------------------------------------------------------- #
    def append(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _thread(self, func):
        if self._running:
            messagebox.showinfo("Busy", "A task is already running.")
            return
        self._running = True
        def run():
            try:
                func()
            finally:
                self._running = False
        threading.Thread(target=run, daemon=True).start()

    # Build env dicts ------------------------------------------------------- #
    def _base_env(self):
        env = os.environ.copy()
        env["LENGTH_FT"] = self.length_ft.get().strip()
        env["WIDTH_FT"] = self.width_ft.get().strip()
        env["MODE"] = self.sim_mode.get()
        env["OS"] = self.os.get().strip() or "4"
        env["SUBPATCH_GRID"] = self.subgrid.get().strip() or "1"
        return env

    def _smd_env(self):
        env = self._base_env()
        env["LAYOUT_MODE"] = self.layout.get()
        env["ALIGN_LONG_AXIS_X"] = "1" if self.align_long_x.get() else "0"
        env["TARGET_PPFD"] = self.target_ppfd.get().strip() or "1200"
        env["RUN_BASIS"] = "1" if self.run_basis.get() else "0"
        env["W_MIN"] = self.w_min.get().strip() or "10"
        env["W_MAX"] = self.w_max.get().strip() or "100"
        env["LAMBDA_S"] = self.lambda_s.get().strip()
        env["LAMBDA_R"] = self.lambda_r.get().strip()
        env["LAMBDA_MEAN"] = self.lambda_mean.get().strip() or "10"
        env["USE_CHEBYSHEV"] = "1" if self.use_cheby.get() else "0"
        env["OPTICS"] = self.optics.get()
        return env

    def _spydr_env(self):
        env = self._base_env()
        # auto-compute grid from room size and fixed 5" margin
        try:
            L_ft = float(self.length_ft.get())
            W_ft = float(self.width_ft.get())
        except Exception:
            L_ft = W_ft = 0.0
        # Use the larger of wall margin and half sensor step so fixtures stay inside heatmap bbox
        res_x = int(os.getenv("RESOLUTION_X", "15"))
        res_y = int(os.getenv("RESOLUTION_Y", "15"))
        step_x_in = (L_ft * 12.0) / max(1, res_x - 1)
        step_y_in = (W_ft * 12.0) / max(1, res_y - 1)
        sensor_margin = 0.5 * max(step_x_in, step_y_in)
        margin = max(DEFAULT_MARGIN_IN, sensor_margin)
        L_in = max(0.0, L_ft * 12.0 - 2 * margin)
        W_in = max(0.0, W_ft * 12.0 - 2 * margin)
        nx = max(1, int(math.floor(W_in / FIX_X_IN)))
        ny = max(1, int(math.floor(L_in / FIX_Y_IN)))
        self.sp_nx.set(str(nx))
        self.sp_ny.set(str(ny))
        env["NX"] = str(nx)
        env["NY"] = str(ny)
        env["SPYDR_PPF"] = self.sp_ppf.get().strip() or "2200"
        env["SPYDR_Z_M"] = self.sp_z_m.get().strip() or "0.4572"
        env["EFF_SCALE"] = self.sp_eff.get().strip() or "1.0"
        env["SPYDR_PPE_UMOL_PER_J"] = self.sp_ppe.get().strip() or "2.7"
        env["TARGET_PPFD"] = self.target_ppfd.get().strip() or "1200"
        env["MARGIN_IN"] = f"{margin:.3f}"
        return env

    # Actions --------------------------------------------------------------- #
    def run_smd(self):
        self._thread(self._run_smd_impl)

    def _run_smd_impl(self):
        env = self._smd_env()
        cmd = ["bash", str(ROOT / "run_uniformity.sh")]
        self.append(f"$ {' '.join(cmd)}")
        rc = run_cmd(cmd, env=env, cwd=str(ROOT), on_line=self.append)
        self.append(f"[exit {rc}]")

    def run_spydr(self):
        self._thread(self._run_spydr_impl)

    def _run_spydr_impl(self):
        env = self._spydr_env()
        cmd = ["bash", str(ROOT / "run_simulation_spydr3.sh")]
        self.append(f"$ {' '.join(cmd)}")
        rc = run_cmd(cmd, env=env, cwd=str(ROOT), on_line=self.append)
        self.append(f"[exit {rc}]")

    def run_vis(self):
        self._thread(self._run_vis_impl)

    def _run_vis_impl(self):
        env = os.environ.copy()
        outdir = "ppfd_visualizations" if self.mode.get() == "SMD" else "ppfd_visualizations_spydr3"
        overlay = self.overlay.get()
        if self.mode.get() == "SPYDR3" and overlay == "auto":
            overlay = "spydr3"
        cmd = ["python3", str(ROOT / "visualize_ppfd.py"),
               "--overlay", overlay,
               "--outdir", outdir]
        self.append(f"$ {' '.join(cmd)}")
        rc = run_cmd(cmd, env=env, cwd=str(ROOT), on_line=self.append)
        self.append(f"[exit {rc}]")

    def run_all_smd(self):
        self._thread(self._run_all_smd_impl)

    def _run_all_smd_impl(self):
        self._run_smd_impl()
        self._run_vis_impl()


if __name__ == "__main__":
    App().mainloop()
