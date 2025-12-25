#!/usr/bin/env python3
"""
radiance_gui_qt.py
PySide6 desktop launcher for the Radiance workflow (SMD + SPYDR3).

Features:
- Clean dark theme with accent color (#FFD700).
- Inputs: length/width, layout, align axis, target PPFD, run basis, solver knobs, sim mode/OS, overlay, SPYDR params.
- Buttons: Run Uniformity (SMD), Run SPYDR3, Visualize, Run+Visualize.
- Live log console fed from subprocess stdout/stderr.

Requires: pip install PySide6

Usage:
  python3 radiance_gui_qt.py
"""
import os
import sys
import subprocess
from pathlib import Path
from PySide6 import QtWidgets, QtGui, QtCore

# --- optional metrics helpers (cap-based PPFD metrics) ---
try:
    import numpy as np
    from ppfd_metrics import compute_ppfd_metrics, format_ppfd_metrics_line
except Exception:  # pragma: no cover
    np = None
    compute_ppfd_metrics = None
    format_ppfd_metrics_line = None


ROOT = Path(__file__).resolve().parent
ACCENT = "#FFD700"
BG = "#000000"
FG = "#FFFFFF"
LOGO_MAX_W = 420


class LogConsole(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {BG};
                color: {FG};
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 11pt;
                border: 1px solid {ACCENT};
            }}
        """)

    def append_line(self, text: str):
        self.appendPlainText(text.rstrip())
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None, collapsed=False):
        super().__init__(parent)
        self.toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=not collapsed)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        self.toggle.setStyleSheet(f"""
            QToolButton {{
                color: {ACCENT};
                background: {BG};
                border: 1px solid {ACCENT};
                padding: 6px;
                font-weight: 600;
            }}
        """)
        self.toggle.clicked.connect(self._on_toggled)

        self.content = QtWidgets.QFrame()
        self.content.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.content.setStyleSheet(f"QFrame {{ border: 1px solid {ACCENT}; background:{BG}; }}")
        self.content_lay = QtWidgets.QVBoxLayout(self.content)
        self.content_lay.setContentsMargins(8, 8, 8, 8)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

        if collapsed:
            self.content.setVisible(False)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def layout(self) -> QtWidgets.QVBoxLayout:
        return self.content_lay


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiance Lighting Simulation Engine")
        self.setMinimumSize(1100, 760)
        self._procs = []

        self._build_ui()
        # Fit within available screen (leave room for dock/menu)
        scr = QtGui.QGuiApplication.primaryScreen()
        if scr:
            avail = scr.availableGeometry()
            target_w = min(max(1100, int(avail.width()*0.9)), avail.width())
            target_h = min(max(760, int(avail.height()*0.82)), avail.height())
            self.resize(target_w, target_h)

    def _build_ui(self):
        self.setStyleSheet(f"""
            QWidget {{ background: {BG}; color: {FG}; font-family: 'Helvetica Neue', Arial; font-size: 11pt; }}
            QLineEdit {{
                background: #0e0e0e; color: {FG}; border: 1px solid {ACCENT}; padding: 6px;
            }}
            QComboBox {{
                background: #0e0e0e; color: {FG}; border: 1px solid {ACCENT}; padding: 8px; font-size: 12pt; font-weight: 600;
            }}
            QComboBox QAbstractItemView {{
                background: #0e0e0e; color: {FG}; selection-background-color: {ACCENT}; selection-color: {BG}; font-size: 12pt; padding: 6px;
            }}
            QComboBox:hover {{
                border: 1px solid #ffea66;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: #ffea66; color: {BG};
            }}
            QCheckBox, QLabel {{ color: {FG}; }}
            QPushButton {{
                background: {ACCENT}; color: {BG}; border: none; padding: 9px 14px; font-weight: 700; border-radius: 4px;
            }}
            QPushButton:hover {{
                background: #ffea66; color: {BG};
            }}
            QPushButton:disabled {{ background: #444; color: #999; }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8,8,8,8)

        # Header with logo + title
        header = QtWidgets.QHBoxLayout()
        logo_label = QtWidgets.QLabel()
        logo_pix = self._load_logo()
        if logo_pix:
            logo_label.setPixmap(logo_pix)
        logo_label.setMaximumHeight(72)
        logo_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        header.addWidget(logo_label, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        title = QtWidgets.QLabel("Radiance Lighting Simulation Engine")
        title.setStyleSheet(f"color:{ACCENT}; font-size: 20pt; font-weight: 800;")
        header.addWidget(title, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        header.addStretch(1)
        layout.addLayout(header)

        # --- Top controls ---
        top = QtWidgets.QGridLayout()
        row = 0
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["SMD", "COB", "Quantum Board", "Competitor"])
        top.addWidget(QtWidgets.QLabel("Mode"), row, 0)
        top.addWidget(self.mode, row, 1)

        self.layout_mode = QtWidgets.QComboBox(); self.layout_mode.addItems(["square", "rect_rect"])
        top.addWidget(QtWidgets.QLabel("Layout"), row, 2)
        top.addWidget(self.layout_mode, row, 3)

        self.align_x = QtWidgets.QCheckBox("Align long axis to X"); self.align_x.setChecked(True)
        top.addWidget(self.align_x, row, 4)

        self.run_basis = QtWidgets.QCheckBox("Run basis (rebuild A)")
        top.addWidget(self.run_basis, row, 5)

        # Scrollable controls area
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(4,4,4,4)

        controls_layout.addLayout(top)

        # --- Room ---
        room_box = QtWidgets.QGroupBox("Room")
        room = QtWidgets.QGridLayout(room_box)
        self.length = QtWidgets.QLineEdit("12")
        self.width = QtWidgets.QLineEdit("12")
        self.target = QtWidgets.QLineEdit("1000")
        room.addWidget(QtWidgets.QLabel("Length (ft)"), 0, 0); room.addWidget(self.length, 0, 1)
        room.addWidget(QtWidgets.QLabel("Width (ft)"), 0, 2);  room.addWidget(self.width, 0, 3)
        room.addWidget(QtWidgets.QLabel("Target PPFD"), 0, 4); room.addWidget(self.target, 0, 5)
        controls_layout.addWidget(room_box)

        # --- Collapsible panels in a row ---
        row_boxes = QtWidgets.QHBoxLayout()
        row_boxes.setSpacing(10)

        solver_box = CollapsibleBox("Solver", collapsed=True)
        sol = QtWidgets.QGridLayout()
        self.wmin = QtWidgets.QLineEdit("10"); self.wmax = QtWidgets.QLineEdit("100")
        self.lams = QtWidgets.QLineEdit("0 1e-3 1e-2 1e-1 1.0 10.0")
        self.lamr = QtWidgets.QLineEdit("0 1e-3 1e-2 1e-1")
        self.lammean = QtWidgets.QLineEdit("10")
        self.cheby = QtWidgets.QCheckBox("Use Chebyshev")
        self.smd_outer_per_module = QtWidgets.QCheckBox("SMD outer ring per-module")
        self.smd_outer_per_module.setChecked(False)
        sol.addWidget(QtWidgets.QLabel("W min"), 0, 0); sol.addWidget(self.wmin, 0, 1)
        sol.addWidget(QtWidgets.QLabel("W max"), 0, 2); sol.addWidget(self.wmax, 0, 3)
        sol.addWidget(QtWidgets.QLabel("lambda_s"), 1, 0); sol.addWidget(self.lams, 1, 1, 1, 3)
        sol.addWidget(QtWidgets.QLabel("lambda_r"), 2, 0); sol.addWidget(self.lamr, 2, 1, 1, 3)
        sol.addWidget(QtWidgets.QLabel("lambda_mean"), 3, 0); sol.addWidget(self.lammean, 3, 1)
        sol.addWidget(self.cheby, 3, 2, 1, 2)
        sol.addWidget(self.smd_outer_per_module, 4, 0, 1, 4)
        solver_box.layout().addLayout(sol)
        row_boxes.addWidget(solver_box, 1)

        sim_box = CollapsibleBox("Simulation", collapsed=True)
        sim = QtWidgets.QGridLayout()
        self.sim_mode = QtWidgets.QComboBox(); self.sim_mode.addItems(["instant","fast","quality","direct"])
        self.os = QtWidgets.QLineEdit("4")
        self.subgrid = QtWidgets.QLineEdit("1")
        self.mount_z = QtWidgets.QLineEdit("0.4572")
        self.optics = QtWidgets.QComboBox(); self.optics.addItems(["none","lens"])
        self.overlay = QtWidgets.QComboBox(); self.overlay.addItems(["auto","smd","spydr3","quantum","cob","both","none"])
        self.qb_perim = QtWidgets.QCheckBox("Fill Perimeter (Quantum)")
        self.qb_perim.setChecked(False)
        self.smd_perim_fill = QtWidgets.QCheckBox("SMD perimeter gap fill")
        self.smd_perim_fill.setChecked(False)
        self.smd_outer_optics = QtWidgets.QCheckBox("SMD outer ring optics")
        self.smd_outer_optics.setChecked(False)
        self.smd_outer_fwhm = QtWidgets.QLineEdit("40")
        self.smd_base_ring = QtWidgets.QLineEdit("7")
        self.cob_base_ring = QtWidgets.QLineEdit("6")
        self.cob_wall_clear_in = QtWidgets.QLineEdit("3.0")
        sim.addWidget(QtWidgets.QLabel("Sim mode"), 0, 0); sim.addWidget(self.sim_mode, 0, 1)
        sim.addWidget(QtWidgets.QLabel("Oversample (OS)"), 0, 2); sim.addWidget(self.os, 0, 3)
        sim.addWidget(QtWidgets.QLabel("Subpatch grid"), 1, 0); sim.addWidget(self.subgrid, 1, 1)
        sim.addWidget(QtWidgets.QLabel("Mount Z (m)"), 1, 2); sim.addWidget(self.mount_z, 1, 3)
        sim.addWidget(QtWidgets.QLabel("Optics"), 2, 2); sim.addWidget(self.optics, 2, 3)
        sim.addWidget(QtWidgets.QLabel("Overlay"), 2, 0); sim.addWidget(self.overlay, 2, 1)
        sim.addWidget(self.qb_perim, 3, 0, 1, 4)
        sim.addWidget(self.smd_perim_fill, 4, 0, 1, 4)
        sim.addWidget(self.smd_outer_optics, 5, 0, 1, 2)
        sim.addWidget(QtWidgets.QLabel("Outer FWHM (deg)"), 5, 2); sim.addWidget(self.smd_outer_fwhm, 5, 3)
        sim.addWidget(QtWidgets.QLabel("SMD base ring (pitch)"), 6, 0); sim.addWidget(self.smd_base_ring, 6, 1)
        sim.addWidget(QtWidgets.QLabel("COB base ring (pitch)"), 7, 0); sim.addWidget(self.cob_base_ring, 7, 1)
        sim.addWidget(QtWidgets.QLabel("COB wall clearance (in)"), 8, 0); sim.addWidget(self.cob_wall_clear_in, 8, 1)
        sim_box.layout().addLayout(sim)
        row_boxes.addWidget(sim_box, 1)

        sp_box = CollapsibleBox("Competitor", collapsed=True)
        sp = QtWidgets.QGridLayout()
        self.sp_ppf = QtWidgets.QLineEdit("2200")
        self.sp_z = QtWidgets.QLineEdit("0.4572")
        self.sp_eff = QtWidgets.QLineEdit("1.0")
        self.sp_ppe = QtWidgets.QLineEdit("2.76")
        self.sp_margin = QtWidgets.QLineEdit("0")
        self.sp_edge_inset = QtWidgets.QLineEdit("0")
        sp.addWidget(QtWidgets.QLabel("PPF per fixture (µmol/s)"), 0, 0); sp.addWidget(self.sp_ppf, 0, 1)
        sp.addWidget(QtWidgets.QLabel("Mount Z (m)"), 0, 2); sp.addWidget(self.sp_z, 0, 3)
        sp.addWidget(QtWidgets.QLabel("Eff scale"), 1, 0); sp.addWidget(self.sp_eff, 1, 1)
        sp.addWidget(QtWidgets.QLabel("PPE (µmol/J)"), 1, 2); sp.addWidget(self.sp_ppe, 1, 3)
        sp.addWidget(QtWidgets.QLabel("Margin (in)"), 2, 0); sp.addWidget(self.sp_margin, 2, 1)
        sp.addWidget(QtWidgets.QLabel("Edge inset (in)"), 2, 2); sp.addWidget(self.sp_edge_inset, 2, 3)
        sp_box.layout().addLayout(sp)
        row_boxes.addWidget(sp_box, 1)

        controls_layout.addLayout(row_boxes)

        # --- Buttons ---
        btns = QtWidgets.QHBoxLayout()
        self.btn_run_smd = QtWidgets.QPushButton("Run Uniformity")
        self.btn_run_spy = QtWidgets.QPushButton("Run Competitor")
        self.btn_vis = QtWidgets.QPushButton("Visualize")
        self.btn_all = QtWidgets.QPushButton("Run + Visualize")
        btns.addWidget(self.btn_run_smd); btns.addWidget(self.btn_run_spy); btns.addWidget(self.btn_vis); btns.addWidget(self.btn_all)
        controls_layout.addLayout(btns)

        # Wrap controls in scroll area (so expanded panels can scroll)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_widget)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # --- Viewer and Log split ---
        self.viewer_tabs = QtWidgets.QTabWidget()
        self.viewer_tabs.setStyleSheet(f"QTabWidget::pane {{ border:1px solid {ACCENT}; }} QTabBar::tab {{ background:#111; color:{FG}; padding:6px; }} QTabBar::tab:selected {{ background:{ACCENT}; color:{BG}; }}")
        self.img_overlay = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img_overlay.setStyleSheet(f"background-color:#111; border:1px solid {ACCENT};")
        self.img_annot = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img_annot.setStyleSheet(f"background-color:#111; border:1px solid {ACCENT};")
        self.img_surface = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img_surface.setStyleSheet(f"background-color:#111; border:1px solid {ACCENT};")
        self.viewer_tabs.addTab(self.img_overlay, "Overlay")
        self.viewer_tabs.addTab(self.img_annot, "Annotated")
        self.viewer_tabs.addTab(self.img_surface, "3D Surface")
        # 3D scatter fallback button
        self.btn_open_3d = QtWidgets.QPushButton("Open 3D scatter in browser")
        vtab = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(vtab)
        vlay.addWidget(self.btn_open_3d)
        vlay.addStretch(1)
        self.viewer_tabs.addTab(vtab, "3D Scatter")

        # Metrics
        self.metrics_text = QtWidgets.QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #0f0f0f;
                color: {FG};
                border: 1px solid {ACCENT};
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 11pt;
            }}
        """)
        self.btn_refresh_metrics = QtWidgets.QPushButton("Refresh metrics")
        self.btn_copy_metrics = QtWidgets.QPushButton("Copy metrics")
        mtab = QtWidgets.QWidget()
        mlay = QtWidgets.QVBoxLayout(mtab)
        mbtns = QtWidgets.QHBoxLayout()
        mbtns.addWidget(self.btn_refresh_metrics)
        mbtns.addWidget(self.btn_copy_metrics)
        mbtns.addStretch(1)
        mlay.addLayout(mbtns)
        mlay.addWidget(self.metrics_text)
        self.viewer_tabs.addTab(mtab, "Metrics")

        self.log = LogConsole()

        bottom_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom_split.addWidget(self.log)
        bottom_split.addWidget(self.viewer_tabs)
        bottom_split.setStretchFactor(0, 2)
        bottom_split.setStretchFactor(1, 3)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_split.addWidget(scroll)
        main_split.addWidget(bottom_split)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 2)
        main_split.setSizes([360, 520])

        layout.addWidget(main_split)

        # Connections
        self.btn_run_smd.clicked.connect(self.run_smd)
        self.btn_run_spy.clicked.connect(self.run_spydr)
        self.btn_vis.clicked.connect(self.run_vis)
        self.btn_all.clicked.connect(self.run_all)
        self.btn_open_3d.clicked.connect(self.open_3d)
        self.btn_refresh_metrics.clicked.connect(self.refresh_metrics_tab)
        self.btn_copy_metrics.clicked.connect(self.copy_metrics_to_clipboard)
        self.mode.currentTextChanged.connect(self._update_button_labels)
        self._update_button_labels()

    def _load_logo(self):
        for name in (os.getenv("LOGO_PATH"), "luminous_logo.png", "logo.png", "luminous.png"):
            if not name:
                continue
            p = ROOT / name
            if p.exists():
                try:
                    pix = QtGui.QPixmap(str(p))
                    if not pix.isNull() and pix.width() > LOGO_MAX_W:
                        pix = pix.scaledToWidth(LOGO_MAX_W, QtCore.Qt.SmoothTransformation)
                    return pix
                except Exception:
                    continue
        return None

    def _make_env_base(self):
        env = os.environ.copy()
        # Prefer local venv python if present, ensure Homebrew paths, and unbuffered output for logs
        venv_py = ROOT / "venv" / "bin" / "python3"
        if not venv_py.exists():
            venv_py = ROOT / "venv" / "bin" / "python"
        if venv_py.exists():
            env["PY"] = str(venv_py)
            env["PATH"] = f"{venv_py.parent}:{env.get('PATH','')}"
        else:
            env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{env.get('PATH','')}"
        env["PYTHONUNBUFFERED"] = "1"
        env["LENGTH_FT"] = self.length.text().strip() or "24"
        env["WIDTH_FT"] = self.width.text().strip() or "24"
        # Ensure scripts can compute PPF consistently (PPF = mean_PPFD * CANOPY_AREA_M2).
        # We treat the canopy footprint as the full room footprint shown in the heatmap.
        try:
            L_ft = float(env["LENGTH_FT"])
            W_ft = float(env["WIDTH_FT"])
            if self.align_x.isChecked() and W_ft > L_ft:
                L_ft, W_ft = W_ft, L_ft
            area_m2 = (L_ft * W_ft) * 0.09290304
            if area_m2 > 0:
                env["CANOPY_AREA_M2"] = f"{area_m2:.6f}"
        except Exception:
            pass
        env["MODE"] = self.sim_mode.currentText()
        env["OS"] = self.os.text().strip() or "4"
        env["SUBPATCH_GRID"] = self.subgrid.text().strip() or "1"
        env["MOUNT_Z_M"] = self.mount_z.text().strip() or "0.4572"
        return env

    def _update_button_labels(self):
        m = self.mode.currentText()
        if m == "Competitor":
            self.btn_run_smd.setText("Run Uniformity")
            self.btn_all.setText("Run + Visualize")
            self.smd_perim_fill.setDisabled(True)
            self.smd_outer_optics.setDisabled(True)
            self.smd_outer_fwhm.setDisabled(True)
            self.smd_outer_per_module.setDisabled(True)
            self.smd_base_ring.setDisabled(True)
            self.cob_base_ring.setDisabled(True)
            self.cob_wall_clear_in.setDisabled(True)
        elif m == "COB":
            self.btn_run_smd.setText("Run Uniformity (COB)")
            self.btn_all.setText("Run + Visualize (COB)")
            # COB solver variables are W/COB per ring (physically capped).
            if (self.wmin.text().strip() or "10") == "10":
                self.wmin.setText("0")
            if (self.wmax.text().strip() or "100") == "100":
                self.wmax.setText("100")
            self.smd_perim_fill.setDisabled(True)
            self.smd_outer_optics.setDisabled(True)
            self.smd_outer_fwhm.setDisabled(True)
            self.smd_outer_per_module.setDisabled(True)
            self.smd_base_ring.setDisabled(True)
            self.cob_base_ring.setDisabled(False)
            self.cob_wall_clear_in.setDisabled(False)
        elif m == "Quantum Board":
            self.btn_run_smd.setText("Run Uniformity (Quantum)")
            self.btn_all.setText("Run + Visualize (Quantum)")
            self.smd_perim_fill.setDisabled(False)
            self.smd_outer_optics.setDisabled(True)
            self.smd_outer_fwhm.setDisabled(True)
            self.smd_outer_per_module.setDisabled(True)
            self.smd_base_ring.setDisabled(False)
            self.cob_base_ring.setDisabled(True)
            self.cob_wall_clear_in.setDisabled(True)
        else:
            self.btn_run_smd.setText("Run Uniformity (SMD)")
            self.btn_all.setText("Run + Visualize (SMD)")
            self.smd_perim_fill.setDisabled(False)
            self.smd_outer_optics.setDisabled(False)
            self.smd_outer_fwhm.setDisabled(False)
            self.smd_outer_per_module.setDisabled(False)
            self.smd_base_ring.setDisabled(False)
            self.cob_base_ring.setDisabled(True)
            self.cob_wall_clear_in.setDisabled(True)

    def _env_smd(self):
        env = self._make_env_base()
        env["LAYOUT_MODE"] = self.layout_mode.currentText()
        env["ALIGN_LONG_AXIS_X"] = "1" if self.align_x.isChecked() else "0"
        env["PPE_IS_SYSTEM"] = os.getenv("PPE_IS_SYSTEM", "0")
        env["TARGET_PPFD"] = self.target.text().strip() or "1000"
        env["RUN_BASIS"] = "1" if self.run_basis.isChecked() else "0"
        env["W_MIN"] = self.wmin.text().strip() or "10"
        env["W_MAX"] = self.wmax.text().strip() or "100"
        env["LAMBDA_S"] = self.lams.text().strip()
        env["LAMBDA_R"] = self.lamr.text().strip()
        env["LAMBDA_MEAN"] = self.lammean.text().strip() or "10"
        env["USE_CHEBYSHEV"] = "1" if self.cheby.isChecked() else "0"
        if self.smd_outer_optics.isChecked():
            env["OPTICS"] = "lens"
        else:
            env["OPTICS"] = self.optics.currentText()
        env["SMD_PERIM_GAP_FILL"] = "1" if self.smd_perim_fill.isChecked() else "0"
        env["SMD_OUTER_PER_MODULE"] = "1" if self.smd_outer_per_module.isChecked() else "0"
        env["SMD_BASE_RING_N"] = self.smd_base_ring.text().strip() or "7"
        env["SMD_OUTER_OPTICS"] = "1" if self.smd_outer_optics.isChecked() else "0"
        env["SMD_OUTER_FWHM_DEG"] = self.smd_outer_fwhm.text().strip() or "40"
        return env

    def _env_spydr(self):
        env = self._make_env_base()
        # Auto-compute grid so fixtures fill area with minimal margin.
        # SPYDR footprint: 47" along X (bar length), ~43" along Y (bars span).
        try:
            L_ft = float(self.length.text())
            W_ft = float(self.width.text())
        except Exception:
            L_ft = W_ft = 0.0
        align = self.align_x.isChecked()
        if align and W_ft > L_ft:
            L_ft, W_ft = W_ft, L_ft  # long axis to X
        try:
            margin_in = float(self.sp_margin.text().strip() or "0")
        except Exception:
            margin_in = 0.0
        try:
            edge_inset_in = float(self.sp_edge_inset.text().strip() or "0")
        except Exception:
            edge_inset_in = 0.0
        usable_x_in = max(0.0, L_ft * 12.0 - 2 * margin_in)
        usable_y_in = max(0.0, W_ft * 12.0 - 2 * margin_in)
        nx = max(1, int(usable_x_in // 47.0))
        ny = max(1, int(usable_y_in // 43.0))
        env["NX"] = str(nx)
        env["NY"] = str(ny)
        env["SPYDR_PPF"] = self.sp_ppf.text().strip() or "2200"
        env["SPYDR_Z_M"] = self.sp_z.text().strip() or "0.4572"
        env["EFF_SCALE"] = self.sp_eff.text().strip() or "1.0"
        env["SPYDR_PPE_UMOL_PER_J"] = self.sp_ppe.text().strip() or "2.76"
        env["TARGET_PPFD"] = self.target.text().strip() or "1000"
        env["MARGIN_IN"] = str(margin_in)
        env["SPYDR_EDGE_INSET_IN"] = str(edge_inset_in)
        env["ALIGN_LONG_AXIS_X"] = "1" if align else "0"
        return env

    def _env_quantum(self):
        # Same solver knobs as SMD, routed to the quantum pipeline.
        env = self._make_env_base()
        env["LAYOUT_MODE"] = self.layout_mode.currentText()
        env["ALIGN_LONG_AXIS_X"] = "1" if self.align_x.isChecked() else "0"
        env["PPE_IS_SYSTEM"] = "1"
        env["TARGET_PPFD"] = self.target.text().strip() or "1000"
        env["RUN_BASIS"] = "1" if self.run_basis.isChecked() else "0"
        env["W_MIN"] = self.wmin.text().strip() or "10"
        env["W_MAX"] = self.wmax.text().strip() or "120"
        env["LAMBDA_S"] = self.lams.text().strip()
        env["LAMBDA_R"] = self.lamr.text().strip()
        env["LAMBDA_MEAN"] = self.lammean.text().strip() or "10"
        env["USE_CHEBYSHEV"] = "1" if self.cheby.isChecked() else "0"
        env["SUBPATCH_GRID"] = self.subgrid.text().strip() or "1"
        env["QB_EDGE_PERIM"] = "1" if self.qb_perim.isChecked() else "0"
        env["QB_PERIM_INSET_M"] = "0.0" if self.qb_perim.isChecked() else "0.02"
        env["SMD_PERIM_GAP_FILL"] = "1" if self.smd_perim_fill.isChecked() else "0"
        env["SMD_BASE_RING_N"] = self.smd_base_ring.text().strip() or "7"
        return env

    def _env_cob(self):
        # Same solver knobs as SMD, routed to the COB pipeline.
        env = self._make_env_base()
        env["LAYOUT_MODE"] = self.layout_mode.currentText()
        env["ALIGN_LONG_AXIS_X"] = "1" if self.align_x.isChecked() else "0"
        env["TARGET_PPFD"] = self.target.text().strip() or "1000"
        env["RUN_BASIS"] = "1" if self.run_basis.isChecked() else "0"
        env["W_MIN"] = self.wmin.text().strip() or "10"
        env["W_MAX"] = self.wmax.text().strip() or "100"
        env["COB_BASE_RING_N"] = self.cob_base_ring.text().strip() or "6"
        env["COB_WALL_CLEARANCE_IN"] = self.cob_wall_clear_in.text().strip() or "3.0"
        env["COB_STRIP_MIN_RING"] = "0"
        env["COB_OPTICS"] = os.getenv("COB_OPTICS", "none") # set to: outer, none
        env["COB_OPTICS_RINGS"] = os.getenv("COB_OPTICS_RINGS", "")
        env["COB_OPTICS_KIND"] = os.getenv("COB_OPTICS_KIND", "sym")
        env["COB_OPTICS_FWHM_DEG"] = os.getenv("COB_OPTICS_FWHM_DEG", "60.0")
        env["LAMBDA_S"] = self.lams.text().strip()
        env["LAMBDA_R"] = self.lamr.text().strip()
        env["LAMBDA_MEAN"] = self.lammean.text().strip() or "10"
        env["USE_CHEBYSHEV"] = "1" if self.cheby.isChecked() else "0"
        return env

    def _disable_buttons(self, disable=True):
        for b in (self.btn_run_smd, self.btn_run_spy, self.btn_vis, self.btn_all):
            b.setDisabled(disable)

    def _ppfd_map_path(self) -> Path:
        return ROOT / "ppfd_map.txt"

    def _first_float_in_line(self, line: str) -> float | None:
        parts = line.replace("≈", " ").replace("W", " ").replace("µmol/J", " ").split()
        for tok in parts:
            try:
                return float(tok)
            except Exception:
                continue
        return None

    def _read_smd_summary_details(self) -> dict[str, float]:
        summ = self._summary_path_for_mode()
        if not summ or not summ.exists():
            return {}
        try:
            txt = summ.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}
        out: dict[str, float] = {}
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith("total electrical input"):
                val = self._first_float_in_line(s)
                if val is not None:
                    out["watts_elec"] = val
            elif s.startswith("total effective"):
                val = self._first_float_in_line(s)
                if val is not None:
                    out["watts_effective"] = val
            elif s.startswith("avg PPE (mix, base)"):
                val = self._first_float_in_line(s)
                if val is not None:
                    out["ppe_base"] = val
            elif s.startswith("thermal tier:") and "avg" in s:
                val = self._first_float_in_line(s[s.find("avg"):])
                if val is not None:
                    out["thermal_eff_avg"] = val
        return out

    def _read_spydr_power_meta(self) -> dict[str, float]:
        meta_path = ROOT / "ies_sources" / "spydr3_power.txt"
        if not meta_path.exists():
            return {}
        out: dict[str, float] = {}
        try:
            txt = meta_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}
        for ln in txt.splitlines():
            if "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k = k.strip()
            try:
                out[k] = float(v.strip())
            except Exception:
                continue
        return out

    def _estimate_total_input_watts(self) -> float | None:
        mode = self.mode.currentText()
        if mode == "Competitor":
            meta = self._read_spydr_power_meta()
            if "total_w" in meta and float(meta["total_w"]) > 0:
                return float(meta["total_w"])
            try:
                ppe = float(self.sp_ppe.text().strip() or "0")
            except Exception:
                return None
            if "ppe_effective" in meta and float(meta["ppe_effective"]) > 0:
                ppe = float(meta["ppe_effective"])
            if not (ppe > 0):
                return None
            emitted_ppf = self._estimate_total_emitted_ppf()
            if emitted_ppf is None:
                return None
            return emitted_ppf / ppe

        details = self._read_smd_summary_details()
        if "watts_elec" in details and float(details["watts_elec"]) > 0:
            return float(details["watts_elec"])
        if "watts_effective" in details and float(details["watts_effective"]) > 0:
            return float(details["watts_effective"])
        return None

    def _estimate_total_effective_watts(self) -> float | None:
        details = self._read_smd_summary_details()
        if "watts_effective" in details and float(details["watts_effective"]) > 0:
            return float(details["watts_effective"])
        return None

    def _estimate_total_emitted_ppf(self) -> float | None:
        mode = self.mode.currentText()
        if mode == "Competitor":
            sp = ROOT / "ies_sources" / "spydr3_summary.txt"
            if sp.exists():
                try:
                    txt = sp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    txt = ""
                fixtures = None
                ppf_fixture = None
                eff_scale = None
                for ln in txt.splitlines():
                    s = ln.strip()
                    if s.startswith("fixtures="):
                        try:
                            fixtures = int(s.split("fixtures=", 1)[1].split()[0])
                        except Exception:
                            fixtures = None
                    if "PPF/fixture=" in s and "derate(EFF_SCALE)=" in s:
                        try:
                            ppf_fixture = float(s.split("PPF/fixture=", 1)[1].split()[0])
                        except Exception:
                            ppf_fixture = None
                        try:
                            eff_scale = float(s.split("derate(EFF_SCALE)=", 1)[1].split()[0])
                        except Exception:
                            eff_scale = None
                if fixtures and ppf_fixture and eff_scale is not None:
                    total_ppf = float(ppf_fixture) * int(fixtures) * float(eff_scale)
                    return total_ppf if total_ppf > 0 else None

            try:
                sp_ppf = float(self.sp_ppf.text().strip() or "0")
                eff_scale = float(self.sp_eff.text().strip() or "1.0")
            except Exception:
                return None
            try:
                env_spy = self._env_spydr()
                nx = int(env_spy.get("NX", "1"))
                ny = int(env_spy.get("NY", "1"))
            except Exception:
                nx = ny = 1
            fixtures = max(1, nx * ny)
            total_ppf = sp_ppf * fixtures * eff_scale
            return total_ppf if total_ppf > 0 else None

        summ = self._summary_path_for_mode()
        if summ and summ.exists():
            try:
                txt = summ.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None
            for ln in txt.splitlines():
                s = ln.strip()
                if "total photons" in s:
                    parts = s.replace("≈", " ").replace("µmol/s", " ").replace("umol/s", " ").split()
                    for tok in parts:
                        try:
                            v = float(tok)
                            return v if v > 0 else None
                        except ValueError:
                            continue
            return None

    def _format_efficiency_report(
        self,
        *,
        mode: str,
        cap_scale: float | None,
        ppf_out: float | None,
        ppf_at_cap: float | None,
        watts_elec: float | None,
        watts_effective: float | None,
        emitted_ppf: float | None,
        details: dict[str, float] | None = None,
    ) -> list[str]:
        lines: list[str] = []
        details = details or {}
        if cap_scale is None:
            cap_scale = 1.0
        if ppf_at_cap is None and ppf_out is not None:
            ppf_at_cap = ppf_out * cap_scale

        watts_elec_at_cap = (watts_elec * cap_scale) if (watts_elec and cap_scale is not None) else None
        watts_eff_at_cap = (watts_effective * cap_scale) if (watts_effective and cap_scale is not None) else None

        ppe_system_elec = None
        ppf_for_ppe = emitted_ppf if emitted_ppf else ppf_out
        if ppf_for_ppe and watts_elec and watts_elec > 0:
            ppe_system_elec = ppf_for_ppe / watts_elec

        deuc_elec = None
        if ppf_at_cap and watts_elec_at_cap and watts_elec_at_cap > 0:
            deuc_elec = ppf_at_cap / watts_elec_at_cap

        deuc_eff = None
        if ppf_at_cap and watts_eff_at_cap and watts_eff_at_cap > 0:
            deuc_eff = ppf_at_cap / watts_eff_at_cap

        if os.getenv("DEBUG_EFF", "0") == "1" and cap_scale is not None:
            if ppe_system_elec and deuc_elec and cap_scale <= 1.0:
                if deuc_elec > ppe_system_elec + 1e-6:
                    raise AssertionError("DEUC_elec exceeds PPE_system_elec; check denominators.")

        if cap_scale is not None:
            cap_line = f"  cap_scale={cap_scale:.3f}"
            if ppf_out is not None:
                cap_line += f"  ppf_out={ppf_out:.1f} umol/s"
            if ppf_at_cap is not None:
                cap_line += f"  ppf_at_cap={ppf_at_cap:.1f} umol/s"
            lines.append(cap_line)

        if watts_elec is not None:
            w_line = f"  watts_elec={watts_elec:.1f} W"
            if watts_elec_at_cap is not None:
                w_line += f"  watts_elec@cap={watts_elec_at_cap:.1f} W"
            lines.append(w_line)

        if watts_effective is not None:
            w_eff_line = f"  watts_effective={watts_effective:.1f} W"
            if watts_eff_at_cap is not None:
                w_eff_line += f"  watts_effective@cap={watts_eff_at_cap:.1f} W"
            lines.append(w_eff_line)

        ppe_base = details.get("ppe_base")
        if ppe_base is not None:
            lines.append(f"  PPE_base={ppe_base:.3f} umol/J")

        if ppe_system_elec is not None or deuc_elec is not None:
            ppe_line = ""
            if ppe_system_elec is not None:
                ppe_line += f"  PPE_system_elec={ppe_system_elec:.3f} umol/J"
            if deuc_elec is not None:
                ppe_line += f"  DEUC_elec={deuc_elec:.3f} umol/J"
            if ppe_line:
                lines.append(ppe_line)

        if deuc_eff is not None:
            lines.append(f"  DEUC_eff={deuc_eff:.3f} umol/J")

        if mode == "SMD":
            thermal_avg = details.get("thermal_eff_avg")
            if thermal_avg is not None:
                lines.append(f"  thermal: tiered avg={thermal_avg:.3f}")

        return lines

    def _summary_path_for_mode(self) -> Path | None:
        m = self.mode.currentText()
        if m == "Quantum Board":
            return ROOT / "ies_sources" / "quantum_summary.txt"
        if m == "COB":
            return ROOT / "ies_sources" / "cob_summary.txt"
        if m == "Competitor":
            return None
        return ROOT / "ies_sources" / "smd_summary.txt"

    def _metrics_text_block(self) -> str:
        lines: list[str] = []
        mode = self.mode.currentText()
        lines.append(f"Mode: {mode}")

        # Room + target
        area_m2 = None
        try:
            length_ft = float(self.length.text())
            width_ft = float(self.width.text())
        except Exception:
            length_ft = width_ft = 0.0
        align = self.align_x.isChecked()
        if align and width_ft > length_ft:
            length_ft, width_ft = width_ft, length_ft
        area_m2 = (length_ft * width_ft) * 0.09290304 if (length_ft > 0 and width_ft > 0) else None
        if area_m2:
            lines.append(f"Room: {length_ft:.2f} ft x {width_ft:.2f} ft  (area={area_m2:.3f} m^2)")
        else:
            lines.append("Room: (invalid dimensions)")

        try:
            cap = float(self.target.text().strip())
        except Exception:
            cap = None
        lines.append(f"Cap/setpoint PPFD: {cap:.2f} µmol/m²/s" if cap else "Cap/setpoint PPFD: (not set)")
        if cap:
            lines.append(f"Coverage thresholds (after peak-capping): 90%={0.90*cap:.2f}, 95%={0.95*cap:.2f} µmol/m²/s")

        # PPFD metrics from ppfd_map.txt
        lines.append("")
        lines.append("PPFD metrics (from ppfd_map.txt):")
        m_result = None
        if compute_ppfd_metrics is None or format_ppfd_metrics_line is None or np is None:
            lines.append("  (metrics helpers unavailable; install numpy)")
        else:
            pmap = self._ppfd_map_path()
            if not pmap.exists():
                lines.append("  (ppfd_map.txt not found)")
            else:
                try:
                    raw = pmap.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception as e:
                    raw = []
                    lines.append(f"  (failed reading ppfd_map.txt) {e}")

                vals = []
                for line in raw:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    s = s.replace(",", " ")
                    parts = s.split()
                    if not parts:
                        continue
                    try:
                        vals.append(float(parts[-1]))
                    except Exception:
                        continue
                if not vals:
                    lines.append("  (no parseable values found)")
                else:
                    watts = self._estimate_total_input_watts()
                    emitted_ppf = self._estimate_total_emitted_ppf()
                    m = compute_ppfd_metrics(
                        np.asarray(vals, dtype=float),
                        setpoint_ppfd=cap,
                        canopy_area_m2=area_m2,
                        total_input_watts=watts,
                        emitted_ppf_umol_s=emitted_ppf,
                        legacy_metrics=True,
                    )
                    m_result = m
                    try:
                        if cap and float(cap) > 0:
                            pmax = float(m.get("max", 0.0))
                            cap_scale = float(m.get("cap_scale", 1.0))
                            over_ppfd = pmax - float(cap)
                            if cap_scale < 0.999999:
                                lines.append(f"  cap status: binding (cap_scale={cap_scale:.6f}, max-cap={over_ppfd:+.2f})")
                            else:
                                lines.append(f"  cap status: not binding (cap_scale={cap_scale:.6f}, max-cap={over_ppfd:+.2f})")

                            if mode == "Competitor":
                                try:
                                    eff_scale = float(self.sp_eff.text().strip() or "1.0")
                                except Exception:
                                    eff_scale = 1.0
                                if eff_scale >= 0.999 and pmax < float(cap) - 1e-3:
                                    lines.append("  note: cap is unattainable at full output (increase fixtures or PPF/fixture)")
                    except Exception:
                        pass
                    lines.append(f"  {format_ppfd_metrics_line(m)}")

        # Electrical / efficiency
        lines.append("")
        lines.append("Electrical / efficiency:")
        cap_scale = None
        ppf_out = None
        ppf_at_cap = None
        if isinstance(m_result, dict):
            try:
                cap_scale = float(m_result.get("cap_scale", 1.0))
            except Exception:
                cap_scale = None
            try:
                ppf_out = float(m_result.get("ppf_out")) if "ppf_out" in m_result else None
            except Exception:
                ppf_out = None
            try:
                ppf_at_cap = float(m_result.get("ppf_at_cap")) if "ppf_at_cap" in m_result else None
            except Exception:
                ppf_at_cap = None
        if mode == "Competitor":
            try:
                sp_ppf = float(self.sp_ppf.text().strip() or "0")
                ppe = float(self.sp_ppe.text().strip() or "0")
            except Exception:
                sp_ppf = ppe = 0.0
            meta = self._read_spydr_power_meta()
            try:
                env_spy = self._env_spydr()
                nx = int(env_spy.get("NX", "1"))
                ny = int(env_spy.get("NY", "1"))
            except Exception:
                nx = ny = 1
            fixtures = max(1, nx * ny)
            eff_scale_used = None
            total_ppf_used = self._estimate_total_emitted_ppf()
            sp_sum = ROOT / "ies_sources" / "spydr3_summary.txt"
            if sp_sum.exists():
                try:
                    txt = sp_sum.read_text(encoding="utf-8", errors="ignore").rstrip()
                    for ln in txt.splitlines():
                        if "derate(EFF_SCALE)=" in ln:
                            try:
                                eff_scale_used = float(ln.split("derate(EFF_SCALE)=", 1)[1].split()[0])
                            except Exception:
                                eff_scale_used = None
                except Exception:
                    pass
            if eff_scale_used is None:
                try:
                    eff_scale_used = float(self.sp_eff.text().strip() or "1.0")
                except Exception:
                    eff_scale_used = 1.0
            if total_ppf_used is None:
                total_ppf_used = sp_ppf * fixtures * eff_scale_used

            lines.append(f"  fixtures={fixtures} (NX={nx}, NY={ny})  PPF/fixture={sp_ppf:.2f} µmol/s  dimmer={eff_scale_used:.4f}")
            lines.append(f"  total PPF ≈ {float(total_ppf_used):.2f} µmol/s")
            ppe_eff = float(meta.get("ppe_effective", 0.0)) if meta else 0.0
            total_w = float(meta.get("total_w", 0.0)) if meta else 0.0
            watts_elec = float(total_w) if total_w > 0 else None
            if watts_elec is None and ppe_eff > 0:
                watts_elec = float(total_ppf_used) / ppe_eff
            elif watts_elec is None and ppe > 0:
                watts_elec = float(total_ppf_used) / ppe
            details = {"ppe_base": ppe}
            emitted_ppf = self._estimate_total_emitted_ppf()
            lines.extend(self._format_efficiency_report(
                mode=mode,
                cap_scale=cap_scale,
                ppf_out=ppf_out,
                ppf_at_cap=ppf_at_cap,
                watts_elec=watts_elec,
                watts_effective=None,
                emitted_ppf=emitted_ppf,
                details=details,
            ))
        else:
            if mode == "COB":
                report = ROOT / "ies_sources" / "cob_solver_report.txt"
                if report.exists():
                    try:
                        txt = report.read_text(encoding="utf-8", errors="ignore").rstrip()
                        for ln in txt.splitlines():
                            lines.append(f"  {ln}")
                        lines.append("")
                    except Exception:
                        pass
            details = self._read_smd_summary_details()
            watts_elec = details.get("watts_elec")
            watts_effective = details.get("watts_effective")
            emitted_ppf = self._estimate_total_emitted_ppf()
            lines.extend(self._format_efficiency_report(
                mode=mode,
                cap_scale=cap_scale,
                ppf_out=ppf_out,
                ppf_at_cap=ppf_at_cap,
                watts_elec=watts_elec,
                watts_effective=watts_effective,
                emitted_ppf=emitted_ppf,
                details=details,
            ))
            summ = self._summary_path_for_mode()
            if summ and summ.exists():
                try:
                    txt = summ.read_text(encoding="utf-8", errors="ignore").rstrip()
                except Exception as e:
                    txt = f"(failed to read {summ}) {e}"
                for ln in txt.splitlines():
                    lines.append(f"  {ln}")
            else:
                lines.append("  (summary file not found yet; run a simulation first)")

        return "\n".join(lines).rstrip() + "\n"

    def refresh_metrics_tab(self) -> None:
        try:
            self.metrics_text.setPlainText(self._metrics_text_block())
        except Exception as e:
            self.metrics_text.setPlainText(f"(metrics render failed) {e}\n")

    def copy_metrics_to_clipboard(self) -> None:
        try:
            QtWidgets.QApplication.clipboard().setText(self.metrics_text.toPlainText())
        except Exception:
            pass

    def _append_cap_metrics_from_map(self, *, label: str = "CAP_METRICS") -> None:
        """Append cap-based PPFD metrics for the latest ppfd_map.txt to the log."""
        if compute_ppfd_metrics is None or format_ppfd_metrics_line is None or np is None:
            return
        pmap = self._ppfd_map_path()
        if not pmap.exists():
            return
        try:
            raw = pmap.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        vals = []
        for line in raw:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            s = s.replace(',', ' ')
            parts = s.split()
            if not parts:
                continue
            try:
                vals.append(float(parts[-1]))
            except Exception:
                continue
        if not vals:
            return

        # cap/setpoint: the user-entered Target PPFD is treated as the canopy ceiling for the cap-math.
        cap = None
        try:
            cap = float(self.target.text().strip())
        except Exception:
            cap = None

        # optional area for PPF reporting (full room footprint)
        area_m2 = None
        try:
            L_ft = float(self.length.text())
            W_ft = float(self.width.text())
            if self.align_x.isChecked() and W_ft > L_ft:
                L_ft, W_ft = W_ft, L_ft
            area_m2 = (L_ft * W_ft) * 0.09290304
        except Exception:
            area_m2 = None

        try:
            watts = self._estimate_total_input_watts()
            emitted_ppf = self._estimate_total_emitted_ppf()
            m = compute_ppfd_metrics(
                np.asarray(vals, dtype=float),
                setpoint_ppfd=cap,
                canopy_area_m2=area_m2,
                total_input_watts=watts,
                emitted_ppf_umol_s=emitted_ppf,
                legacy_metrics=True,
            )
            self.log.append_line(f"{label}: {format_ppfd_metrics_line(m)}")
        except Exception as e:
            self.log.append_line(f"{label}: (metrics failed) {e}")
        self.refresh_metrics_tab()


    def _run_cmd_async(self, cmd, env, on_finish=None, *, compute_metrics=False):
        self._disable_buttons(True)
        self.log.append_line(f"$ {' '.join(cmd)}")
        proc = QtCore.QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(ROOT))
        penv = QtCore.QProcessEnvironment()
        for k, v in env.items():
            penv.insert(k, v)
        proc.setProcessEnvironment(penv)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        def handle_ready():
            data = proc.readAllStandardOutput().data().decode(errors="ignore")
            for line in data.splitlines():
                if line.strip():
                    self.log.append_line(line)

        def handle_finished(code, _status):
            handle_ready()
            self.log.append_line(f"[exit {code}]")
            if proc in self._procs:
                self._procs.remove(proc)
            self._disable_buttons(False)
            if on_finish:
                on_finish(code)
            if compute_metrics and code == 0:
                self._append_cap_metrics_from_map()
            proc.deleteLater()

        proc.readyReadStandardOutput.connect(handle_ready)
        proc.finished.connect(handle_finished)
        self._procs.append(proc)
        proc.start()

    def run_smd(self):
        mode = self.mode.currentText()
        if mode == "Quantum Board":
            env = self._env_quantum()
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity_quantum.sh")], env, compute_metrics=True)
        elif mode == "COB":
            env = self._env_cob()
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity_cob.sh")], env, compute_metrics=True)
        else:
            env = self._env_smd()
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity.sh")], env, compute_metrics=True)

    def run_spydr(self):
        env = self._env_spydr()
        self._run_cmd_async(["bash", str(ROOT/"run_simulation_spydr3.sh")], env, compute_metrics=True)

    def run_vis(self):
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
        overlay = self.overlay.currentText()
        m = self.mode.currentText()
        if m == "Competitor" and overlay == "auto":
            overlay = "spydr3"
        if m == "Quantum Board" and overlay == "auto":
            overlay = "quantum"
        if m == "COB" and overlay == "auto":
            overlay = "cob"
        if m == "Competitor":
            outdir = "ppfd_visualizations_spydr3"
        elif m == "Quantum Board":
            outdir = "ppfd_visualizations_quantum"
        elif m == "COB":
            outdir = "ppfd_visualizations_cob"
        else:
            outdir = "ppfd_visualizations"
        cmd = ["python3", str(ROOT/"visualize_ppfd.py"), "--overlay", overlay, "--outdir", outdir]
        self._run_cmd_async(cmd, env, on_finish=lambda rc: (self.load_visuals(outdir), self.refresh_metrics_tab()))

    def run_all(self):
        # run appropriate pipeline then visualize
        m = self.mode.currentText()
        if m == "Competitor":
            self._run_cmd_async(["bash", str(ROOT/"run_simulation_spydr3.sh")], self._env_spydr(),
                                on_finish=lambda rc: self.run_vis(), compute_metrics=True)
        elif m == "Quantum Board":
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity_quantum.sh")], self._env_quantum(),
                                on_finish=lambda rc: self.run_vis(), compute_metrics=True)
        elif m == "COB":
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity_cob.sh")], self._env_cob(),
                                on_finish=lambda rc: self.run_vis(), compute_metrics=True)
        else:
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity.sh")], self._env_smd(),
                                on_finish=lambda rc: self.run_vis(), compute_metrics=True)

    def load_visuals(self, outdir: str):
        out = Path(outdir)
        ov = out / "ppfd_heatmap_overlay.png"
        ann = out / "ppfd_heatmap_annotated.png"
        surf = out / "ppfd_surface_3d.png"
        if ov.exists():
            pix = QtGui.QPixmap(str(ov))
            self.img_overlay.setPixmap(pix.scaled(self.img_overlay.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        if ann.exists():
            pix = QtGui.QPixmap(str(ann))
            self.img_annot.setPixmap(pix.scaled(self.img_annot.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        if surf.exists():
            pix = QtGui.QPixmap(str(surf))
            self.img_surface.setPixmap(pix.scaled(self.img_surface.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # rescale visuals on resize
        for lbl, fname in ((self.img_overlay, "ppfd_heatmap_overlay.png"),
                           (self.img_annot, "ppfd_heatmap_annotated.png"),
                           (self.img_surface, "ppfd_surface_3d.png")):
            pm = lbl.pixmap()
            if pm:
                lbl.setPixmap(pm.scaled(lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def open_3d(self):
        # Open 3D scatter in default browser
        m = self.mode.currentText()
        if m == "Competitor":
            outdir = "ppfd_visualizations_spydr3"
        elif m == "Quantum Board":
            outdir = "ppfd_visualizations_quantum"
        elif m == "COB":
            outdir = "ppfd_visualizations_cob"
        else:
            outdir = "ppfd_visualizations"
        html = Path(outdir) / "ppfd_scatter_3d.html"
        if html.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(html.resolve())))
        else:
            self.log.append_line(f"3D scatter not found at {html}")


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
