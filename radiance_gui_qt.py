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
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["SMD", "Quantum Board", "Competitor"])
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
        self.target = QtWidgets.QLineEdit("800")
        room.addWidget(QtWidgets.QLabel("Length (ft)"), 0, 0); room.addWidget(self.length, 0, 1)
        room.addWidget(QtWidgets.QLabel("Width (ft)"), 0, 2);  room.addWidget(self.width, 0, 3)
        room.addWidget(QtWidgets.QLabel("Target PPFD"), 0, 4); room.addWidget(self.target, 0, 5)
        controls_layout.addWidget(room_box)

        # --- Collapsible panels in a row ---
        row_boxes = QtWidgets.QHBoxLayout()
        row_boxes.setSpacing(10)

        solver_box = CollapsibleBox("Solver", collapsed=True)
        sol = QtWidgets.QGridLayout()
        self.wmin = QtWidgets.QLineEdit("10"); self.wmax = QtWidgets.QLineEdit("120")
        self.lams = QtWidgets.QLineEdit("0 1e-3 1e-2 1e-1 1.0 10.0")
        self.lamr = QtWidgets.QLineEdit("0 1e-3 1e-2 1e-1")
        self.lammean = QtWidgets.QLineEdit("10")
        self.cheby = QtWidgets.QCheckBox("Use Chebyshev")
        sol.addWidget(QtWidgets.QLabel("W min"), 0, 0); sol.addWidget(self.wmin, 0, 1)
        sol.addWidget(QtWidgets.QLabel("W max"), 0, 2); sol.addWidget(self.wmax, 0, 3)
        sol.addWidget(QtWidgets.QLabel("lambda_s"), 1, 0); sol.addWidget(self.lams, 1, 1, 1, 3)
        sol.addWidget(QtWidgets.QLabel("lambda_r"), 2, 0); sol.addWidget(self.lamr, 2, 1, 1, 3)
        sol.addWidget(QtWidgets.QLabel("lambda_mean"), 3, 0); sol.addWidget(self.lammean, 3, 1)
        sol.addWidget(self.cheby, 3, 2, 1, 2)
        solver_box.layout().addLayout(sol)
        row_boxes.addWidget(solver_box, 1)

        sim_box = CollapsibleBox("Simulation", collapsed=True)
        sim = QtWidgets.QGridLayout()
        self.sim_mode = QtWidgets.QComboBox(); self.sim_mode.addItems(["instant","fast","quality","direct"])
        self.os = QtWidgets.QLineEdit("4")
        self.subgrid = QtWidgets.QLineEdit("1")
        self.optics = QtWidgets.QComboBox(); self.optics.addItems(["none","lens"])
        self.overlay = QtWidgets.QComboBox(); self.overlay.addItems(["auto","smd","spydr3","quantum","both","none"])
        self.qb_perim = QtWidgets.QCheckBox("Edge-to-edge perimeter (Quantum)")
        self.qb_perim.setChecked(False)
        sim.addWidget(QtWidgets.QLabel("Sim mode"), 0, 0); sim.addWidget(self.sim_mode, 0, 1)
        sim.addWidget(QtWidgets.QLabel("Oversample (OS)"), 0, 2); sim.addWidget(self.os, 0, 3)
        sim.addWidget(QtWidgets.QLabel("Subpatch grid"), 1, 0); sim.addWidget(self.subgrid, 1, 1)
        sim.addWidget(QtWidgets.QLabel("Optics"), 1, 2); sim.addWidget(self.optics, 1, 3)
        sim.addWidget(QtWidgets.QLabel("Overlay"), 2, 0); sim.addWidget(self.overlay, 2, 1)
        sim.addWidget(self.qb_perim, 3, 0, 1, 4)
        sim_box.layout().addLayout(sim)
        row_boxes.addWidget(sim_box, 1)

        sp_box = CollapsibleBox("Competitor", collapsed=True)
        sp = QtWidgets.QGridLayout()
        self.sp_ppf = QtWidgets.QLineEdit("2200")
        self.sp_z = QtWidgets.QLineEdit("0.4572")
        self.sp_eff = QtWidgets.QLineEdit("1.0")
        self.sp_ppe = QtWidgets.QLineEdit("2.76")
        sp.addWidget(QtWidgets.QLabel("PPF per fixture (µmol/s)"), 0, 0); sp.addWidget(self.sp_ppf, 0, 1)
        sp.addWidget(QtWidgets.QLabel("Mount Z (m)"), 0, 2); sp.addWidget(self.sp_z, 0, 3)
        sp.addWidget(QtWidgets.QLabel("Eff scale"), 1, 0); sp.addWidget(self.sp_eff, 1, 1)
        sp.addWidget(QtWidgets.QLabel("PPE (µmol/J)"), 1, 2); sp.addWidget(self.sp_ppe, 1, 3)
        sp.addWidget(QtWidgets.QLabel("Margin (in)"), 2, 2); sp.addWidget(QtWidgets.QLabel("5 (fixed)"), 2, 3)
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
        env["MODE"] = self.sim_mode.currentText()
        env["OS"] = self.os.text().strip() or "4"
        env["SUBPATCH_GRID"] = self.subgrid.text().strip() or "1"
        return env

    def _update_button_labels(self):
        m = self.mode.currentText()
        if m == "Competitor":
            self.btn_run_smd.setText("Run Uniformity")
            self.btn_all.setText("Run + Visualize")
        elif m == "Quantum Board":
            self.btn_run_smd.setText("Run Uniformity (Quantum)")
            self.btn_all.setText("Run + Visualize (Quantum)")
        else:
            self.btn_run_smd.setText("Run Uniformity (SMD)")
            self.btn_all.setText("Run + Visualize (SMD)")

    def _env_smd(self):
        env = self._make_env_base()
        env["LAYOUT_MODE"] = self.layout_mode.currentText()
        env["ALIGN_LONG_AXIS_X"] = "1" if self.align_x.isChecked() else "0"
        env["TARGET_PPFD"] = self.target.text().strip() or "800"
        env["RUN_BASIS"] = "1" if self.run_basis.isChecked() else "0"
        env["W_MIN"] = self.wmin.text().strip() or "10"
        env["W_MAX"] = self.wmax.text().strip() or "120"
        env["LAMBDA_S"] = self.lams.text().strip()
        env["LAMBDA_R"] = self.lamr.text().strip()
        env["LAMBDA_MEAN"] = self.lammean.text().strip() or "10"
        env["USE_CHEBYSHEV"] = "1" if self.cheby.isChecked() else "0"
        env["OPTICS"] = self.optics.currentText()
        return env

    def _env_spydr(self):
        env = self._make_env_base()
        # Auto-compute grid so fixtures fill area with 5" inset margin.
        # SPYDR footprint: 47" along X (bar length), ~43" along Y (bars span).
        try:
            L_ft = float(self.length.text())
            W_ft = float(self.width.text())
        except Exception:
            L_ft = W_ft = 0.0
        align = self.align_x.isChecked()
        if align and W_ft > L_ft:
            L_ft, W_ft = W_ft, L_ft  # long axis to X
        margin_in = 5.0
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
        env["TARGET_PPFD"] = self.target.text().strip() or "800"
        env["MARGIN_IN"] = "0"
        env["ALIGN_LONG_AXIS_X"] = "1" if align else "0"
        return env

    def _env_quantum(self):
        # Same solver knobs as SMD, routed to the quantum pipeline.
        env = self._make_env_base()
        env["LAYOUT_MODE"] = self.layout_mode.currentText()
        env["ALIGN_LONG_AXIS_X"] = "1" if self.align_x.isChecked() else "0"
        env["TARGET_PPFD"] = self.target.text().strip() or "800"
        env["RUN_BASIS"] = "1" if self.run_basis.isChecked() else "0"
        env["W_MIN"] = self.wmin.text().strip() or "10"
        env["W_MAX"] = self.wmax.text().strip() or "120"
        env["LAMBDA_S"] = self.lams.text().strip()
        env["LAMBDA_R"] = self.lamr.text().strip()
        env["LAMBDA_MEAN"] = self.lammean.text().strip() or "10"
        env["USE_CHEBYSHEV"] = "1" if self.cheby.isChecked() else "0"
        env["SUBPATCH_GRID"] = self.subgrid.text().strip() or "1"
        env["QB_EDGE_PERIM"] = "1" if self.qb_perim.isChecked() else "0"
        return env

    def _disable_buttons(self, disable=True):
        for b in (self.btn_run_smd, self.btn_run_spy, self.btn_vis, self.btn_all):
            b.setDisabled(disable)

    def _run_cmd_async(self, cmd, env, on_finish=None):
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
            proc.deleteLater()

        proc.readyReadStandardOutput.connect(handle_ready)
        proc.finished.connect(handle_finished)
        self._procs.append(proc)
        proc.start()

    def run_smd(self):
        mode = self.mode.currentText()
        if mode == "Quantum Board":
            env = self._env_quantum()
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity_quantum.sh")], env)
        else:
            env = self._env_smd()
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity.sh")], env)

    def run_spydr(self):
        env = self._env_spydr()
        self._run_cmd_async(["bash", str(ROOT/"run_simulation_spydr3.sh")], env)

    def run_vis(self):
        env = os.environ.copy()
        overlay = self.overlay.currentText()
        m = self.mode.currentText()
        if m == "Competitor" and overlay == "auto":
            overlay = "spydr3"
        if m == "Quantum Board" and overlay == "auto":
            overlay = "quantum"
        if m == "Competitor":
            outdir = "ppfd_visualizations_spydr3"
        elif m == "Quantum Board":
            outdir = "ppfd_visualizations_quantum"
        else:
            outdir = "ppfd_visualizations"
        cmd = ["python3", str(ROOT/"visualize_ppfd.py"), "--overlay", overlay, "--outdir", outdir]
        self._run_cmd_async(cmd, env, on_finish=lambda rc: self.load_visuals(outdir))

    def run_all(self):
        # run appropriate pipeline then visualize
        m = self.mode.currentText()
        if m == "Competitor":
            self._run_cmd_async(["bash", str(ROOT/"run_simulation_spydr3.sh")], self._env_spydr(),
                                on_finish=lambda rc: self.run_vis())
        elif m == "Quantum Board":
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity_quantum.sh")], self._env_quantum(),
                                on_finish=lambda rc: self.run_vis())
        else:
            self._run_cmd_async(["bash", str(ROOT/"run_uniformity.sh")], self._env_smd(),
                                on_finish=lambda rc: self.run_vis())

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
        outdir = "ppfd_visualizations" if self.mode.currentText() == "SMD" else "ppfd_visualizations_spydr3"
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
