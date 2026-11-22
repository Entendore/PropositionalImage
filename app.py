#!/usr/bin/env python3
"""
Propositional Logic Image Generator - Ultimate Performance Edition
==================================================================
Features:
  - Numba JIT (CPU) & CuPy (GPU) acceleration for massive speedups.
  - Fused kernels for predicate generation and HSV->RGB conversion.
  - Granular Propositional Predicates & Advanced WFC.
  - Interactive Per-Pixel Truth Table & Step-by-Step Logic Trace.
"""

import sys
import numpy as np
import random

# ==============================================================================
# PERFORMANCE BACKENDS (Numba & CuPy)
# ==============================================================================

USE_NUMBA = False
USE_CUPY = False
cp = None
numba = None

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    pass

try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    pass

# --- Numba CPU Kernels ---
if USE_NUMBA:
    @njit(parallel=True, cache=True)
    def compute_predicates_cpu(Hg, rb_lo, rb_hi, bp_h1, bp_h2, bp_pcts, out_rainbow, out_between):
        H, W = Hg.shape
        n_rb = rb_lo.shape[0]
        n_bp = bp_h1.shape[0]
        for y in prange(H):
            for x in range(W):
                h = Hg[y, x]
                for i in range(n_rb):
                    if h >= rb_lo[i] and h <= rb_hi[i]:
                        out_rainbow[i, y, x] = True
                for i in range(n_bp):
                    h1 = bp_h1[i]; h2 = bp_h2[i]
                    arc = (h2 - h1) % 360.0
                    if arc == 0.0: arc = 360.0
                    diff = (h - h1) % 360.0
                    pct = (diff / arc) * 100.0
                    if pct <= bp_pcts[i] or diff <= 0.0001:
                        out_between[i, y, x] = True

    @njit(parallel=True, cache=True)
    def hsv_to_rgb_cpu(Hg, Sg, Vg, out_rgb):
        H, W = Hg.shape
        for y in prange(H):
            for x in range(W):
                h = Hg[y, x]; s = Sg[y, x] / 255.0; v = Vg[y, x] / 255.0
                c = v * s; hp = h / 60.0
                x_val = c * (1.0 - abs((hp % 2.0) - 1.0)); m = v - c
                r = g = b = 0.0
                if 0 <= hp < 1: r=c; g=x_val
                elif 1 <= hp < 2: r=x_val; g=c
                elif 2 <= hp < 3: g=c; b=x_val
                elif 3 <= hp < 4: g=x_val; b=c
                elif 4 <= hp < 5: r=x_val; b=c
                elif 5 <= hp < 6: r=c; b=x_val
                out_rgb[y, x, 0] = int((r + m) * 255)
                out_rgb[y, x, 1] = int((g + m) * 255)
                out_rgb[y, x, 2] = int((b + m) * 255)

# --- CuPy GPU Kernels ---
if USE_CUPY:
    try:
        hsv_to_rgb_gpu = cp.ElementwiseKernel(
            'float32 h, float32 s, float32 v', 'uint8 r, uint8 g, uint8 b',
            '''
            float c = (v / 255.0f) * (s / 255.0f);
            float hp = h / 60.0f;
            float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
            float m = (v / 255.0f) - c;
            float r_f = 0, g_f = 0, b_f = 0;
            if (hp >= 0 && hp < 1) { r_f = c; g_f = x; }
            else if (hp >= 1 && hp < 2) { r_f = x; g_f = c; }
            else if (hp >= 2 && hp < 3) { g_f = c; b_f = x; }
            else if (hp >= 3 && hp < 4) { g_f = x; b_f = c; }
            else if (hp >= 4 && hp < 5) { r_f = x; b_f = c; }
            else if (hp >= 5 && hp < 6) { r_f = c; b_f = x; }
            r = (uint8)(((r_f + m) * 255.0f));
            g = (uint8)(((g_f + m) * 255.0f));
            b = (uint8)(((b_f + m) * 255.0f));
            ''', 'hsv_to_rgb_gpu'
        )
    except Exception:
        hsv_to_rgb_gpu = None

# ==============================================================================
# AST & LOGIC ENGINE
# ==============================================================================

class ASTNode: pass
class Var(ASTNode):
    def __init__(self, name): self.name = name
class Not(ASTNode):
    def __init__(self, child): self.child = child
class BinOp(ASTNode):
    def __init__(self, op, left, right): self.op = op; self.left = left; self.right = right

class LogicParser:
    def __init__(self, expr):
        self.tokens = self._tokenise(expr.upper())
        self.pos = 0
    @staticmethod
    def _tokenise(s):
        toks = []; i = 0
        while i < len(s):
            c = s[i]
            if c.isspace(): i += 1; continue
            if c in "()": toks.append(c); i += 1
            elif c == "&": toks.append("AND"); i += 1
            elif c == "|": toks.append("OR"); i += 1
            elif c == "^": toks.append("XOR"); i += 1
            elif c == "~": toks.append("NOT"); i += 1
            elif c == ">" and i+1 < len(s) and s[i+1] == ">": toks.append("IMPLIES"); i += 2
            elif c.isalnum() or c == '_':
                j = i
                while j < len(s) and (s[j].isalnum() or s[j] == '_'): j += 1
                toks.append(s[i:j]); i = j
            else: raise ValueError(f"Unexpected character: {c!r}")
        return toks
    def parse(self):
        if not self.tokens: raise ValueError("Empty expression")
        node = self._parse_or()
        if self.pos < len(self.tokens): raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        return node
    def _parse_or(self):
        left = self._parse_xor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "OR": self.pos += 1; left = BinOp("OR", left, self._parse_xor())
        return left
    def _parse_xor(self):
        left = self._parse_and()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "XOR": self.pos += 1; left = BinOp("XOR", left, self._parse_and())
        return left
    def _parse_and(self):
        left = self._parse_impl()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "AND": self.pos += 1; left = BinOp("AND", left, self._parse_impl())
        return left
    def _parse_impl(self):
        left = self._parse_not()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "IMPLIES": self.pos += 1; left = BinOp("IMPLIES", left, self._parse_not())
        return left
    def _parse_not(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == "NOT": self.pos += 1; return Not(self._parse_not())
        return self._parse_atom()
    def _parse_atom(self):
        if self.pos >= len(self.tokens): raise ValueError("Unexpected end of expression")
        t = self.tokens[self.pos]
        if t == "(":
            self.pos += 1; node = self._parse_or()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ")": raise ValueError("Missing ')'")
            self.pos += 1; return node
        if t in ("AND", "OR", "NOT", "XOR", "IMPLIES"): raise ValueError(f"Operator in wrong place: {t}")
        self.pos += 1; return Var(t)

def evaluate_ast(node, ns):
    if isinstance(node, Var): val = bool(ns.get(node.name, False)); return val, f"{node.name}={val}"
    if isinstance(node, Not): val, trace = evaluate_ast(node.child, ns); return not val, f"NOT({trace}) = {not val}"
    if isinstance(node, BinOp):
        lval, ltrace = evaluate_ast(node.left, ns); rval, rtrace = evaluate_ast(node.right, ns)
        if node.op == "AND": res = lval and rval
        elif node.op == "OR": res = lval or rval
        elif node.op == "XOR": res = lval ^ rval
        elif node.op == "IMPLIES": res = (not lval) or rval
        else: res = False
        return res, f"({ltrace}) {node.op} ({rtrace}) = {res}"
    return False, "?"


# ==============================================================================
# WAVE FUNCTION COLLAPSE CORE
# ==============================================================================

class WFCTile:
    def __init__(self, id, color_hsv): self.id = id; self.color_hsv = color_hsv

class WFCGrid:
    def __init__(self, width, height, tiles, adjacency_rules, weights):
        self.width = width; self.height = height; self.tiles = tiles
        self.rules = adjacency_rules; self.weights = weights
        self.grid = [[set(range(len(tiles))) for _ in range(width)] for _ in range(height)]
        self.collapsed = [[False for _ in range(width)] for _ in range(height)]
    def entropy(self, x, y):
        if self.collapsed[y][x]: return float('inf')
        return len(self.grid[y][x])
    def collapse(self):
        min_ent = float('inf'); candidates = []
        for y in range(self.height):
            for x in range(self.width):
                ent = self.entropy(x, y)
                if 0 < ent < min_ent: min_ent = ent; candidates = [(x, y)]
                elif ent == min_ent and ent != float('inf'): candidates.append((x, y))
        if not candidates: return False
        x, y = random.choice(candidates); possible = list(self.grid[y][x])
        weights = [self.weights[p] for p in possible]; total = sum(weights)
        chosen_tile = random.choices(possible, weights=[w/total for w in weights], k=1)[0] if total > 0 else random.choice(possible)
        self.grid[y][x] = {chosen_tile}; self.collapsed[y][x] = True
        return self.propagate(x, y)
    def propagate(self, start_x, start_y):
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            for dx, dy, dir_out, dir_in in [(0, -1, 'N', 'S'), (1, 0, 'E', 'W'), (0, 1, 'S', 'N'), (-1, 0, 'W', 'E')]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not self.collapsed[ny][nx]:
                    old_possibilities = self.grid[ny][nx].copy(); new_possibilities = set()
                    for tile_id in self.grid[y][x]: new_possibilities.update(self.rules.get((tile_id, dir_out), set()))
                    self.grid[ny][nx] = old_possibilities.intersection(new_possibilities)
                    if not self.grid[ny][nx]: return False
                    if self.grid[ny][nx] != old_possibilities: stack.append((nx, ny))
        return True
    def run(self):
        while True:
            if all(all(self.collapsed[y][x] for x in range(self.width)) for y in range(self.height)): return True
            if not self.collapse(): return False


# ==============================================================================
# UI WIDGETS
# ==============================================================================

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QTextEdit, QColorDialog, QFileDialog,
    QMessageBox, QScrollArea, QGridLayout, QTabWidget, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QAbstractItemView,
    QTextBrowser
)
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import Qt, Signal

class ClickableLabel(QLabel):
    clicked = Signal(object)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.clicked.emit(event.pos())
        super().mousePressEvent(event)

class BetweenPredicateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Add Between Predicate"); self.resize(400, 200)
        layout = QVBoxLayout(); form = QFormLayout()
        self.c1 = QComboBox(); self.c1.addItems(['R', 'O', 'Y', 'G', 'B', 'I', 'V'])
        self.c2 = QComboBox(); self.c2.addItems(['R', 'O', 'Y', 'G', 'B', 'I', 'V']); self.c2.setCurrentIndex(1)
        self.pct = QDoubleSpinBox(); self.pct.setRange(0, 100); self.pct.setValue(50); self.pct.setSuffix(" %")
        self.name = QTextEdit(); self.name.setMaximumHeight(30); self.name.setPlainText("R_O_50")
        form.addRow("From:", self.c1); form.addRow("To:", self.c2); form.addRow("Percentage:", self.pct); form.addRow("Name:", self.name)
        layout.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns); self.setLayout(layout)
        self.c1.currentTextChanged.connect(self._upd); self.c2.currentTextChanged.connect(self._upd); self.pct.valueChanged.connect(self._upd)
    def _upd(self): self.name.setPlainText(f"{self.c1.currentText()}_{self.c2.currentText()}_{int(self.pct.value())}")
    def get_values(self): return {'color1': self.c1.currentText(), 'color2': self.c2.currentText(), 'percentage': self.pct.value(), 'name': self.name.toPlainText().strip()}


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class PropLogicImageApp(QMainWindow):
    RAINBOW = {'R': ((0, 15), (345, 360)), 'O': ((15, 45),), 'Y': ((45, 75),), 'G': ((75, 165),), 'B': ((165, 255),), 'I': ((255, 285),), 'V': ((285, 345),)}
    RAINBOW_CENTERS = {'R': 0, 'O': 30, 'Y': 60, 'G': 120, 'B': 210, 'I': 270, 'V': 315}
    RAINBOW_ORDER = ['R', 'O', 'Y', 'G', 'B', 'I', 'V']
    COLOR_HEX = {'R': '#FF0000', 'O': '#FF7F00', 'Y': '#FFFF00', 'G': '#00FF00', 'B': '#0000FF', 'I': '#4B0082', 'V': '#9400D3'}
    DEFAULT_BETWEEN = [
        ('R', 'O', 30, 'R_O_30'), ('R', 'O', 50, 'R_O_50'), ('R', 'O', 70, 'R_O_70'),
        ('O', 'Y', 30, 'O_Y_30'), ('O', 'Y', 50, 'O_Y_50'), ('O', 'Y', 70, 'O_Y_70'),
        ('Y', 'G', 30, 'Y_G_30'), ('Y', 'G', 50, 'Y_G_50'), ('Y', 'G', 70, 'Y_G_70'),
        ('G', 'B', 30, 'G_B_30'), ('G', 'B', 50, 'G_B_50'), ('G', 'B', 70, 'G_B_70'),
        ('B', 'I', 30, 'B_I_30'), ('B', 'I', 50, 'B_I_50'), ('B', 'I', 70, 'B_I_70'),
        ('I', 'V', 30, 'I_V_30'), ('I', 'V', 50, 'I_V_50'), ('I', 'V', 70, 'I_V_70'),
        ('V', 'R', 30, 'V_R_30'), ('V', 'R', 50, 'V_R_50'), ('V', 'R', 70, 'V_R_70'),
    ]
    EXAMPLES = [
        ("Rainbow Red", "R"), ("Warm colors", "R OR O OR Y"), ("Cool colors", "G OR B OR I OR V"),
        ("Not green", "NOT G"), ("Red XOR Blue", "R XOR B"), ("If red then green", "R IMPLIES G"),
        ("30% between R-O", "R_O_30"), ("Complex", "(R OR O) AND NOT (B AND V)"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Propositional Logic & WFC - Ultimate Performance Edition")
        self.resize(1600, 1000)
        self.true_color = QColor(255, 255, 255); self.false_color = QColor(0, 0, 0)
        self.current_image = None; self.predicates = {}; self.h_grid = None
        self.between_predicates = list(self.DEFAULT_BETWEEN); self.wfc_enabled = False
        self.selected_pixel = None; self.ast_root = None
        self.backend = "Auto" # Auto, Numba, NumPy, CuPy
        self._build_ui()
        self._warmup_backends()

    def _get_color_hex(self, color): return self.COLOR_HEX.get(color, '#FFFFFF')

    def _warmup_backends(self):
        """Pre-compile Numba kernels to prevent UI freeze on first generation."""
        if USE_NUMBA:
            self.statusBar().showMessage("Warming up Numba JIT kernels (one-time)...")
            QApplication.processEvents()
            dummy_h = np.zeros((10, 10), dtype=np.float32)
            dummy_out_rb = np.zeros((7, 10, 10), dtype=np.bool_)
            dummy_out_bp = np.zeros((21, 10, 10), dtype=np.bool_)
            dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            compute_predicates_cpu(dummy_h, np.zeros(7), np.zeros(7), np.zeros(21), np.zeros(21), np.zeros(21), dummy_out_rb, dummy_out_bp)
            hsv_to_rgb_cpu(dummy_h, dummy_h, dummy_h, dummy_rgb)
            self.statusBar().showMessage("Ready")

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central); root = QHBoxLayout(central)
        left = QVBoxLayout()

        size_box = QGroupBox("Image Size"); sf = QFormLayout()
        self.w_spin = QSpinBox(); self.w_spin.setRange(10, 4000); self.w_spin.setValue(1000)
        self.h_spin = QSpinBox(); self.h_spin.setRange(10, 4000); self.h_spin.setValue(1000)
        sf.addRow("Width:", self.w_spin); sf.addRow("Height:", self.h_spin); size_box.setLayout(sf); left.addWidget(size_box)

        # Backend Selector
        backend_box = QGroupBox("Compute Backend"); bf = QFormLayout()
        self.backend_combo = QComboBox()
        options = ["Auto (Best Available)"]
        if USE_CUPY: options.append("GPU (CuPy)")
        if USE_NUMBA: options.append("CPU (Numba JIT)")
        options.append("CPU (NumPy)")
        self.backend_combo.addItems(options)
        self.backend_combo.currentTextChanged.connect(lambda t: setattr(self, 'backend', t.split('(')[0].strip().replace('Auto (Best Available)', 'Auto')))
        bf.addRow("Backend:", self.backend_combo)
        
        status_lbl = QLabel()
        status_lbl.setText(f"CuPy: {'✅' if USE_CUPY else '❌'} | Numba: {'✅' if USE_NUMBA else '❌'}")
        bf.addRow("Status:", status_lbl)
        backend_box.setLayout(bf); left.addWidget(backend_box)

        # WFC Section
        wfc_box = QGroupBox("Wave Function Collapse (WFC)"); wfc_layout = QVBoxLayout()
        self.wfc_check = QCheckBox("Enable WFC Mode"); self.wfc_check.toggled.connect(self._toggle_wfc); wfc_layout.addWidget(self.wfc_check)
        self.wfc_controls = QWidget(); self.wfc_controls.setEnabled(False)
        preset_layout = QHBoxLayout(); preset_layout.addWidget(QLabel("Preset:"))
        self.wfc_preset_combo = QComboBox()
        self.wfc_preset_combo.addItems(["Smooth Rainbow", "Sharp Blocks", "Random Noise", "Checkerboard", "Complementary", "Analogous", "Custom Matrix"])
        self.wfc_preset_combo.currentTextChanged.connect(self._apply_wfc_preset); preset_layout.addWidget(self.wfc_preset_combo); wfc_layout.addLayout(preset_layout)
        
        matrix_group = QGroupBox("Adjacency Truth Table"); matrix_layout = QVBoxLayout()
        tool_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Clear"); self.btn_clear.clicked.connect(lambda: self._set_matrix(False))
        self.btn_fill = QPushButton("Fill"); self.btn_fill.clicked.connect(lambda: self._set_matrix(True))
        self.btn_diag = QPushButton("Diagonal"); self.btn_diag.clicked.connect(self._set_matrix_diagonal)
        self.btn_inv = QPushButton("Invert"); self.btn_inv.clicked.connect(self._invert_matrix)
        tool_layout.addWidget(self.btn_clear); tool_layout.addWidget(self.btn_fill); tool_layout.addWidget(self.btn_diag); tool_layout.addWidget(self.btn_inv)
        matrix_layout.addLayout(tool_layout)
        
        header_widget = QWidget(); header_layout = QHBoxLayout(); header_layout.setContentsMargins(0,0,0,0); header_layout.addWidget(QLabel(""), 1)
        for color in self.RAINBOW_ORDER:
            lbl = QLabel(color); lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet(f"font-weight:bold; color:{self._get_color_hex(color)}"); header_layout.addWidget(lbl, 1)
        header_widget.setLayout(header_layout); matrix_layout.addWidget(header_widget)
        
        self.matrix_checkboxes = {}
        for row_color in self.RAINBOW_ORDER:
            row_widget = QWidget(); row_layout = QHBoxLayout(); row_layout.setContentsMargins(0,0,0,0)
            row_lbl = QLabel(row_color); row_lbl.setAlignment(Qt.AlignCenter); row_lbl.setStyleSheet(f"font-weight:bold; color:{self._get_color_hex(row_color)}"); row_layout.addWidget(row_lbl, 1)
            self.matrix_checkboxes[row_color] = {}
            for col_color in self.RAINBOW_ORDER:
                cb = QCheckBox(); cb.setChecked(True); row_layout.addWidget(cb, 1); self.matrix_checkboxes[row_color][col_color] = cb
            row_widget.setLayout(row_layout); matrix_layout.addWidget(row_widget)
        matrix_group.setLayout(matrix_layout); wfc_layout.addWidget(matrix_group)
        
        weights_group = QGroupBox("Tile Weights"); weights_layout = QFormLayout()
        self.tile_weights = {}
        for color in self.RAINBOW_ORDER:
            spin = QSpinBox(); spin.setRange(1, 100); spin.setValue(10); self.tile_weights[color] = spin; weights_layout.addRow(f"{color}:", spin)
        weights_group.setLayout(weights_layout); wfc_layout.addWidget(weights_group)
        
        size_layout = QFormLayout(); self.wfc_size_spin = QSpinBox(); self.wfc_size_spin.setRange(8, 128); self.wfc_size_spin.setValue(32)
        size_layout.addRow("Grid Size:", self.wfc_size_spin); wfc_layout.addLayout(size_layout)
        self.wfc_controls.setLayout(wfc_layout); wfc_layout.addWidget(self.wfc_controls); wfc_box.setLayout(wfc_layout); left.addWidget(wfc_box)

        # HSV & Predicates
        map_box = QGroupBox("HSV Mapping"); mf = QFormLayout()
        self.h_combo = QComboBox(); self.h_combo.addItems(["X (horizontal)", "Y (vertical)", "X+Y diagonal"])
        self.s_combo = QComboBox(); self.s_combo.addItems(["Y (vertical)", "X (horizontal)", "Constant 255"])
        self.v_combo = QComboBox(); self.v_combo.addItems(["Constant 255", "X (horizontal)", "Y (vertical)"])
        mf.addRow("Hue:", self.h_combo); mf.addRow("Sat:", self.s_combo); mf.addRow("Val:", self.v_combo); map_box.setLayout(mf); left.addWidget(map_box)

        between_box = QGroupBox("Between Predicates"); between_layout = QVBoxLayout()
        self.between_list = QListWidget(); self.between_list.setSelectionMode(QAbstractItemView.ExtendedSelection); self._refresh_between_list(); between_layout.addWidget(self.between_list)
        btn_layout = QHBoxLayout()
        self.add_between_btn = QPushButton("Add"); self.add_between_btn.clicked.connect(self._add_between)
        self.remove_between_btn = QPushButton("Remove"); self.remove_between_btn.clicked.connect(self._remove_between)
        self.reset_between_btn = QPushButton("Reset"); self.reset_between_btn.clicked.connect(self._reset_between)
        btn_layout.addWidget(self.add_between_btn); btn_layout.addWidget(self.remove_between_btn); btn_layout.addWidget(self.reset_between_btn)
        between_layout.addLayout(btn_layout); between_box.setLayout(between_layout); left.addWidget(between_box)

        pred_box = QGroupBox("Predicate"); pred_layout = QVBoxLayout()
        self.pred_combo = QComboBox(); self.pred_combo.addItem("Custom Expression"); self._refresh_pred_combo()
        self.pred_combo.currentIndexChanged.connect(self._on_pred_change); pred_layout.addWidget(self.pred_combo); pred_box.setLayout(pred_layout); left.addWidget(pred_box)

        expr_box = QGroupBox("Logical Expression"); ef = QVBoxLayout()
        ef.addWidget(QLabel("Vars: R O Y G B I V, R_O_30 etc. | Ops: AND OR NOT XOR IMPLIES ( )"))
        self.expr_edit = QTextEdit(); self.expr_edit.setPlainText("(R OR Y) AND NOT B"); self.expr_edit.setMaximumHeight(70); ef.addWidget(self.expr_edit)
        ex_row = QHBoxLayout(); ex_row.addWidget(QLabel("Examples:")); self.ex_combo = QComboBox()
        for name, _ in self.EXAMPLES: self.ex_combo.addItem(name)
        self.ex_combo.currentIndexChanged.connect(self._load_example); ex_row.addWidget(self.ex_combo); ef.addLayout(ex_row); expr_box.setLayout(ef); left.addWidget(expr_box)

        out_box = QGroupBox("Output Colors"); of = QFormLayout()
        self.t_btn = QPushButton("True"); self.t_btn.clicked.connect(lambda: self._pick(1))
        self.f_btn = QPushButton("False"); self.f_btn.clicked.connect(lambda: self._pick(2))
        of.addRow("TRUE:", self.t_btn); of.addRow("FALSE:", self.f_btn)
        self.use_hsv = QCheckBox("Use pixel HSV when TRUE"); self.use_hsv.setChecked(True); of.addRow(self.use_hsv); out_box.setLayout(of); left.addWidget(out_box)

        self.gen_btn = QPushButton("▶ Generate"); self.gen_btn.setStyleSheet("QPushButton{padding:10px;font-weight:bold;background:#2a2;color:#fff}"); self.gen_btn.clicked.connect(self.generate); left.addWidget(self.gen_btn)
        self.save_btn = QPushButton("💾 Save"); self.save_btn.clicked.connect(self.save); self.save_btn.setEnabled(False); left.addWidget(self.save_btn)
        left.addStretch()

        # Right Side
        right = QVBoxLayout(); tabs = QTabWidget()
        preview_widget = QWidget(); preview_layout = QVBoxLayout(preview_widget)
        self.preview = ClickableLabel("Press 'Generate'"); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(700, 700)
        self.preview.setStyleSheet("QLabel{border:1px solid #888;background:#111;color:#aaa}"); self.preview.setMouseTracking(True)
        self.preview.mouseMoveEvent = self._on_mouse_move; self.preview.clicked.connect(self._on_pixel_click)
        preview_layout.addWidget(self.preview, stretch=1); tabs.addTab(preview_widget, "Preview")
        
        self.grid_scroll = QScrollArea(); self.grid_widget = QWidget(); self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_scroll.setWidget(self.grid_widget); self.grid_scroll.setWidgetResizable(True); tabs.addTab(self.grid_scroll, "All Predicates")
        right.addWidget(tabs, stretch=3)

        insp_box = QGroupBox("Pixel Inspector"); insp_layout = QVBoxLayout()
        self.insp_label = QLabel("Click a pixel"); self.insp_label.setWordWrap(True); self.insp_label.setStyleSheet("QLabel{font-family:monospace;font-size:10px}")
        insp_layout.addWidget(self.insp_label); insp_box.setLayout(insp_layout); right.addWidget(insp_box, stretch=1)

        tt_box = QGroupBox("Truth Table & Logic Trace"); tt_layout = QVBoxLayout()
        self.tt_view = QTextBrowser(); self.tt_view.setMaximumHeight(300); self.tt_view.setStyleSheet("QTextBrowser{background:#111; color:#fff; font-family:monospace;}")
        tt_layout.addWidget(self.tt_view)
        self.trace_view = QTextBrowser(); self.trace_view.setMaximumHeight(150); self.trace_view.setStyleSheet("QTextBrowser{background:#222; color:#0f0; font-family:monospace;font-size:12px}")
        self.trace_view.setHtml("<p style='color:#aaa'>Click a pixel to see step-by-step evaluation...</p>")
        tt_layout.addWidget(self.trace_view)
        tt_box.setLayout(tt_layout); right.addWidget(tt_box, stretch=1)

        root.addLayout(left, stretch=1); root.addLayout(right, stretch=2); self.statusBar().showMessage("Ready")

    # ==============================================================================
    # MATRIX TOOLS & UI HELPERS
    # ==============================================================================
    def _set_matrix(self, val):
        for r in self.RAINBOW_ORDER:
            for c in self.RAINBOW_ORDER: self.matrix_checkboxes[r][c].setChecked(val)
    def _set_matrix_diagonal(self):
        self._set_matrix(False)
        for r in self.RAINBOW_ORDER: self.matrix_checkboxes[r][r].setChecked(True)
    def _invert_matrix(self):
        for r in self.RAINBOW_ORDER:
            for c in self.RAINBOW_ORDER: self.matrix_checkboxes[r][c].setChecked(not self.matrix_checkboxes[r][c].isChecked())

    def _on_pixel_click(self, pos):
        if self.current_image is None or self.h_grid is None: return
        pix = self.preview.pixmap()
        if pix is None: return
        scale_x = self.h_grid.shape[1] / pix.width(); scale_y = self.h_grid.shape[0] / pix.height()
        x = int(pos.x() * scale_x); y = int(pos.y() * scale_y)
        if 0 <= x < self.h_grid.shape[1] and 0 <= y < self.h_grid.shape[0]:
            self.selected_pixel = (x, y); self._update_inspector(x, y); self._update_trace(x, y)
            expr = self.expr_edit.toPlainText().strip() if self.pred_combo.currentIndex() == 0 else self.pred_combo.currentText()
            if expr: self._render_truth_table(expr)

    def _update_trace(self, x, y):
        if self.ast_root is None: return
        # Convert to numpy bool for trace evaluation if it's cupy
        ns = {}
        for v in self.predicates:
            val = self.predicates[v][y, x]
            ns[v] = bool(val.get() if USE_CUPY and self.backend == "CuPy" else val)
        try:
            res, trace = evaluate_ast(self.ast_root, ns)
            self.trace_view.setHtml(f"<b>Step-by-Step Evaluation for Pixel ({x},{y}):</b><br><br>{trace.replace('=', '=<b>').replace('False', '</b>False').replace('True', '</b>True')}<br><br><b>Final Result: {res}</b>")
        except Exception as e:
            self.trace_view.setHtml(f"<span style='color:red'>Error: {e}</span>")

    def _update_inspector(self, x, y):
        hue = self.h_grid[y, x]; sat = self.s_grid[y, x]; val = self.v_grid[y, x]
        text = f"<b>Pixel ({x}, {y})</b><br>HSV: H={hue:.1f}° S={sat:.0f} V={val:.0f}<br><br><b>Rainbow:</b><br>"
        for name in ['R', 'O', 'Y', 'G', 'B', 'I', 'V']: 
            v = self.predicates[name][y, x]
            text += f"  {name}: <span style='color:{'#0f0' if v else '#f55'}'>{bool(v)}</span><br>"
        text += "<br><b>Between:</b><br>"
        for _, _, _, name in self.between_predicates: 
            v = self.predicates[name][y, x]
            text += f"  {name}: <span style='color:{'#0f0' if v else '#f55'}'>{bool(v)}</span><br>"
        self.insp_label.setText(text)

    def _toggle_wfc(self, enabled):
        self.wfc_enabled = enabled; self.wfc_controls.setEnabled(enabled)
        self.h_combo.setEnabled(not enabled); self.s_combo.setEnabled(not enabled); self.v_combo.setEnabled(not enabled)

    def _apply_wfc_preset(self, preset):
        self._set_matrix(False)
        if preset in ["Smooth Rainbow", "Analogous"]:
            for i, r in enumerate(self.RAINBOW_ORDER):
                for j, c in enumerate(self.RAINBOW_ORDER):
                    diff = abs(i - j)
                    if diff <= 1 or diff == len(self.RAINBOW_ORDER) - 1: self.matrix_checkboxes[r][c].setChecked(True)
        elif preset == "Sharp Blocks": self._set_matrix_diagonal()
        elif preset in ["Random Noise", "Custom Matrix"]: self._set_matrix(True)
        elif preset == "Checkerboard":
            for i, r in enumerate(self.RAINBOW_ORDER):
                for j, c in enumerate(self.RAINBOW_ORDER):
                    if (i % 2) != (j % 2): self.matrix_checkboxes[r][c].setChecked(True)
        elif preset == "Complementary":
            for i, r in enumerate(self.RAINBOW_ORDER):
                for j, c in enumerate(self.RAINBOW_ORDER):
                    diff = abs(i - j)
                    if diff == 3 or diff == 4 or r == c: self.matrix_checkboxes[r][c].setChecked(True)

    def _refresh_between_list(self):
        self.between_list.clear()
        for c1, c2, pct, name in self.between_predicates: self.between_list.addItem(f"{name} ({c1}→{c2} @ {pct}%)")

    def _refresh_pred_combo(self):
        current = self.pred_combo.currentText(); self.pred_combo.clear(); self.pred_combo.addItem("Custom Expression")
        self.pred_combo.addItems(['R', 'O', 'Y', 'G', 'B', 'I', 'V'])
        for _, _, _, name in self.between_predicates: self.pred_combo.addItem(name)
        idx = self.pred_combo.findText(current)
        if idx >= 0: self.pred_combo.setCurrentIndex(idx)

    def _add_between(self):
        dialog = BetweenPredicateDialog(self)
        if dialog.exec() == QDialog.Accepted:
            values = dialog.get_values()
            if not values['name']: QMessageBox.warning(self, "Error", "Name cannot be empty."); return
            if any(p[3] == values['name'] for p in self.between_predicates): QMessageBox.warning(self, "Error", "Already exists."); return
            self.between_predicates.append((values['color1'], values['color2'], values['percentage'], values['name']))
            self._refresh_between_list(); self._refresh_pred_combo()

    def _remove_between(self):
        selected = self.between_list.selectedItems()
        if not selected: QMessageBox.warning(self, "Error", "Select predicates."); return
        indices = sorted([self.between_list.row(item) for item in selected], reverse=True)
        for idx in indices: self.between_predicates.pop(idx)
        self._refresh_between_list(); self._refresh_pred_combo()

    def _reset_between(self):
        self.between_predicates = list(self.DEFAULT_BETWEEN); self._refresh_between_list(); self._refresh_pred_combo()

    def _on_pred_change(self, idx): self.expr_edit.setEnabled(idx == 0)

    def _pick(self, which):
        c = QColorDialog.getColor()
        if c.isValid():
            if which == 1: self.true_color = c
            elif which == 2: self.false_color = c
            self.t_btn.setStyleSheet(f"background:{self.true_color.name()}"); self.f_btn.setStyleSheet(f"background:{self.false_color.name()}")

    def _load_example(self, idx): _, expr = self.EXAMPLES[idx]; self.expr_edit.setPlainText(expr)

    # ==============================================================================
    # LOGIC & GENERATION
    # ==============================================================================

    def _build_hsv_grids(self, W, H):
        xs = np.linspace(0, 360, W, endpoint=False); ys = np.linspace(0, 255, H)
        X, Y = np.meshgrid(xs, ys)
        def axis(arr_x, arr_y, spec):
            if "X" in spec and "Y" not in spec: return arr_x
            if "Y" in spec and "X" not in spec: return arr_y
            if "X+Y" in spec: return (arr_x + arr_y) / 2
            return np.full_like(arr_x, 255.0)
        Hg = axis(X, Y, self.h_combo.currentText()); Sg = axis(X, Y, self.s_combo.currentText()); Vg = axis(X, Y, self.v_combo.currentText())
        return np.clip(Hg, 0, 360).astype(np.float32), np.clip(Sg, 0, 255).astype(np.float32), np.clip(Vg, 0, 255).astype(np.float32)

    def _compute_all_predicates(self, Hg, Sg, Vg, xp):
        preds = {}
        # Prepare band arrays
        rb_lo = []; rb_hi = []
        for name, bands in self.RAINBOW.items():
            for lo, hi in bands:
                rb_lo.append(lo); rb_hi.append(hi)
        rb_lo = xp.array(rb_lo, dtype=xp.float32); rb_hi = xp.array(rb_hi, dtype=xp.float32)
        
        bp_h1 = []; bp_h2 = []; bp_pcts = []
        for c1, c2, pct, name in self.between_predicates:
            bp_h1.append(self.RAINBOW_CENTERS[c1]); bp_h2.append(self.RAINBOW_CENTERS[c2]); bp_pcts.append(pct)
        bp_h1 = xp.array(bp_h1, dtype=xp.float32); bp_h2 = xp.array(bp_h2, dtype=xp.float32); bp_pcts = xp.array(bp_pcts, dtype=xp.float32)

        n_rb = len(rb_lo); n_bp = len(bp_h1)
        H, W = Hg.shape
        
        # Dispatch to backend
        if self.backend == "CuPy" and USE_CUPY:
            out_rainbow = cp.zeros((n_rb, H, W), dtype=cp.bool_)
            out_between = cp.zeros((n_bp, H, W), dtype=cp.bool_)
            # CuPy doesn't have prange, use standard vectorized or custom kernel. 
            # For simplicity and max speed, we use the Numba kernel on CPU if CuPy isn't fully set up for this specific shape, 
            # but let's use CuPy's native broadcasting for predicates if Numba isn't available for GPU.
            # Actually, Numba doesn't support CuPy arrays directly. We fallback to vectorized CuPy.
            for i in range(n_rb):
                out_rainbow[i] = (Hg >= rb_lo[i]) & (Hg <= rb_hi[i])
            for i in range(n_bp):
                arc = (bp_h2[i] - bp_h1[i]) % 360.0
                if arc == 0.0: arc = 360.0
                diff = (Hg - bp_h1[i]) % 360.0
                pct = (diff / arc) * 100.0
                out_between[i] = (pct <= bp_pcts[i]) | (diff <= 0.0001)
        elif self.backend == "Numba" and USE_NUMBA:
            out_rainbow = np.zeros((n_rb, H, W), dtype=np.bool_)
            out_between = np.zeros((n_bp, H, W), dtype=np.bool_)
            compute_predicates_cpu(Hg, rb_lo, rb_hi, bp_h1, bp_h2, bp_pcts, out_rainbow, out_between)
        else: # NumPy
            out_rainbow = np.zeros((n_rb, H, W), dtype=np.bool_)
            out_between = np.zeros((n_bp, H, W), dtype=np.bool_)
            for i in range(n_rb): out_rainbow[i] = (Hg >= rb_lo[i]) & (Hg <= rb_hi[i])
            for i in range(n_bp):
                arc = (bp_h2[i] - bp_h1[i]) % 360.0
                if arc == 0.0: arc = 360.0
                diff = (Hg - bp_h1[i]) % 360.0
                pct = (diff / arc) * 100.0
                out_between[i] = (pct <= bp_pcts[i]) | (diff <= 0.0001)

        # Map back to names (handling multi-band rainbow colors like Red)
        rb_idx = 0
        for name, bands in self.RAINBOW.items():
            if len(bands) == 1:
                preds[name] = out_rainbow[rb_idx]
                rb_idx += 1
            else:
                combined = out_rainbow[rb_idx]
                rb_idx += 1
                for _ in range(len(bands) - 1):
                    combined = combined | out_rainbow[rb_idx]
                    rb_idx += 1
                preds[name] = combined

        for i, (_, _, _, name) in enumerate(self.between_predicates):
            preds[name] = out_between[i]
            
        return preds

    def _build_wfc_tiles_and_rules(self):
        tiles = []; rules = {}; weights = []
        for i, color in enumerate(self.RAINBOW_ORDER):
            tiles.append(WFCTile(i, (self.RAINBOW_CENTERS[color], 255, 255))); weights.append(self.tile_weights[color].value())
        for r in self.RAINBOW_ORDER:
            for c in self.RAINBOW_ORDER:
                if self.matrix_checkboxes[r][c].isChecked():
                    r_idx = self.RAINBOW_ORDER.index(r); c_idx = self.RAINBOW_ORDER.index(c)
                    rules[(r_idx, 'E')] = rules.get((r_idx, 'E'), set()); rules[(r_idx, 'E')].add(c_idx)
                    rules[(c_idx, 'W')] = rules.get((c_idx, 'W'), set()); rules[(c_idx, 'W')].add(r_idx)
                    rules[(r_idx, 'S')] = rules.get((r_idx, 'S'), set()); rules[(r_idx, 'S')].add(c_idx)
                    rules[(c_idx, 'N')] = rules.get((c_idx, 'N'), set()); rules[(c_idx, 'N')].add(r_idx)
        return tiles, rules, weights

    def _run_wfc(self):
        size = self.wfc_size_spin.value(); tiles, rules, weights = self._build_wfc_tiles_and_rules()
        wfc = WFCGrid(size, size, tiles, rules, weights)
        for _ in range(10):
            if wfc.run():
                result = np.zeros((size, size), dtype=int)
                for y in range(size):
                    for x in range(size): result[y, x] = list(wfc.grid[y][x])[0]
                return result, tiles
        return None, tiles

    def generate(self):
        W, H = self.w_spin.value(), self.h_spin.value()
        self.statusBar().showMessage("Generating..."); QApplication.processEvents()
        self.selected_pixel = None; self.ast_root = None

        # Determine backend
        active_backend = self.backend
        if active_backend == "Auto":
            if USE_CUPY: active_backend = "CuPy"
            elif USE_NUMBA: active_backend = "Numba"
            else: active_backend = "NumPy"
        
        xp = cp if active_backend == "CuPy" else np

        if self.wfc_enabled:
            self.statusBar().showMessage("Running WFC..."); QApplication.processEvents()
            wfc_result, wfc_tiles = self._run_wfc()
            if wfc_result is None: QMessageBox.critical(self, "WFC Error", "Failed to converge."); self.statusBar().showMessage("WFC Failed"); return
            
            y_idx = (np.arange(H) * wfc_result.shape[0] // H).clip(max=wfc_result.shape[0]-1)
            x_idx = (np.arange(W) * wfc_result.shape[1] // W).clip(max=wfc_result.shape[1]-1)
            upsampled = wfc_result[np.ix_(y_idx, x_idx)]
            
            Hg = xp.zeros((H, W), dtype=xp.float32); Sg = xp.full((H, W), 255.0, dtype=xp.float32); Vg = xp.full((H, W), 255.0, dtype=xp.float32)
            for i, tile in enumerate(wfc_tiles):
                mask = (upsampled == i); Hg[mask] = tile.color_hsv[0]
            self.h_grid, self.s_grid, self.v_grid = Hg, Sg, Vg
        else:
            Hg, Sg, Vg = self._build_hsv_grids(W, H)
            if active_backend == "CuPy":
                Hg, Sg, Vg = cp.asarray(Hg), cp.asarray(Sg), cp.asarray(Vg)
            self.h_grid, self.s_grid, self.v_grid = Hg, Sg, Vg

        self.predicates = self._compute_all_predicates(Hg, Sg, Vg, xp)
        pred_idx = self.pred_combo.currentIndex()
        expr = self.expr_edit.toPlainText().strip() if pred_idx == 0 else self.pred_combo.currentText()
        if not expr: QMessageBox.warning(self, "Error", "Enter expression."); return
        
        try: self.ast_root = LogicParser(expr).parse()
        except Exception as e: QMessageBox.critical(self, "Parse Error", str(e)); return

        def eval_grid(node):
            if isinstance(node, Var): return self.predicates.get(node.name, xp.zeros((H, W), dtype=xp.bool_))
            if isinstance(node, Not): return ~eval_grid(node.child)
            if isinstance(node, BinOp):
                l = eval_grid(node.left); r = eval_grid(node.right)
                if node.op == "AND": return l & r
                if node.op == "OR": return l | r
                if node.op == "XOR": return l ^ r
                if node.op == "IMPLIES": return (~l) | r
            return xp.zeros((H, W), dtype=xp.bool_)
        
        mask = eval_grid(self.ast_root).astype(xp.bool_)
        self._render_truth_table(expr)

        # HSV to RGB
        rgb = xp.zeros((H, W, 3), dtype=xp.uint8)
        fc = xp.array([self.false_color.red(), self.false_color.green(), self.false_color.blue()], dtype=xp.uint8)
        rgb[:] = fc
        
        if self.use_hsv.isChecked():
            if active_backend == "CuPy" and USE_CUPY and hsv_to_rgb_gpu:
                r_ch = xp.zeros((H, W), dtype=xp.uint8)
                g_ch = xp.zeros((H, W), dtype=xp.uint8)
                b_ch = xp.zeros((H, W), dtype=xp.uint8)
                hsv_to_rgb_gpu(Hg[mask], Sg[mask], Vg[mask], r_ch, g_ch, b_ch)
                rgb[mask, 0] = r_ch; rgb[mask, 1] = g_ch; rgb[mask, 2] = b_ch
            elif active_backend == "Numba" and USE_NUMBA:
                # Numba requires numpy arrays
                Hg_np = cp.asnumpy(Hg) if USE_CUPY else Hg
                Sg_np = cp.asnumpy(Sg) if USE_CUPY else Sg
                Vg_np = cp.asnumpy(Vg) if USE_CUPY else Vg
                mask_np = cp.asnumpy(mask) if USE_CUPY else mask
                out_rgb = np.zeros((int(mask_np.sum()), 3), dtype=np.uint8)
                hsv_to_rgb_cpu(Hg_np[mask_np], Sg_np[mask_np], Vg_np[mask_np], out_rgb)
                rgb[mask] = xp.asarray(out_rgb)
            else:
                # NumPy fallback
                Hg_np = cp.asnumpy(Hg) if USE_CUPY else Hg
                Sg_np = cp.asnumpy(Sg) if USE_CUPY else Sg
                Vg_np = cp.asnumpy(Vg) if USE_CUPY else Vg
                mask_np = cp.asnumpy(mask) if USE_CUPY else mask
                # Vectorized NumPy HSV->RGB
                h = Hg_np[mask_np]; s = Sg_np[mask_np]/255.0; v = Vg_np[mask_np]/255.0
                c = v*s; hp = h/60.0; x = c*(1-np.abs((hp%2)-1)); m = v-c
                r=g=b=np.zeros_like(c)
                m0=(hp>=0)&(hp<1); r[m0]=c[m0]; g[m0]=x[m0]
                m1=(hp>=1)&(hp<2); r[m1]=x[m1]; g[m1]=c[m1]
                m2=(hp>=2)&(hp<3); g[m2]=c[m2]; b[m2]=x[m2]
                m3=(hp>=3)&(hp<4); g[m3]=x[m3]; b[m3]=c[m3]
                m4=(hp>=4)&(hp<5); r[m4]=x[m4]; b[m4]=c[m4]
                m5=(hp>=5)&(hp<6); r[m5]=c[m5]; b[m5]=x[m5]
                out_rgb = np.stack([((r+m)*255).clip(0,255), ((g+m)*255).clip(0,255), ((b+m)*255).clip(0,255)], axis=-1).astype(np.uint8)
                rgb[mask] = xp.asarray(out_rgb)
        else:
            tc = xp.array([self.true_color.red(), self.true_color.green(), self.true_color.blue()], dtype=xp.uint8)
            rgb[mask] = tc

        # Convert to NumPy for QImage
        rgb_np = cp.asnumpy(rgb) if USE_CUPY and active_backend == "CuPy" else rgb
        
        bytes_per_line = 3 * W; img = QImage(rgb_np.data, W, H, bytes_per_line, QImage.Format_RGB888).copy(); self.current_image = img
        pix = QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation); self.preview.setPixmap(pix); self.save_btn.setEnabled(True)
        true_count = int(mask.sum() if not USE_CUPY else mask.get().sum())
        self.statusBar().showMessage(f"Done [{active_backend}] — {W}×{H}, TRUE: {true_count}/{W*H} ({100*true_count/(W*H):.1f}%)")

    def _on_mouse_move(self, event):
        if self.current_image is None or self.h_grid is None: return
        pos = event.pos(); pix = self.preview.pixmap()
        if pix is None: return
        scale_x = self.h_grid.shape[1] / pix.width(); scale_y = self.h_grid.shape[0] / pix.height()
        x = int(pos.x() * scale_x); y = int(pos.y() * scale_y)
        if 0 <= x < self.h_grid.shape[1] and 0 <= y < self.h_grid.shape[0]:
            if self.selected_pixel is None: self._update_inspector(x, y)

    def _render_truth_table(self, expr: str):
        import re
        variables = sorted(set(re.findall(r'\b([A-Z][A-Z0-9_]*)\b', expr.upper())))
        all_preds = ['R', 'O', 'Y', 'G', 'B', 'I', 'V'] + [p[3] for p in self.between_predicates]
        variables = [v for v in variables if v in all_preds]
        if not variables: self.tt_view.setHtml("<p style='color:#aaa'>(no variables)</p>"); return
        if len(variables) > 8: variables = variables[:8]
        
        n = len(variables)
        html = "<table border='1' style='border-collapse: collapse; font-family: monospace; width: 100%; text-align: center;'>"
        html += "<tr style='background-color: #333; color: #fff;'>"
        for v in variables: html += f"<th style='padding: 4px;'>{v}</th>"
        html += "<th style='padding: 4px;'>Result</th></tr>"

        for i in range(1 << n):
            bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
            ns = {v: bool(bits[k]) for k, v in enumerate(variables)}
            try: r, _ = evaluate_ast(self.ast_root, ns); out = "T" if r else "F"
            except: out = "?"

            is_highlight = False
            if self.selected_pixel:
                px, py = self.selected_pixel
                try:
                    pixel_bits = [bool(self.predicates[v][py, px].get() if USE_CUPY and self.backend=="CuPy" else self.predicates[v][py, px]) for v in variables]
                    if pixel_bits == bits: is_highlight = True
                except: pass

            bg_color = "#00FF00" if is_highlight else ("#222" if i % 2 == 0 else "#111")
            text_color = "#000" if is_highlight else "#FFF"
            html += f"<tr style='background-color: {bg_color}; color: {text_color};'>"
            for b in bits: html += f"<td style='padding: 2px;'>{'T' if b else 'F'}</td>"
            html += f"<td style='padding: 2px;'>{out}</td></tr>"
        html += "</table>"
        if self.selected_pixel: html += f"<p style='color: #0f0; margin-top: 10px;'>✓ Highlighted row matches Pixel {self.selected_pixel}</p>"
        else: html += "<p style='color: #aaa; margin-top: 10px;'>Click a pixel to highlight its row.</p>"
        self.tt_view.setHtml(html)

    def save(self):
        if self.current_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "propositional_logic.png", "PNG (*.png)")
        if path: self.current_image.save(path); self.statusBar().showMessage(f"Saved → {path}")


def main():
    app = QApplication(sys.argv); w = PropLogicImageApp(); w.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()