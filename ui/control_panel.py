"""
ui/control_panel.py — CleanStream control panel with tabs.

Tab 1: Monitor — pipeline status, preview, stats, controls
Tab 2: Training — review flagged detections, label, retrain
"""

import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
from loguru import logger

FLAGGED_LOG    = Path("flagged_audio.jsonl")
TRAINING_DATA  = Path("models/audio/training_data.jsonl")

# View window opacity. Held just under fully solid (254/255) on purpose:
# Chromium browsers (YouTube et al.) and some media players run native
# window-occlusion detection and FREEZE the video feed — audio keeps going —
# the moment their window is fully covered by an *opaque* window. That's
# exactly what the View window does when it sits on the media monitor.
# Chromium treats a layered window with alpha < 255 as non-opaque and skips
# it as an occluder, so the source keeps rendering. 254/255 is visually
# indistinguishable from solid while still tripping that check. Any value
# < 1.0 works; we stay as close to opaque as possible to minimise the (~0.4%)
# bleed-through of the live media behind the window.
VIEW_WINDOW_OPACITY = 0.996


class ControlPanel:
    BG      = "#0f1117"
    SURFACE = "#1a1d27"
    ACCENT  = "#4f6ef7"
    TEXT    = "#e8eaf0"
    MUTED   = "#6b7280"
    GREEN   = "#22c55e"
    AMBER   = "#f59e0b"
    RED     = "#ef4444"
    PURPLE  = "#a78bfa"

    # Per-browser ways to disable window-occlusion detection — the thing that
    # freezes a covered video while audio keeps playing. (name, copy-value,
    # how-to). The flags pages apply on Windows; the command-line flag below
    # covers macOS/Linux too.
    BROWSER_FIXES = [
        ("Brave", "brave://flags/#calculate-native-win-occlusion",
         "Paste this in the address bar, set “Calculate window occlusion "
         "on Windows” to Disabled, then click Relaunch."),
        ("Chrome", "chrome://flags/#calculate-native-win-occlusion",
         "Paste this in the address bar, set the occlusion flag to Disabled, "
         "then Relaunch."),
        ("Edge", "edge://flags/#calculate-native-win-occlusion",
         "Paste this in the address bar, set the occlusion flag to Disabled, "
         "then Restart."),
        ("Opera / Vivaldi", "opera://flags/#calculate-native-win-occlusion",
         "Chromium-based, same flag. Vivaldi uses vivaldi://flags/# instead "
         "of opera://flags/#."),
        ("Firefox", "widget.windows.window_occlusion_tracking.enabled",
         "Open about:config, search this preference, and set it to false. "
         "Takes effect immediately — no restart needed."),
    ]
    # Cross-platform fallback: relaunch any Chromium browser with these flags.
    LAUNCH_FLAGS = ("--disable-features=CalculateNativeWinOcclusion "
                    "--disable-backgrounding-occluded-windows")

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.root = tk.Tk()
        self.root.title("CleanStream")
        self.root.configure(bg=self.BG)
        self.root.geometry("860x620")
        self.root.minsize(720, 520)
        self.root.resizable(True, True)

        self._preview_frame = None
        self._preview_lock = threading.Lock()
        self._running = False
        self._audio_ai_enabled = False

        # Fullscreen / view-mode state
        self._root_fullscreen = False
        self._pre_fullscreen_geometry = None
        self._root_excluded = False     # control panel kept out of capture
        self._view_win = None
        self._view_excluded = False     # True once kept out of screen capture
        self._view_label = None
        self._view_hint = None

        # Training state
        self._pending_entries = []
        self._current_entry_idx = 0

        self._build_ui()
        self._bind_fullscreen_keys()
        # Hide the control panel from the screen capture so it can sit on the
        # same monitor as the media (same ability as the View window).
        self._apply_root_capture_exclusion()
        self._tick()
        self._frame_tick()

    # ══════════════════════════════════════════════════════════════════════
    # UI BUILD
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = self.root

        # ── Shared header ────────────────────────────────────────────────
        header = tk.Frame(root, bg=self.BG, pady=12)
        header.pack(fill="x", padx=20)

        tk.Label(header, text="CleanStream", font=("Helvetica Neue", 18, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(side="left")

        self._status_dot = tk.Label(header, text="●", font=("Helvetica Neue", 14),
                                    bg=self.BG, fg=self.MUTED)
        self._status_dot.pack(side="left", padx=(10, 4))

        self._status_label = tk.Label(header, text="Stopped",
                                      font=("Helvetica Neue", 12),
                                      bg=self.BG, fg=self.MUTED)
        self._status_label.pack(side="left")

        self._btn = tk.Button(
            header, text="Start", font=("Helvetica Neue", 12, "bold"),
            bg=self.ACCENT, fg="white", relief="flat", padx=16, pady=6,
            activebackground="#3b5bdb", activeforeground="white",
            cursor="hand2", command=self._toggle
        )
        self._btn.pack(side="right")

        self._view_btn = tk.Button(
            header, text="⛶ View Mode", font=("Helvetica Neue", 11, "bold"),
            bg=self.SURFACE, fg=self.TEXT, relief="flat", padx=12, pady=6,
            activebackground="#2d3148", cursor="hand2",
            command=self._toggle_view_mode
        )
        self._view_btn.pack(side="right", padx=(0, 8))

        self._fs_btn = tk.Button(
            header, text="⤢ Fullscreen", font=("Helvetica Neue", 11),
            bg=self.SURFACE, fg=self.TEXT, relief="flat", padx=12, pady=6,
            activebackground="#2d3148", cursor="hand2",
            command=self._toggle_root_fullscreen
        )
        self._fs_btn.pack(side="right", padx=(0, 8))

        # ── Notebook (tabs) ──────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=self.BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=self.SURFACE, foreground=self.MUTED,
                        padding=(16, 6), font=("Helvetica Neue", 11))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", self.BG)],
                  foreground=[("selected", self.TEXT)])

        self._nb = ttk.Notebook(root, style="Dark.TNotebook")
        self._nb.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        # Tab frames
        monitor_tab       = tk.Frame(self._nb, bg=self.BG)
        training_tab      = tk.Frame(self._nb, bg=self.BG)
        wordlist_tab      = tk.Frame(self._nb, bg=self.BG)
        troubleshoot_tab  = tk.Frame(self._nb, bg=self.BG)

        self._nb.add(monitor_tab,      text="  Monitor  ")
        self._nb.add(training_tab,     text="  Training  ")
        self._nb.add(wordlist_tab,     text="  Word List  ")
        self._nb.add(troubleshoot_tab, text="  Troubleshooting  ")

        self._build_monitor_tab(monitor_tab)
        self._build_training_tab(training_tab)
        self._build_wordlist_tab(wordlist_tab)
        self._build_troubleshooting_tab(troubleshoot_tab)

    # ── Monitor tab ───────────────────────────────────────────────────────

    def _build_monitor_tab(self, parent):
        # Content row
        content = tk.Frame(parent, bg=self.BG)
        content.pack(fill="both", expand=True, pady=(8, 0))

        # Preview
        left = tk.Frame(content, bg=self.SURFACE, highlightthickness=1,
                        highlightbackground="#2d3148")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Output preview", font=("Helvetica Neue", 10),
                 bg=self.SURFACE, fg=self.MUTED, pady=6).pack()

        self._preview_label = tk.Label(left, bg="#000000", bd=0, highlightthickness=0)
        self._preview_label.pack(padx=8, pady=(0, 8), fill="both", expand=True)

        # Stats column
        right = tk.Frame(content, bg=self.BG, padx=14)
        right.pack(side="right", fill="y")

        self._make_stat(right, "Video buffer",  "video_buf")
        self._make_stat(right, "Audio buffer",  "audio_buf")
        self._make_stat(right, "Frames out",    "frames_out")
        self._make_stat(right, "Delay",         "delay")
        self._make_stat(right, "AI analyzed",   "ai_analyzed")
        self._make_stat(right, "Detections",    "ai_detections")

        # Pipeline flow
        flow = tk.Frame(parent, bg=self.SURFACE, padx=16, pady=8,
                        highlightthickness=1, highlightbackground="#2d3148")
        flow.pack(fill="x", pady=(8, 6))

        self._stage_labels = {}
        for s in ["Capture", "→", "Buffer", "→", "AI", "→", "Output"]:
            is_arrow = s == "→"
            lbl = tk.Label(flow, text=s,
                           font=("Helvetica Neue", 11, "" if is_arrow else "bold"),
                           bg=self.SURFACE,
                           fg=self.MUTED if is_arrow else self.TEXT)
            lbl.pack(side="left", padx=(0 if is_arrow else 6))
            if not is_arrow:
                self._stage_labels[s.lower()] = lbl

        # Settings row 1 — delay
        row1 = tk.Frame(parent, bg=self.BG, padx=4, pady=2)
        row1.pack(fill="x")

        tk.Label(row1, text="Buffer delay (sec):", font=("Helvetica Neue", 11),
                 bg=self.BG, fg=self.MUTED).pack(side="left")

        self._delay_var = tk.StringVar(value=str(self.pipeline.cfg.buffer.delay_seconds))
        tk.Entry(row1, textvariable=self._delay_var, width=6,
                 font=("Helvetica Neue", 11), bg=self.SURFACE, fg=self.TEXT,
                 insertbackground=self.TEXT, relief="flat", bd=4).pack(side="left", padx=(4, 0))

        # Settings row 2 — filter toggle
        row2 = tk.Frame(parent, bg=self.BG, padx=4, pady=2)
        row2.pack(fill="x")

        tk.Label(row2, text="Profanity filter:", font=("Helvetica Neue", 11),
                 bg=self.BG, fg=self.MUTED).pack(side="left")

        self._audio_filter_var = tk.BooleanVar(value=False)
        self._audio_filter_cb = tk.Checkbutton(
            row2, text="Enable (Whisper small + classifier)",
            variable=self._audio_filter_var,
            bg=self.BG, fg=self.TEXT, selectcolor=self.SURFACE,
            font=("Helvetica Neue", 11), activebackground=self.BG,
            command=self._on_audio_filter_toggle,
            state="disabled"
        )
        self._audio_filter_cb.pack(side="left", padx=(4, 12))

        self._ai_status = tk.Label(row2, text="(start pipeline first)",
                                   font=("Helvetica Neue", 10),
                                   bg=self.BG, fg=self.MUTED)
        self._ai_status.pack(side="left")

        # Settings row 3 — capture source
        row3 = tk.Frame(parent, bg=self.BG, padx=4, pady=2)
        row3.pack(fill="x")

        tk.Label(row3, text="Capture source:", font=("Helvetica Neue", 11),
                 bg=self.BG, fg=self.MUTED).pack(side="left")

        self._window_var = tk.StringVar(value="Full monitor")
        self._window_menu = tk.OptionMenu(row3, self._window_var, "Full monitor",
                                          command=self._on_window_select)
        self._window_menu.config(font=("Helvetica Neue", 10), bg=self.SURFACE,
                                 fg=self.TEXT, relief="flat", highlightthickness=0,
                                 activebackground="#2d3148", width=28)
        self._window_menu["menu"].config(bg=self.SURFACE, fg=self.TEXT)
        self._window_menu.pack(side="left", padx=(4, 8))

        tk.Button(row3, text="↻ Refresh",
                  font=("Helvetica Neue", 10), bg=self.SURFACE, fg=self.TEXT,
                  relief="flat", padx=8, pady=2, cursor="hand2",
                  activebackground="#2d3148",
                  command=self._refresh_window_list).pack(side="left")

        self._refresh_window_list()

    # ── Training tab ──────────────────────────────────────────────────────

    def _build_training_tab(self, parent):
        # Top bar
        top = tk.Frame(parent, bg=self.BG, pady=10)
        top.pack(fill="x", padx=16)

        tk.Label(top, text="Review & Label Detections",
                 font=("Helvetica Neue", 14, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(side="left")

        self._retrain_btn = tk.Button(
            top, text="Retrain Model", font=("Helvetica Neue", 11, "bold"),
            bg=self.PURPLE, fg="white", relief="flat", padx=12, pady=4,
            activebackground="#7c3aed", cursor="hand2",
            command=self._retrain
        )
        self._retrain_btn.pack(side="right")

        tk.Button(
            top, text="Refresh", font=("Helvetica Neue", 11),
            bg=self.SURFACE, fg=self.TEXT, relief="flat", padx=10, pady=4,
            activebackground="#2d3148", cursor="hand2",
            command=self._load_pending
        ).pack(side="right", padx=(0, 8))

        # Counter
        self._pending_label = tk.Label(parent, text="",
                                       font=("Helvetica Neue", 11),
                                       bg=self.BG, fg=self.MUTED)
        self._pending_label.pack(anchor="w", padx=16)

        # Detection card
        card = tk.Frame(parent, bg=self.SURFACE, padx=20, pady=16,
                        highlightthickness=1, highlightbackground="#2d3148")
        card.pack(fill="x", padx=16, pady=(8, 0))

        tk.Label(card, text="Detected text", font=("Helvetica Neue", 10),
                 bg=self.SURFACE, fg=self.MUTED).pack(anchor="w")

        self._detection_text = tk.Label(
            card, text="No detections yet — start the pipeline and enable the profanity filter.",
            font=("Helvetica Neue", 13), bg=self.SURFACE, fg=self.TEXT,
            wraplength=700, justify="left"
        )
        self._detection_text.pack(anchor="w", pady=(4, 8))

        # Severity + hits
        meta = tk.Frame(card, bg=self.SURFACE)
        meta.pack(fill="x")

        tk.Label(meta, text="Auto-detected as:", font=("Helvetica Neue", 10),
                 bg=self.SURFACE, fg=self.MUTED).pack(side="left")

        self._detection_severity = tk.Label(meta, text="—",
                                            font=("Helvetica Neue", 11, "bold"),
                                            bg=self.SURFACE, fg=self.AMBER)
        self._detection_severity.pack(side="left", padx=(6, 20))

        tk.Label(meta, text="Matched words:", font=("Helvetica Neue", 10),
                 bg=self.SURFACE, fg=self.MUTED).pack(side="left")

        self._detection_hits = tk.Label(meta, text="—",
                                        font=("Helvetica Neue", 11),
                                        bg=self.SURFACE, fg=self.TEXT)
        self._detection_hits.pack(side="left", padx=6)

        # Label buttons
        btn_row = tk.Frame(parent, bg=self.BG, pady=12)
        btn_row.pack()

        label_cfg = [
            ("✓ Clean",    "clean",    self.GREEN),
            ("~ Mild",     "mild",     self.AMBER),
            ("! Moderate", "moderate", "#f97316"),
            ("✕ Severe",   "severe",   self.RED),
            ("→ Skip",     None,       self.MUTED),
        ]

        for text, label, color in label_cfg:
            tk.Button(
                btn_row, text=text,
                font=("Helvetica Neue", 11, "bold"),
                bg=color, fg="white", relief="flat",
                padx=14, pady=8, cursor="hand2",
                activebackground=color,
                command=lambda l=label: self._label_current(l)
            ).pack(side="left", padx=4)

        # Training status
        self._train_status = tk.Label(parent, text="",
                                      font=("Helvetica Neue", 11),
                                      bg=self.BG, fg=self.GREEN)
        self._train_status.pack(pady=(8, 0))

        # Load entries on startup
        self._load_pending()

    # ══════════════════════════════════════════════════════════════════════
    # WORD LIST TAB
    # ══════════════════════════════════════════════════════════════════════

    def _build_wordlist_tab(self, parent):
        # Top bar
        top = tk.Frame(parent, bg=self.BG, pady=10)
        top.pack(fill="x", padx=16)

        tk.Label(top, text="Word List Editor",
                 font=("Helvetica Neue", 14, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(side="left")

        tk.Button(
            top, text="Save Changes",
            font=("Helvetica Neue", 11, "bold"),
            bg=self.GREEN, fg="white", relief="flat", padx=12, pady=4,
            activebackground="#16a34a", cursor="hand2",
            command=self._save_word_list
        ).pack(side="right")

        tk.Button(
            top, text="Add Word",
            font=("Helvetica Neue", 11),
            bg=self.ACCENT, fg="white", relief="flat", padx=12, pady=4,
            activebackground="#3b5bdb", cursor="hand2",
            command=self._add_word_dialog
        ).pack(side="right", padx=(0, 8))

        # Info label
        tk.Label(parent,
                 text="Custom words override the built-in list. Changes take effect after saving.",
                 font=("Helvetica Neue", 10), bg=self.BG, fg=self.MUTED).pack(anchor="w", padx=16)

        # Filter bar
        filter_row = tk.Frame(parent, bg=self.BG, padx=16, pady=6)
        filter_row.pack(fill="x")

        tk.Label(filter_row, text="Filter:", font=("Helvetica Neue", 11),
                 bg=self.BG, fg=self.MUTED).pack(side="left")

        self._wl_filter_var = tk.StringVar()
        self._wl_filter_var.trace_add("write", lambda *_: self._refresh_word_list())
        tk.Entry(filter_row, textvariable=self._wl_filter_var, width=20,
                 font=("Helvetica Neue", 11), bg=self.SURFACE, fg=self.TEXT,
                 insertbackground=self.TEXT, relief="flat", bd=4).pack(side="left", padx=(4, 16))

        # Severity filter buttons
        self._wl_severity_filter = tk.StringVar(value="all")
        for label, val, color in [
            ("All", "all", self.MUTED),
            ("Mild", "mild", self.AMBER),
            ("Moderate", "moderate", "#f97316"),
            ("Severe", "severe", self.RED),
            ("Custom", "custom", self.PURPLE),
        ]:
            tk.Radiobutton(
                filter_row, text=label, variable=self._wl_severity_filter,
                value=val, font=("Helvetica Neue", 10),
                bg=self.BG, fg=color, selectcolor=self.SURFACE,
                activebackground=self.BG, cursor="hand2",
                command=self._refresh_word_list
            ).pack(side="left", padx=2)

        # Word list table
        table_frame = tk.Frame(parent, bg=self.SURFACE, highlightthickness=1,
                               highlightbackground="#2d3148")
        table_frame.pack(fill="both", expand=True, padx=16, pady=(4, 8))

        # Header row
        header = tk.Frame(table_frame, bg="#2d3148")
        header.pack(fill="x")
        tk.Label(header, text="Word / Phrase", font=("Helvetica Neue", 10, "bold"),
                 bg="#2d3148", fg=self.MUTED, width=30, anchor="w",
                 padx=8, pady=4).pack(side="left")
        tk.Label(header, text="Severity", font=("Helvetica Neue", 10, "bold"),
                 bg="#2d3148", fg=self.MUTED, width=12, anchor="w",
                 padx=8, pady=4).pack(side="left")
        tk.Label(header, text="Source", font=("Helvetica Neue", 10, "bold"),
                 bg="#2d3148", fg=self.MUTED, width=10, anchor="w",
                 padx=8, pady=4).pack(side="left")
        tk.Label(header, text="Actions", font=("Helvetica Neue", 10, "bold"),
                 bg="#2d3148", fg=self.MUTED, width=12, anchor="w",
                 padx=8, pady=4).pack(side="left")

        # Scrollable list
        canvas = tk.Canvas(table_frame, bg=self.SURFACE, highlightthickness=0)
        scrollbar = tk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
        self._wl_scrollframe = tk.Frame(canvas, bg=self.SURFACE)

        self._wl_scrollframe.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self._wl_scrollframe, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Save status
        self._wl_status = tk.Label(parent, text="",
                                   font=("Helvetica Neue", 10),
                                   bg=self.BG, fg=self.GREEN)
        self._wl_status.pack(pady=(0, 4))

        self._refresh_word_list()

    def _refresh_word_list(self):
        """Rebuild the word list display."""
        from models.audio.word_lookup import load_words
        all_data = load_words()
        severity_filter = self._wl_severity_filter.get()
        text_filter = self._wl_filter_var.get().lower().strip()

        # Clear existing rows
        for widget in self._wl_scrollframe.winfo_children():
            widget.destroy()

        # Build display dict from single words.json
        all_words = {
            word: (entry["severity"], "custom" if entry.get("custom") else "built-in")
            for word, entry in all_data.items()
        }

        # Apply filters
        filtered = {
            w: (s, src) for w, (s, src) in sorted(all_words.items())
            if (severity_filter == "all" or
                (severity_filter == "custom" and src == "custom") or
                s == severity_filter)
            and (not text_filter or text_filter in w)
        }

        sev_colors = {"mild": self.AMBER, "moderate": "#f97316",
                      "severe": self.RED, "clean": self.GREEN}

        if not filtered:
            tk.Label(self._wl_scrollframe, text="No words match the current filter.",
                     font=("Helvetica Neue", 11), bg=self.SURFACE,
                     fg=self.MUTED, pady=20).pack()
            return

        for word, (severity, source) in filtered.items():
            row = tk.Frame(self._wl_scrollframe, bg=self.SURFACE)
            row.pack(fill="x")

            # Separator line
            tk.Frame(row, bg="#2d3148", height=1).pack(fill="x")

            content_row = tk.Frame(row, bg=self.SURFACE)
            content_row.pack(fill="x")

            tk.Label(content_row, text=word,
                     font=("Helvetica Neue", 11),
                     bg=self.SURFACE, fg=self.TEXT,
                     width=30, anchor="w", padx=8, pady=5).pack(side="left")

            tk.Label(content_row, text=severity.upper(),
                     font=("Helvetica Neue", 10, "bold"),
                     bg=self.SURFACE, fg=sev_colors.get(severity, self.TEXT),
                     width=12, anchor="w", padx=8).pack(side="left")

            tk.Label(content_row, text=source,
                     font=("Helvetica Neue", 10),
                     bg=self.SURFACE,
                     fg=self.PURPLE if source == "custom" else self.MUTED,
                     width=10, anchor="w", padx=8).pack(side="left")

            # Action buttons
            actions = tk.Frame(content_row, bg=self.SURFACE)
            actions.pack(side="left")

            # Severity change dropdown
            sev_var = tk.StringVar(value=severity)
            sev_menu = tk.OptionMenu(actions, sev_var, "mild", "moderate", "severe", "clean",
                                     command=lambda val, w=word: self._change_severity(w, val))
            sev_menu.config(font=("Helvetica Neue", 9), bg=self.SURFACE, fg=self.TEXT,
                           relief="flat", highlightthickness=0, activebackground="#2d3148")
            sev_menu["menu"].config(bg=self.SURFACE, fg=self.TEXT)
            sev_menu.pack(side="left", padx=2)

            # Remove button (custom words only)
            if source == "custom":
                tk.Button(
                    actions, text="✕",
                    font=("Helvetica Neue", 10),
                    bg=self.RED, fg="white", relief="flat",
                    padx=6, pady=2, cursor="hand2",
                    activebackground="#b91c1c",
                    command=lambda w=word: self._remove_word(w)
                ).pack(side="left", padx=2)

    def _change_severity(self, word: str, new_severity: str):
        """Change severity of a word (saves to custom words)."""
        from models.audio.word_lookup import set_severity
        set_severity(word, new_severity)
        self._wl_status.config(text=f"Updated: '{word}' → {new_severity}", fg=self.AMBER)
        self._refresh_word_list()

    def _remove_word(self, word: str):
        """Remove a custom word."""
        from models.audio.word_lookup import remove_word
        remove_word(word)
        self._wl_status.config(text=f"Removed: '{word}'", fg=self.RED)
        self._refresh_word_list()

    def _add_word_dialog(self):
        """Open a dialog to add a new custom word."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Word")
        dialog.configure(bg=self.BG)
        dialog.geometry("360x180")
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog, text="Word or phrase:",
                 font=("Helvetica Neue", 11), bg=self.BG, fg=self.MUTED).pack(
                 anchor="w", padx=20, pady=(20, 4))

        word_var = tk.StringVar()
        tk.Entry(dialog, textvariable=word_var, width=30,
                 font=("Helvetica Neue", 12), bg=self.SURFACE, fg=self.TEXT,
                 insertbackground=self.TEXT, relief="flat", bd=6).pack(padx=20)

        tk.Label(dialog, text="Severity:",
                 font=("Helvetica Neue", 11), bg=self.BG, fg=self.MUTED).pack(
                 anchor="w", padx=20, pady=(12, 4))

        sev_var = tk.StringVar(value="mild")
        sev_row = tk.Frame(dialog, bg=self.BG)
        sev_row.pack(padx=20, anchor="w")
        for label, val, color in [("Mild", "mild", self.AMBER),
                                   ("Moderate", "moderate", "#f97316"),
                                   ("Severe", "severe", self.RED),
                                   ("Clean", "clean", self.GREEN)]:
            tk.Radiobutton(sev_row, text=label, variable=sev_var, value=val,
                          font=("Helvetica Neue", 10), bg=self.BG, fg=color,
                          selectcolor=self.SURFACE, activebackground=self.BG,
                          cursor="hand2").pack(side="left", padx=4)

        def do_add():
            word = word_var.get().strip().lower()
            if not word:
                return
            from models.audio.word_lookup import add_word
            add_word(word, sev_var.get())
            self._wl_status.config(text=f"Added: '{word}' as {sev_var.get()}", fg=self.GREEN)
            self._refresh_word_list()
            dialog.destroy()

        tk.Button(dialog, text="Add", font=("Helvetica Neue", 11, "bold"),
                  bg=self.ACCENT, fg="white", relief="flat", padx=20, pady=6,
                  activebackground="#3b5bdb", cursor="hand2",
                  command=do_add).pack(pady=(12, 0))

    def _save_word_list(self):
        """Reload the word list in the running classifier."""
        from models.audio.word_lookup import rebuild
        rebuild()
        if self.pipeline._audio_analysis is not None:
            self.pipeline._audio_analysis.reload_classifier()
            self._wl_status.config(
                text="Word list saved and classifier reloaded ✓", fg=self.GREEN)
        else:
            self._wl_status.config(
                text="Word list saved ✓ (will apply next time filter is enabled)",
                fg=self.GREEN)

    # ══════════════════════════════════════════════════════════════════════
    # TROUBLESHOOTING TAB
    # ══════════════════════════════════════════════════════════════════════

    def _build_troubleshooting_tab(self, parent):
        # Header
        top = tk.Frame(parent, bg=self.BG, pady=10)
        top.pack(fill="x", padx=16)
        tk.Label(top, text="Troubleshooting",
                 font=("Helvetica Neue", 14, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(side="left")

        # Scrollable body (several browser cards may overflow the window)
        container = tk.Frame(parent, bg=self.BG)
        container.pack(fill="both", expand=True, padx=16, pady=(0, 4))
        canvas = tk.Canvas(container, bg=self.BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=self.BG)
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── Section: covered-video freeze ─────────────────────────────────
        tk.Label(body, text="Video freezes when the View window covers the player",
                 font=("Helvetica Neue", 12, "bold"),
                 bg=self.BG, fg=self.TEXT, anchor="w").pack(fill="x", pady=(4, 2))
        tk.Label(
            body,
            text=("CleanStream holds the View window slightly translucent so most "
                  "browsers keep playing while it's covered. If your browser still "
                  "freezes the picture (audio keeps going), turn off its window-"
                  "occlusion detection using the setting for your browser below. "
                  "Click Copy, paste it into the browser, and apply."),
            font=("Helvetica Neue", 10), bg=self.BG, fg=self.MUTED,
            wraplength=720, justify="left", anchor="w").pack(fill="x", pady=(0, 10))

        for name, value, desc in self.BROWSER_FIXES:
            self._make_fix_card(body, name, value, desc)

        # ── Command-line fallback (cross-platform) ────────────────────────
        tk.Label(body, text="Command-line alternative (any Chromium browser · any OS)",
                 font=("Helvetica Neue", 12, "bold"),
                 bg=self.BG, fg=self.TEXT, anchor="w").pack(fill="x", pady=(14, 2))
        tk.Label(
            body,
            text=("Fully quit the browser, then start it with these flags. Works on "
                  "Windows, macOS and Linux — useful where the flags page above isn't "
                  "available."),
            font=("Helvetica Neue", 10), bg=self.BG, fg=self.MUTED,
            wraplength=720, justify="left", anchor="w").pack(fill="x", pady=(0, 6))
        self._make_copy_row(body, self.LAUNCH_FLAGS)

        # Status line (shared across copy buttons)
        self._ts_status = tk.Label(parent, text="",
                                   font=("Helvetica Neue", 10),
                                   bg=self.BG, fg=self.GREEN)
        self._ts_status.pack(pady=(0, 4))

    def _make_fix_card(self, parent, name, value, desc):
        card = tk.Frame(parent, bg=self.SURFACE, padx=12, pady=8,
                        highlightthickness=1, highlightbackground="#2d3148")
        card.pack(fill="x", pady=4)
        tk.Label(card, text=name, font=("Helvetica Neue", 11, "bold"),
                 bg=self.SURFACE, fg=self.ACCENT, anchor="w").pack(fill="x")
        tk.Label(card, text=desc, font=("Helvetica Neue", 10),
                 bg=self.SURFACE, fg=self.MUTED, wraplength=680,
                 justify="left", anchor="w").pack(fill="x", pady=(2, 6))
        self._make_copy_row(card, value, bg=self.SURFACE)

    def _make_copy_row(self, parent, value, bg=None):
        bg = bg or self.BG
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x")
        var = tk.StringVar(value=value)
        entry = tk.Entry(row, textvariable=var, font=("Courier New", 10),
                         readonlybackground=self.BG, fg=self.TEXT,
                         insertbackground=self.TEXT, relief="flat", bd=4,
                         state="readonly")
        entry.pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Copy", font=("Helvetica Neue", 10),
                  bg=self.ACCENT, fg="white", relief="flat", padx=10, pady=2,
                  activebackground="#3b5bdb", cursor="hand2",
                  command=lambda v=value: self._copy_to_clipboard(v)).pack(
                  side="right", padx=(8, 0))

    def _copy_to_clipboard(self, text: str):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            if getattr(self, "_ts_status", None) is not None:
                self._ts_status.config(text=f"Copied: {text}", fg=self.GREEN)
        except Exception as e:
            logger.warning(f"Clipboard copy failed: {e}")
            if getattr(self, "_ts_status", None) is not None:
                self._ts_status.config(text=f"Copy failed: {e}", fg=self.RED)

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def _load_pending(self):
        """Load unreviewed detections from flagged_audio.jsonl."""
        if not FLAGGED_LOG.exists():
            self._pending_entries = []
            self._pending_label.config(text="No detections logged yet.")
            self._show_empty_card()
            return

        with open(FLAGGED_LOG) as f:
            all_entries = [json.loads(l) for l in f if l.strip()]

        self._pending_entries = [e for e in all_entries if not e.get("reviewed")]
        self._current_entry_idx = 0

        count = len(self._pending_entries)
        if count == 0:
            self._pending_label.config(text="All detections reviewed ✓")
            self._show_empty_card()
        else:
            self._pending_label.config(
                text=f"{count} unreviewed detection{'s' if count != 1 else ''}")
            self._show_entry(0)

    def _show_entry(self, idx: int):
        if not self._pending_entries or idx >= len(self._pending_entries):
            self._show_empty_card()
            return

        entry = self._pending_entries[idx]
        self._detection_text.config(text=f'"{entry["text"]}"')

        sev = entry.get("severity", "—")
        sev_colors = {"clean": self.GREEN, "mild": self.AMBER,
                      "moderate": "#f97316", "severe": self.RED}
        self._detection_severity.config(text=sev.upper(),
                                        fg=sev_colors.get(sev, self.TEXT))

        hits = entry.get("hits", [])
        hits_str = ", ".join(f"{w} ({s})" for w, s in hits) if hits else "none"
        self._detection_hits.config(text=hits_str)

        remaining = len(self._pending_entries) - idx
        self._pending_label.config(
            text=f"{remaining} unreviewed detection{'s' if remaining != 1 else ''} "
                 f"({idx + 1} of {len(self._pending_entries)})")

    def _show_empty_card(self):
        self._detection_text.config(
            text="No unreviewed detections — click Refresh after the filter runs.")
        self._detection_severity.config(text="—", fg=self.AMBER)
        self._detection_hits.config(text="—")

    def _label_current(self, label: str | None):
        """Label current entry and advance to next."""
        if not self._pending_entries:
            return

        idx = self._current_entry_idx
        if idx >= len(self._pending_entries):
            return

        entry = self._pending_entries[idx]

        if label is not None:
            # Save to training data
            from models.audio.classifier import add_example
            add_example(entry["text"], label)
            self._train_status.config(
                text=f"Labeled as {label.upper()}: \"{entry['text'][:50]}\"",
                fg=self.GREEN)

        # Mark as reviewed in log
        entry["reviewed"] = True
        self._save_log()

        # Advance
        self._current_entry_idx += 1
        if self._current_entry_idx < len(self._pending_entries):
            self._show_entry(self._current_entry_idx)
        else:
            self._pending_label.config(text="All caught up ✓")
            self._show_empty_card()

    def _save_log(self):
        """Write updated reviewed status back to flagged_audio.jsonl."""
        if not FLAGGED_LOG.exists():
            return
        try:
            with open(FLAGGED_LOG) as f:
                all_entries = [json.loads(l) for l in f if l.strip()]
            # Update reviewed status
            reviewed_texts = {e["text"] for e in self._pending_entries if e.get("reviewed")}
            for e in all_entries:
                if e["text"] in reviewed_texts:
                    e["reviewed"] = True
            with open(FLAGGED_LOG, "w") as f:
                for e in all_entries:
                    f.write(json.dumps(e) + "\n")
        except Exception as ex:
            logger.warning(f"Could not save log: {ex}")

    def _retrain(self):
        """Retrain the classifier in a background thread."""
        self._train_status.config(text="Training...", fg=self.AMBER)
        self._retrain_btn.config(state="disabled")

        def do_train():
            try:
                from models.audio.classifier import train
                stats = train()
                if "error" in stats:
                    msg = f"Training failed: {stats['error']}"
                    color = self.RED
                else:
                    msg = (f"Model trained — accuracy {stats['accuracy']*100:.1f}% "
                           f"on {stats['examples']} examples")
                    color = self.GREEN
                    # Reload classifier in running pipeline
                    if self.pipeline._audio_analysis is not None:
                        self.pipeline._audio_analysis.reload_classifier()
            except Exception as e:
                msg = f"Error: {e}"
                color = self.RED

            self.root.after(0, lambda: self._train_status.config(text=msg, fg=color))
            self.root.after(0, lambda: self._retrain_btn.config(state="normal"))

        threading.Thread(target=do_train, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════
    # MONITOR LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def _make_stat(self, parent, label: str, key: str):
        card = tk.Frame(parent, bg=self.SURFACE, padx=12, pady=6,
                        highlightthickness=1, highlightbackground="#2d3148")
        card.pack(fill="x", pady=(0, 5))
        tk.Label(card, text=label, font=("Helvetica Neue", 10),
                 bg=self.SURFACE, fg=self.MUTED).pack(anchor="w")
        val = tk.Label(card, text="—", font=("Helvetica Neue", 13, "bold"),
                       bg=self.SURFACE, fg=self.TEXT)
        val.pack(anchor="w")
        setattr(self, f"_stat_{key}", val)

    def _toggle(self):
        if not self._running:
            self._start()
        else:
            self._stop()

    def _start(self):
        try:
            delay = float(self._delay_var.get())
            self.pipeline.cfg.buffer.delay_seconds = delay
        except ValueError:
            messagebox.showerror("Error", "Delay must be a number")
            return
        self.pipeline.start(preview_callback=self._on_preview_frame)
        self._running = True
        self._btn.config(text="Stop", bg=self.RED)
        self._set_status("Running", self.GREEN)
        self._audio_filter_cb.config(state="normal")
        self._ai_status.config(text="(check to enable)", fg=self.MUTED)
        for stage in ["capture", "buffer", "output"]:
            self._update_stage(stage, self.GREEN)
        self._update_stage("ai", self.MUTED)

    def _stop(self):
        self.pipeline.stop()
        self._running = False
        self._audio_ai_enabled = False
        self._audio_filter_var.set(False)
        self._btn.config(text="Start", bg=self.ACCENT)
        self._set_status("Stopped", self.MUTED)
        self._audio_filter_cb.config(state="disabled")
        self._ai_status.config(text="(start pipeline first)", fg=self.MUTED)
        for stage in self._stage_labels:
            self._update_stage(stage, self.MUTED)

    def _on_audio_filter_toggle(self):
        if self._audio_filter_var.get() and not self._audio_ai_enabled:
            self._audio_ai_enabled = True
            self._ai_status.config(text="Loading Whisper...", fg=self.AMBER)
            self._update_stage("ai", self.AMBER)
            threading.Thread(target=self._load_audio_model, daemon=True).start()
        elif not self._audio_filter_var.get():
            self._audio_filter_var.set(True)
            messagebox.showinfo("CleanStream",
                "Audio filter stays active until you stop and restart the pipeline.")

    def _load_audio_model(self):
        try:
            self.pipeline.attach_audio_model(whisper_model_size="base")
            self.root.after(0, lambda: self._ai_status.config(
                text="Whisper ready ✓", fg=self.GREEN))
            self.root.after(0, lambda: self._update_stage("ai", self.GREEN))
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            self.root.after(0, lambda: self._ai_status.config(
                text=f"Failed to load: {e}", fg=self.RED))
            self.root.after(0, lambda: self._update_stage("ai", self.RED))

    def _refresh_window_list(self):
        """Refresh the capture source dropdown with current open windows."""
        from capture.capture import list_capturable_windows
        windows = list_capturable_windows()
        menu = self._window_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(label="Full monitor",
                         command=lambda: self._window_var.set("Full monitor"))
        for w in windows:
            title = w["title"]
            if len(title) > 50:
                title = title[:47] + "..."
            menu.add_command(
                label=title,
                command=lambda t=w["title"]: self._window_var.set(t)
            )

    def _on_window_select(self, value: str):
        """Apply capture source selection to pipeline config."""
        title = "" if value == "Full monitor" else value
        self.pipeline.set_capture_window(title)
        if title:
            logger.info(f"Capture source: {title!r} (takes effect on next Start)")
        else:
            logger.info("Capture source: full monitor (takes effect on next Start)")

    def _set_status(self, text: str, color: str):
        self._status_label.config(text=text, fg=color)
        self._status_dot.config(fg=color)

    def _update_stage(self, stage: str, color: str):
        if stage in self._stage_labels:
            self._stage_labels[stage].config(fg=color)

    def _on_preview_frame(self, frame):
        with self._preview_lock:
            self._preview_frame = frame

    def _tick(self):
        """Slow loop — stats and labels. 5Hz is plenty for numbers; frame
        rendering runs on its own faster loop (see _frame_tick) so video
        smoothness isn't tied to how often these labels need to refresh."""
        if self._running:
            stats = self.pipeline.stats()
            self._stat_video_buf.config(
                text=f"{stats.get('video_frames', 0)} frames")
            self._stat_audio_buf.config(
                text=f"{stats.get('audio_seconds', 0):.1f}s")
            self._stat_frames_out.config(
                text=str(stats.get('frames_out', 0)))
            self._stat_delay.config(
                text=f"{self.pipeline.cfg.buffer.delay_seconds:.1f}s")

            ai = stats.get("audio_ai", {})
            self._stat_ai_analyzed.config(
                text=str(ai.get("chunks_analyzed", "—")))
            det = ai.get("detections", 0)
            self._stat_ai_detections.config(
                text=str(det) if ai else "—",
                fg=self.RED if det > 0 else self.TEXT)

        scheduler = self._view_win if self._view_win is not None else self.root
        scheduler.after(200, self._tick)

    def _frame_tick(self):
        """
        Fast loop — pushes the newest buffered frame to whichever display
        surface is actually visible right now (the embedded Monitor-tab
        preview, or the full-screen View Mode window — never both at once,
        since only one is ever on screen). Runs at roughly the configured
        capture fps instead of the old fixed 200ms stats interval, which
        was capping the in-app preview at ~5fps while the virtual camera
        (and OBS) kept receiving the full frame rate.
        """
        with self._preview_lock:
            frame = self._preview_frame

        # Refresh the capture-exclusion rect so the mask tracks whichever
        # window is visible (View window, or the control panel) as it's
        # dragged/resized. No-op effect on Windows (display affinity handles it).
        self._publish_exclusion_rect()

        if frame is not None:
            if self._view_win is not None:
                self._render_view_frame(frame)
            else:
                self._render_preview(frame)

        fps = max(1, int(getattr(self.pipeline.cfg.capture, "fps", 30)))
        interval_ms = max(15, round(1000 / fps))
        scheduler = self._view_win if self._view_win is not None else self.root
        scheduler.after(interval_ms, self._frame_tick)

    def _render_preview(self, frame):
        w = self._preview_label.winfo_width()
        h = self._preview_label.winfo_height()
        if w < 20 or h < 20:
            w, h = 460, 260  # not yet laid out — fall back to the old default
        canvas = self._scale_to_fit(frame, w, h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self._preview_label.config(image=photo)
        self._preview_label.image = photo

    @staticmethod
    def _scale_to_fit(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Fit `frame` within target_w x target_h while preserving aspect
        ratio, then letterbox it onto a black canvas of exactly that size.

        Scale is capped at 1.0 — the frame is downscaled if it's bigger
        than the target box, but never upscaled past its native pixel
        resolution. Without this cap, View Mode on a monitor larger than
        the source frame (e.g. the 1280x720 output stretched across a
        1440p/4K screen) would magnify it and visibly soften/blur the
        image. Capping at 1:1 and centering on black keeps every pixel
        that's shown crisp, at the cost of not filling the whole screen
        on a monitor much bigger than the source resolution.
        """
        target_w, target_h = max(1, target_w), max(1, target_h)
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h, 1.0)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        if (new_w, new_h) != (w, h):
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        else:
            resized = frame  # exact native size already — no resample needed
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    # ── Fullscreen (control panel) ───────────────────────────────────────

    def _bind_fullscreen_keys(self):
        """F11 toggles fullscreen for the control panel itself; Esc exits it."""
        self.root.bind("<F11>", self._toggle_root_fullscreen)
        self.root.bind("<Escape>", self._exit_root_fullscreen)

    def _get_monitor_bounds(self) -> tuple[int, int, int, int]:
        """
        Return (left, top, width, height) of the physical monitor that
        currently contains the control panel window.

        Plain Tk winfo_screenwidth()/height() — and the '-fullscreen'
        window attribute built on top of them — report the *primary*
        monitor's size on Windows (and some Linux WMs) even when the
        window actually lives on a different monitor. That's why
        fullscreen was snapping back to the main display. mss (already
        a dependency for screen capture) enumerates real per-monitor
        geometry, so we use that to find whichever monitor the window
        is actually sitting on right now.
        """
        try:
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = max(1, self.root.winfo_width())
            h = max(1, self.root.winfo_height())
            cx, cy = x + w // 2, y + h // 2

            import mss
            with mss.mss() as sct:
                for m in sct.monitors[1:]:  # skip index 0 = combined virtual desktop
                    if m["left"] <= cx < m["left"] + m["width"] and \
                       m["top"] <= cy < m["top"] + m["height"]:
                        return m["left"], m["top"], m["width"], m["height"]
                # Window center didn't land inside any reported monitor
                # (can happen mid-drag) — fall back to the combined desktop.
                combined = sct.monitors[0]
                return combined["left"], combined["top"], combined["width"], combined["height"]
        except Exception as e:
            logger.warning(f"Monitor detection failed, falling back to primary display: {e}")
            return 0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def _toggle_root_fullscreen(self, event=None):
        self._root_fullscreen = not self._root_fullscreen
        if self._root_fullscreen:
            self._pre_fullscreen_geometry = self.root.geometry()
            left, top, w, h = self._get_monitor_bounds()
            self.root.overrideredirect(True)
            self.root.geometry(f"{w}x{h}+{left}+{top}")
        else:
            self.root.overrideredirect(False)
            self.root.geometry(self._pre_fullscreen_geometry or "860x620")
        self._fs_btn.config(text="⤡ Exit Fullscreen" if self._root_fullscreen else "⤢ Fullscreen")
        # overrideredirect re-creates the native window on some platforms,
        # which drops display affinity — re-assert capture exclusion.
        self.root.after(60, self._apply_root_capture_exclusion)

    def _exit_root_fullscreen(self, event=None):
        if self._root_fullscreen:
            self._toggle_root_fullscreen()

    # ── View mode (distraction-free, video only) ─────────────────────────

    def _toggle_view_mode(self):
        if self._view_win is not None:
            self._close_view_mode()
        else:
            self._open_view_mode()

    def _open_view_mode(self):
        """
        Open a resizable video-only window with native title bar and
        window controls. The user can minimize, maximize, resize, or
        move it anywhere — including onto the same monitor the media is
        playing on, because the window is kept out of the screen capture
        (see _apply_capture_exclusion) so it never captures its own output.
        F11 toggles fullscreen. Esc or X to close.
        """
        win = tk.Toplevel(self.root)
        win.title("CleanStream — Output")
        win.configure(bg="black")
        win.resizable(True, True)
        win.geometry("960x540")

        win.protocol("WM_DELETE_WINDOW", self._close_view_mode)
        win.bind("<Escape>", lambda e: self._close_view_mode())
        win.bind("<F11>", lambda e: self._view_toggle_fullscreen())

        label = tk.Label(win, bg="black", bd=0, highlightthickness=0)
        label.pack(fill="both", expand=True)

        hint = tk.Label(win, text="F11 fullscreen  |  Esc to close",
                        font=("Helvetica Neue", 10), bg="black", fg="#3a3f4f")
        hint.place(relx=1.0, rely=1.0, anchor="se", x=-14, y=-10)

        self._view_win = win
        self._view_label = label
        self._view_hint = hint

        # Keep this window out of the screen capture so it can live on the
        # media monitor without feeding its own output back in.
        self._apply_capture_exclusion()

        # Keep the media source (browser/player) from pausing its video feed
        # when this window covers it.
        self._prevent_source_pause()

        # Withdraw control panel to save CPU (no preview rendering while
        # view window is open), but schedule after() on the Toplevel win
        # so the tick loop keeps firing regardless.
        self.root.withdraw()
        win.lift()
        win.focus_force()
        win.after(3000, self._fade_view_hint)

    def _apply_root_capture_exclusion(self):
        """
        Keep the control panel itself out of the screen capture, so it can sit
        on the same monitor as the media — exactly like the View window.

        Windows: SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) draws the
        panel on the monitor but omits it from the capture. Other platforms
        rely on the per-frame rect mask published by _publish_exclusion_rect.
        Safe to call repeatedly; re-applied after deiconify/fullscreen since
        those can reset the window's display affinity.

        Also applies the same slight translucency the View window uses so the
        media source (browser/player) doesn't occlusion-pause its video when
        the panel sits over it — see _prevent_source_pause.
        """
        try:
            self.root.update_idletasks()  # ensure the native HWND exists
            from capture.capture import exclude_from_capture
            self._root_excluded = exclude_from_capture(self.root.winfo_id())
            if self._root_excluded:
                logger.info("Control panel hidden from screen capture ✓")
            elif sys.platform == "win32":
                logger.warning("Control panel capture exclusion unavailable — "
                               "keep it off the media monitor, or use View Mode.")
        except Exception as e:
            logger.warning(f"Control panel capture exclusion failed: {e}")
        # Anti-occlusion: keep the source from pausing when the panel covers it.
        try:
            self.root.attributes("-alpha", VIEW_WINDOW_OPACITY)
        except Exception as e:
            logger.debug(f"Could not set control panel opacity (anti-occlusion): {e}")

    def _apply_capture_exclusion(self):
        """
        Keep the View window from capturing itself when it's on the media
        monitor.

        Windows: SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) — the window
        is drawn on the monitor but omitted from the capture entirely, so the
        media behind it is what gets captured.

        Other platforms (mss reads the composited desktop, no exclusion API):
        publish the window's screen rect so the capture loop blanks that
        region instead (see Pipeline.set_exclusion_rect). The rect is refreshed
        every frame by _publish_exclusion_rect, so it tracks moves/resizes.
        """
        win = self._view_win
        if win is None:
            return
        win.update_idletasks()  # make sure the native window/HWND exists
        from capture.capture import exclude_from_capture
        self._view_excluded = exclude_from_capture(win.winfo_id())
        self._publish_exclusion_rect()

        if self._view_excluded:
            note = "hidden from capture ✓"
        elif sys.platform == "win32":
            note = "⚠ capture exclusion unavailable — move off the media monitor"
        else:
            note = "capture-masked"
        if self._view_hint is not None:
            self._view_hint.config(
                text=f"F11 fullscreen  |  Esc close   ·   {note}")

    def _prevent_source_pause(self):
        """
        Stop the media source (YouTube/Chromium, some players) from pausing its
        video when the View window covers it.

        These apps run native window-occlusion detection and freeze the video
        feed (audio keeps going) once their window is fully hidden behind an
        opaque window. Making the View window very slightly translucent — a
        layered window with alpha < 255, set here via Tk's '-alpha' — means the
        browser no longer counts it as an occluder, so playback continues. The
        ~0.4% transparency is imperceptible. If a particular browser/player
        still pauses, launch it with native occlusion disabled (see README).
        """
        win = self._view_win
        if win is None:
            return
        try:
            win.attributes("-alpha", VIEW_WINDOW_OPACITY)
        except Exception as e:
            logger.warning(f"Could not set View window opacity (anti-occlusion): {e}")

    def _view_toggle_fullscreen(self):
        win = self._view_win
        if win is None:
            return
        win.attributes("-fullscreen", not win.attributes("-fullscreen"))
        # Re-assert exclusion + anti-occlusion — toggling fullscreen can reset
        # window styles on some platforms. Defer briefly so state settles.
        from capture.capture import exclude_from_capture

        def _reassert():
            if self._view_win is None:
                return
            exclude_from_capture(win.winfo_id())
            try:
                win.attributes("-alpha", VIEW_WINDOW_OPACITY)
            except Exception:
                pass

        win.after(60, _reassert)

    def _publish_exclusion_rect(self):
        """
        Push the currently-visible CleanStream window's screen rectangle to the
        pipeline so the mss capture path can blank it (the non-Windows
        fallback). When View Mode is open that's the View window; otherwise it's
        the control panel itself, so the panel gets the same on-the-media-
        monitor masking the View window has. Runs on the UI thread; the capture
        thread only ever reads the published tuple and never touches Tk. No-op
        effect on Windows, where display affinity already removes the window.
        """
        win = self._view_win if self._view_win is not None else self.root
        rect = None
        if win is not None:
            try:
                if win.winfo_viewable():
                    rect = (win.winfo_rootx(), win.winfo_rooty(),
                            win.winfo_width(), win.winfo_height())
            except Exception:
                rect = None
        try:
            self.pipeline.set_exclusion_rect(rect)
        except Exception:
            pass

    def _fade_view_hint(self):
        if self._view_hint is not None:
            try:
                self._view_hint.place_forget()
            except Exception:
                pass

    def _close_view_mode(self):
        if self._view_win is not None:
            try:
                self._view_win.destroy()
            except Exception:
                pass
        self._view_win = None
        self._view_excluded = False
        self._view_label = None
        self._view_hint = None
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        # The control panel is on screen again — re-assert its own capture
        # exclusion (deiconify can reset display affinity) so it too can sit
        # on the media monitor without feeding back into the capture.
        self._apply_root_capture_exclusion()
        # Restart tick loops on root now that view window is gone
        self.root.after(16, self._frame_tick)
        self.root.after(200, self._tick)

    def _render_view_frame(self, frame):
        win, label = self._view_win, self._view_label
        if win is None or label is None:
            return
        w = win.winfo_width() or win.winfo_screenwidth()
        h = win.winfo_height() or win.winfo_screenheight()
        canvas = self._scale_to_fit(frame, w, h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if self._view_win is not None:
            try:
                self._view_win.destroy()
            except Exception:
                pass
        if self._running:
            self.pipeline.stop()
        self.root.destroy()
