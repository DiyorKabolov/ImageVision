import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
import json
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np

from llm_controller import LLMController
from image_processor import ImageProcessor

logger = logging.getLogger(__name__)

BG_DARK       = "#0f1117"
BG_PANEL      = "#1a1d27"
BG_CARD       = "#22263a"
ACCENT        = "#5b7fff"
ACCENT_HOVER  = "#7a9bff"
TEXT_PRIMARY  = "#e8eaf6"
TEXT_SECONDARY= "#8892b0"
TEXT_DIM      = "#4a5568"
SUCCESS       = "#43d98c"
WARNING       = "#f6c90e"
ERROR         = "#ff5f72"
BORDER        = "#2d3353"

FONT_TITLE = ("Consolas", 13, "bold")
FONT_BODY  = ("Consolas", 10)
FONT_SMALL = ("Consolas", 9)
FONT_LOG   = ("Consolas", 9)
FONT_MONO  = ("Courier New", 10)

CONFIG_FILE = "server_config.json"


def load_server_url() -> str:
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f).get("base_url", "http://localhost:1234/v1")
    except Exception:
        return "http://localhost:1234/v1"


def save_server_url(url: str):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"base_url": url}, f)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  ZoomableCanvas — канвас с зумом через колёсико мыши
# ─────────────────────────────────────────────────────────────────────────────

class ZoomableCanvas(tk.Canvas):
    """
    tk.Canvas с поддержкой зума (колёсиком мыши) и прокрутки/панорамирования (drag ЛКМ).
    Хранит последнее BGR-изображение и перерисовывает его при изменении зума/смещения.
    """

    ZOOM_MIN = 0.1
    ZOOM_MAX = 10.0
    ZOOM_STEP = 1.15   # множитель на каждый тик колёсика

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._bgr_img: np.ndarray | None = None
        self._zoom: float = 1.0
        self._offset_x: float = 0.0   # смещение центра изображения по X
        self._offset_y: float = 0.0   # смещение центра изображения по Y
        self._drag_start: tuple | None = None
        self._tk_img = None  # держим ссылку, чтобы GC не удалил

        # Привязки зума
        self.bind("<MouseWheel>",      self._on_mousewheel_win)  # Windows
        self.bind("<Button-4>",        self._on_scroll_up)        # Linux
        self.bind("<Button-5>",        self._on_scroll_down)      # Linux
        self.bind("<Configure>",       self._on_configure)

        # Привязки панорамирования (drag)
        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<ButtonRelease-1>", self._on_drag_end)

    # ── публичный API ──────────────────────────────────────────────────────

    def show(self, bgr_img: np.ndarray | None):
        """Сохранить изображение и отрисовать с текущим зумом."""
        self._bgr_img = bgr_img
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._redraw()

    def reset_zoom(self):
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._redraw()

    @property
    def zoom(self) -> float:
        return self._zoom

    # ── внутренние обработчики ─────────────────────────────────────────────

    def _on_mousewheel_win(self, event):
        if event.delta > 0:
            self._zoom_at(event.x, event.y, self.ZOOM_STEP)
        else:
            self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def _on_scroll_up(self, event):
        self._zoom_at(event.x, event.y, self.ZOOM_STEP)

    def _on_scroll_down(self, event):
        self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def _on_configure(self, event):
        self._redraw()

    def _on_drag_start(self, event):
        if self._bgr_img is None:
            return
        self._drag_start = (event.x, event.y)
        self.config(cursor="fleur")

    def _on_drag_move(self, event):
        if self._drag_start is None or self._bgr_img is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._offset_x += dx
        self._offset_y += dy
        self._clamp_offset()
        self._redraw()

    def _on_drag_end(self, event):
        self._drag_start = None
        self.config(cursor="")

    def _zoom_at(self, mx: int, my: int, factor: float):
        """Зум с сохранением точки под курсором."""
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        self.update_idletasks()
        cw = self.winfo_width()  or 450
        ch = self.winfo_height() or 380
        # Текущий центр изображения на канвасе
        cx = cw // 2 + self._offset_x
        cy = ch // 2 + self._offset_y
        # Сдвинуть центр так, чтобы точка под курсором не сдвинулась
        ratio = new_zoom / self._zoom
        self._offset_x = mx - (mx - cx) * ratio - cw // 2
        self._offset_y = my - (my - cy) * ratio - ch // 2
        self._zoom = new_zoom
        self._clamp_offset()
        self._redraw()

    def _clamp_offset(self):
        """Ограничиваем смещение, чтобы изображение не уходило полностью за край."""
        if self._bgr_img is None:
            return
        self.update_idletasks()
        cw = self.winfo_width()  or 450
        ch = self.winfo_height() or 380
        ih, iw = self._bgr_img.shape[:2]
        base_scale = min(cw / iw, ch / ih)
        scale = base_scale * self._zoom
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        # Минимум четверть изображения должна оставаться видимой
        max_ox = max(0, (nw + cw) // 2 - cw // 4)
        max_oy = max(0, (nh + ch) // 2 - ch // 4)
        self._offset_x = max(-max_ox, min(max_ox, self._offset_x))
        self._offset_y = max(-max_oy, min(max_oy, self._offset_y))

    def _redraw(self):
        if self._bgr_img is None:
            return
        self.update_idletasks()
        cw = self.winfo_width()  or 450
        ch = self.winfo_height() or 380

        rgb = cv2.cvtColor(self._bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ih, iw = self._bgr_img.shape[:2]

        # Базовый масштаб «вписать в канвас»
        base_scale = min(cw / iw, ch / ih)
        scale = base_scale * self._zoom

        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        pil = pil.resize((nw, nh), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(pil)
        self.delete("all")
        cx = cw // 2 + int(self._offset_x)
        cy = ch // 2 + int(self._offset_y)
        self.create_image(cx, cy, anchor=tk.CENTER, image=self._tk_img)

        # Подпись зума в правом нижнем углу
        zoom_pct = int(self._zoom * 100)
        self.create_text(
            cw - 6, ch - 6,
            text=f"{zoom_pct}%",
            anchor=tk.SE,
            fill=TEXT_DIM,
            font=("Consolas", 8),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CanvasPanel — контейнер с заголовком, кнопками и ZoomableCanvas
# ─────────────────────────────────────────────────────────────────────────────

class CanvasPanel:
    """
    Панель с заголовком, кнопками «Открепить» / «Сбросить зум»
    и ZoomableCanvas внутри.

    Поддерживает открепление в отдельный Toplevel.
    """

    def __init__(self, master, title: str, title_color: str = TEXT_SECONDARY):
        self.master = master
        self.title  = title
        self.title_color = title_color
        self._toplevel: tk.Toplevel | None = None
        self._bgr_img: np.ndarray | None = None

        # ── Внешний фрейм (всегда находится в master) ──────────────────────
        self.frame = tk.Frame(master, bg=BG_PANEL,
                              highlightbackground=BORDER, highlightthickness=1)

        # ── Заголовочная строка ────────────────────────────────────────────
        header = tk.Frame(self.frame, bg=BG_PANEL)
        header.pack(fill=tk.X, padx=8, pady=(6, 0))

        tk.Label(header, text=title, bg=BG_PANEL,
                 fg=title_color, font=FONT_SMALL).pack(side=tk.LEFT)

        # Кнопки справа
        btn_cfg = dict(bg=BG_CARD, fg=TEXT_DIM, activebackground=BG_DARK,
                       activeforeground=ACCENT, relief=tk.FLAT,
                       font=("Consolas", 8), cursor="hand2",
                       padx=5, pady=2)

        self._detach_btn = tk.Button(
            header, text="⊞ Открепить",
            command=self._detach, **btn_cfg
        )
        self._detach_btn.pack(side=tk.RIGHT, padx=(2, 0))

        tk.Button(
            header, text="🔍 1:1",
            command=self._reset_zoom, **btn_cfg
        ).pack(side=tk.RIGHT, padx=(2, 0))

        # ── Канвас ────────────────────────────────────────────────────────
        self.canvas = ZoomableCanvas(self.frame, bg=BG_CARD, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Заглушка «открыто в окне» ─────────────────────────────────────
        self._placeholder = tk.Label(
            self.frame,
            text="📐  Открыто в отдельном окне",
            bg=BG_CARD, fg=TEXT_DIM,
            font=("Consolas", 10, "italic")
        )

    # ── публичный API ──────────────────────────────────────────────────────

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def show(self, bgr_img: np.ndarray | None):
        """Отобразить изображение (в главном канвасе или в открепленном окне)."""
        self._bgr_img = bgr_img
        # Показываем в активном канвасе
        self._active_canvas().show(bgr_img)

    def reset_zoom(self):
        self._active_canvas().reset_zoom()

    # ── внутренние методы ──────────────────────────────────────────────────

    def _active_canvas(self) -> ZoomableCanvas:
        """Возвращает канвас, который сейчас видим пользователю."""
        if self._toplevel and self._toplevel.winfo_exists():
            return self._detached_canvas
        return self.canvas

    def _reset_zoom(self):
        self._active_canvas().reset_zoom()

    def _detach(self):
        """Открепить канвас в отдельное Toplevel-окно."""
        if self._toplevel and self._toplevel.winfo_exists():
            self._toplevel.lift()
            return

        # Скрыть основной канвас, показать заглушку
        self.canvas.pack_forget()
        self._placeholder.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._detach_btn.config(text="⊞ В панели", state=tk.DISABLED)

        # Создать окно
        top = tk.Toplevel(self.master)
        top.title(f"ImageVision — {self.title}")
        top.geometry("760x600")
        top.minsize(400, 300)
        top.configure(bg=BG_DARK)
        self._toplevel = top

        # Заголовок в Toplevel
        hdr = tk.Frame(top, bg=BG_PANEL, height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"⊞  {self.title}", bg=BG_PANEL,
                 fg=self.title_color, font=FONT_BODY).pack(side=tk.LEFT, padx=12, pady=6)

        # Кнопки в Toplevel
        btn_cfg = dict(bg=BG_CARD, fg=TEXT_DIM, activebackground=BG_DARK,
                       activeforeground=ACCENT, relief=tk.FLAT,
                       font=("Consolas", 8), cursor="hand2", padx=5, pady=2)

        def reattach():
            top.destroy()

        tk.Button(hdr, text="⊠ Закрепить обратно",
                  command=reattach, **btn_cfg).pack(side=tk.RIGHT, padx=6, pady=4)

        tk.Button(hdr, text="🔍 1:1",
                  command=lambda: self._detached_canvas.reset_zoom(),
                  **btn_cfg).pack(side=tk.RIGHT, padx=(2, 0), pady=4)

        tk.Frame(top, bg=BORDER, height=1).pack(fill=tk.X)

        # Канвас в Toplevel
        self._detached_canvas = ZoomableCanvas(top, bg=BG_CARD, highlightthickness=0)
        self._detached_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Скопировать текущее изображение
        if self._bgr_img is not None:
            self._detached_canvas.show(self._bgr_img)

        # Перехват закрытия окна
        top.protocol("WM_DELETE_WINDOW", self._on_toplevel_close)
        top.bind("<Destroy>", lambda e: self._on_toplevel_destroy(e))

    def _on_toplevel_close(self):
        """Пользователь нажал X на Toplevel."""
        if self._toplevel:
            self._toplevel.destroy()

    def _on_toplevel_destroy(self, event):
        """Срабатывает при уничтожении Toplevel (в т.ч. после destroy())."""
        if event.widget is not self._toplevel:
            return
        self._toplevel = None
        # Восстановить основной канвас
        self._placeholder.pack_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._detach_btn.config(text="⊞ Открепить", state=tk.NORMAL)
        # Перерисовать с актуальным изображением
        if self._bgr_img is not None:
            self.canvas.show(self._bgr_img)


# ─────────────────────────────────────────────────────────────────────────────
#  Главное приложение
# ─────────────────────────────────────────────────────────────────────────────

class ImageVisionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        saved_url = load_server_url()
        self.llm = LLMController(base_url=saved_url)
        self.processor = ImageProcessor()
        self.image_loaded = False
        self.processing = False

        self._configure_root()
        self._build_ui()
        self._log("🚀 Приложение запущено. Загрузите изображение.")
        self._log(f"🌐 Сервер: {saved_url}", "accent")

    def _configure_root(self):
        self.root.title("ImageVision AI — CV + LLM")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)
        self.root.configure(bg=BG_DARK)

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_PANEL, height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="⬡  ImageVision AI", bg=BG_PANEL,
                 fg=ACCENT, font=("Consolas", 16, "bold"), pady=10).pack(side=tk.LEFT, padx=20)
        tk.Label(header, text="Natural Language → Qwen → OpenCV",
                 bg=BG_PANEL, fg=TEXT_SECONDARY, font=FONT_SMALL).pack(side=tk.LEFT, padx=10)

        self._status_dot = tk.Label(header, text="●", bg=BG_PANEL, fg=SUCCESS, font=("Consolas", 14))
        self._status_dot.pack(side=tk.RIGHT, padx=8)
        tk.Label(header, text="LM Studio", bg=BG_PANEL, fg=TEXT_SECONDARY, font=FONT_SMALL).pack(side=tk.RIGHT)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg=BG_PANEL, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self._build_left_panel(left)

        center = tk.Frame(main, bg=BG_DARK)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._build_image_area(center)

        right = tk.Frame(main, bg=BG_PANEL, width=290)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        # ── Сервер ──
        self._section_label(parent, "СЕРВЕР LM STUDIO")

        server_frame = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        server_frame.pack(fill=tk.X, padx=14, pady=(0, 4))

        self.server_var = tk.StringVar(value=self.llm.base_url)
        server_entry = tk.Entry(
            server_frame, textvariable=self.server_var,
            bg=BG_CARD, fg=ACCENT, insertbackground=ACCENT,
            relief=tk.FLAT, font=FONT_SMALL, width=32
        )
        server_entry.pack(fill=tk.X, padx=6, pady=6)
        server_entry.bind("<Return>", lambda e: self._apply_server())

        self._btn(parent, "🔗  Применить адрес", self._apply_server, color=ACCENT)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=10)

        # ── Загрузка ──
        self._section_label(parent, "ИЗОБРАЖЕНИЕ")
        self._btn(parent, "📂  Загрузить изображение", self._load_image, color=ACCENT)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=10)

        # ── Команда ──
        self._section_label(parent, "КОМАНДА (на русском)")

        entry_frame = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        entry_frame.pack(fill=tk.X, padx=14, pady=(0, 8))

        self.cmd_var = tk.StringVar()
        self.cmd_entry = tk.Entry(
            entry_frame, textvariable=self.cmd_var,
            bg=BG_CARD, fg=TEXT_PRIMARY, insertbackground=ACCENT,
            relief=tk.FLAT, font=FONT_BODY, width=28
        )
        self.cmd_entry.pack(fill=tk.X, padx=6, pady=6)
        self.cmd_entry.bind("<Return>", lambda e: self._execute_command())

        self._btn(parent, "▶  Выполнить команду", self._execute_command, color=SUCCESS)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=10)

        # ── Быстрые команды ──
        self._section_label(parent, "БЫСТРЫЕ КОМАНДЫ")
        quick = [
            "Сделай чёрно-белым",
            "Поверни на 90 градусов",
            "Найди границы объектов",
            "Размой изображение",
            "Инвертируй цвета",
            "Увеличь резкость",
            "Выдели красный канал",
        ]
        for label in quick:
            self._quick_btn(parent, label)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=10)

        self._btn(parent, "↺  Сбросить к оригиналу", self._reset_image, color=WARNING)
        self._btn(parent, "💾  Сохранить результат",  self._save_image,  color=TEXT_SECONDARY)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=10)
        self._section_label(parent, "ОТВЕТ НЕЙРОСЕТИ (JSON)")

        json_frame = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        json_frame.pack(fill=tk.X, padx=14, pady=(0, 8))

        self.json_text = tk.Text(
            json_frame, height=5, bg=BG_CARD, fg=ACCENT,
            font=FONT_MONO, relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD, padx=6, pady=6
        )
        self.json_text.pack(fill=tk.X)

    def _build_image_area(self, parent):
        images_frame = tk.Frame(parent, bg=BG_DARK)
        images_frame.pack(fill=tk.BOTH, expand=True)

        # ── Используем CanvasPanel вместо сырых tk.Canvas ──
        self.orig_panel = CanvasPanel(images_frame, "ИСХОДНОЕ",  title_color=TEXT_SECONDARY)
        self.orig_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.res_panel  = CanvasPanel(images_frame, "РЕЗУЛЬТАТ", title_color=SUCCESS)
        self.res_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Прогресс-бар
        self.progress = ttk.Progressbar(parent, mode="indeterminate", length=400)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BG_CARD, background=ACCENT, thickness=4)

    # ── Виджеты ──────────────────────────────────────────────────────────────

    def _section_label(self, parent, text):
        bg = parent.cget("bg")
        tk.Label(parent, text=text, bg=bg, fg=TEXT_DIM,
                 font=("Consolas", 8, "bold"), anchor="w").pack(fill=tk.X, padx=14, pady=(10, 4))

    def _btn(self, parent, text, command, color=ACCENT):
        btn = tk.Button(
            parent, text=text, command=command,
            bg=BG_CARD, fg=color, activebackground=BG_DARK,
            activeforeground=ACCENT_HOVER, relief=tk.FLAT,
            font=FONT_BODY, cursor="hand2", padx=10, pady=8, anchor="w"
        )
        btn.pack(fill=tk.X, padx=14, pady=2)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_DARK))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_CARD))
        return btn

    def _quick_btn(self, parent, label):
        def _run():
            self.cmd_var.set(label)
            self._execute_command()
        btn = tk.Button(
            parent, text=f"  {label}", command=_run,
            bg=BG_DARK, fg=TEXT_SECONDARY, activebackground=BG_CARD,
            activeforeground=TEXT_PRIMARY, relief=tk.FLAT,
            font=FONT_SMALL, cursor="hand2", padx=8, pady=5, anchor="w"
        )
        btn.pack(fill=tk.X, padx=14, pady=1)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_CARD, fg=TEXT_PRIMARY))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_DARK, fg=TEXT_SECONDARY))

    # ── Лог ──────────────────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{ts}] ", "time")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_json_display(self, text: str):
        self.json_text.configure(state=tk.NORMAL)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, text)
        self.json_text.configure(state=tk.DISABLED)

    # ── Правая панель ─────────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        self._section_label(parent, "ЖУРНАЛ ДЕЙСТВИЙ")

        log_frame = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        scrollbar = tk.Scrollbar(log_frame, bg=BG_DARK, troughcolor=BG_DARK,
                                  activebackground=ACCENT, width=8)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_frame, bg=BG_CARD, fg=TEXT_SECONDARY,
            font=FONT_LOG, relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD, padx=8, pady=8,
            yscrollcommand=scrollbar.set
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        self.log_text.tag_config("success", foreground=SUCCESS)
        self.log_text.tag_config("error",   foreground=ERROR)
        self.log_text.tag_config("warning", foreground=WARNING)
        self.log_text.tag_config("info",    foreground=TEXT_SECONDARY)
        self.log_text.tag_config("accent",  foreground=ACCENT)
        self.log_text.tag_config("time",    foreground=TEXT_DIM)

        self._btn(parent, "🗑  Очистить журнал", self._clear_log, color=TEXT_DIM)

    # ── Изображения ──────────────────────────────────────────────────────────

    def _show_on_panel(self, panel: CanvasPanel, bgr_img: np.ndarray):
        """Обёртка для отображения изображения на панели."""
        panel.show(bgr_img)

    # ── Действия ─────────────────────────────────────────────────────────────

    def _apply_server(self):
        url = self.server_var.get().strip()
        if not url:
            return
        if not url.startswith("http"):
            url = "http://" + url
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        self.server_var.set(url)
        self.llm.update_server(url)
        save_server_url(url)
        self._log(f"🌐 Сервер: {url}", "accent")

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            img = self.processor.load(path)
            self.image_loaded = True
            self.orig_panel.show(img)
            self.res_panel.show(img)
            name = path.split("/")[-1].split("\\")[-1]
            self._log(f"📁 Загружено: {name}", "success")
            self._log(f"   Размер: {img.shape[1]}×{img.shape[0]} px")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self._log(f"❌ {e}", "error")

    def _execute_command(self):
        if not self.image_loaded:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение.")
            return
        text = self.cmd_var.get().strip()
        if not text:
            messagebox.showwarning("Внимание", "Введите команду.")
            return
        if self.processing:
            return
        self.processing = True
        self._log(f"💬 Команда: «{text}»", "accent")
        self.progress.pack(fill=tk.X, padx=10, pady=4)
        self.progress.start(12)
        threading.Thread(target=self._process_in_thread, args=(text,), daemon=True).start()

    def _process_in_thread(self, text: str):
        try:
            self._log("🤖 Отправка в Qwen...", "info")
            command = self.llm.send_request(text)
            json_str = json.dumps(command, ensure_ascii=False, indent=2)
            self.root.after(0, self._set_json_display, json_str)
            self._log(f"📡 JSON: {json_str.strip()}", "accent")

            action = command.get("action", "unknown")
            if action == "unknown":
                self._log("⚠️ Команда не распознана LLM.", "warning")
                return

            self._log(f"⚙️ OpenCV → {action}", "info")
            result_img, status_msg = self.processor.execute(command)
            self.root.after(0, self._update_result, result_img, status_msg)

        except ConnectionError as e:
            self.root.after(0, messagebox.showerror, "Ошибка подключения", str(e))
            self._log(f"❌ Сервер недоступен", "error")
        except Exception as e:
            self._log(f"❌ Ошибка: {e}", "error")
            logger.exception("Ошибка обработки команды")
        finally:
            self.root.after(0, self._stop_progress)

    def _update_result(self, img: np.ndarray, status: str):
        if img is not None:
            self.res_panel.show(img)
        self._log(status, "success" if "✅" in status else "warning")

    def _stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.processing = False

    def _reset_image(self):
        if not self.image_loaded:
            return
        try:
            img = self.processor.reset()
            self.res_panel.show(img)
            self._log("↺ Сброшено к оригиналу.", "warning")
            self._set_json_display("")
        except Exception as e:
            self._log(f"❌ {e}", "error")

    def _save_image(self):
        if not self.image_loaded:
            messagebox.showwarning("Внимание", "Нет изображения для сохранения.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if not path:
            return
        try:
            saved = self.processor.save(path)
            self._log(f"💾 Сохранено: {saved}", "success")
            messagebox.showinfo("Готово", f"Файл сохранён:\n{saved}")
        except Exception as e:
            self._log(f"❌ {e}", "error")
            messagebox.showerror("Ошибка", str(e))
