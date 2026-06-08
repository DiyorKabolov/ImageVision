import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np

from llm_controller import LLMController
from image_processor import ImageProcessor

logger = logging.getLogger(__name__)

# ─── Цветовая схема ──────────────────────────────────────────────────────────
BG_DARK      = "#0f1117"
BG_PANEL     = "#1a1d27"
BG_CARD      = "#22263a"
ACCENT       = "#5b7fff"
ACCENT_HOVER = "#7a9bff"
TEXT_PRIMARY = "#e8eaf6"
TEXT_SECONDARY = "#8892b0"
TEXT_DIM     = "#4a5568"
SUCCESS      = "#43d98c"
WARNING      = "#f6c90e"
ERROR        = "#ff5f72"
BORDER       = "#2d3353"

FONT_TITLE   = ("Consolas", 13, "bold")
FONT_BODY    = ("Consolas", 10)
FONT_SMALL   = ("Consolas", 9)
FONT_LOG     = ("Consolas", 9)
FONT_MONO    = ("Courier New", 10)


class ImageVisionApp:
    """Главное окно приложения."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.llm = LLMController()
        self.processor = ImageProcessor()
        self.image_loaded = False
        self.processing = False

        self._configure_root()
        self._build_ui()
        self._setup_bindings()
        self._log("🚀 Приложение запущено. Загрузите изображение.")
        self._log("🤖 LLM: Qwen через LM Studio (localhost:1234)")

    # ──────────────────────────────────────────────
    # Конфигурация окна
    # ──────────────────────────────────────────────

    def _configure_root(self):
        self.root.title("ImageVision AI — CV + LLM")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)
        self.root.configure(bg=BG_DARK)
        self.root.option_add("*Font", FONT_BODY)

    def _setup_bindings(self):
        self.root.bind("<Control-o>", lambda e: self._load_image())
        self.root.bind("<Control-O>", lambda e: self._load_image())
        self.root.bind("<Control-s>", lambda e: self._save_image())
        self.root.bind("<Control-S>", lambda e: self._save_image())
        self.root.bind("<Control-r>", lambda e: self._reset_image())
        self.root.bind("<Control-R>", lambda e: self._reset_image())

    # ──────────────────────────────────────────────
    # Построение интерфейса
    # ──────────────────────────────────────────────

    def _build_ui(self):
        # ── Заголовок ──
        header = tk.Frame(self.root, bg=BG_PANEL, height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="⬡  ImageVision AI", bg=BG_PANEL,
            fg=ACCENT, font=("Consolas", 16, "bold"), pady=10
        ).pack(side=tk.LEFT, padx=20)

        tk.Label(
            header, text="Natural Language → Qwen 3.5 9B → OpenCV",
            bg=BG_PANEL, fg=TEXT_SECONDARY, font=FONT_SMALL
        ).pack(side=tk.LEFT, padx=10)

        self._status_dot = tk.Label(header, text="●", bg=BG_PANEL, fg=SUCCESS, font=("Consolas", 14))
        self._status_dot.pack(side=tk.RIGHT, padx=8)
        tk.Label(header, text="LM Studio", bg=BG_PANEL, fg=TEXT_SECONDARY, font=FONT_SMALL).pack(side=tk.RIGHT)

        # ── Разделительная линия ──
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        # ── Основная область ──
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Левая панель: управление
        left = tk.Frame(main, bg=BG_PANEL, width=310)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Центр: изображения
        center = tk.Frame(main, bg=BG_DARK)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._build_image_area(center)

        # Правая панель: журнал
        right = tk.Frame(main, bg=BG_PANEL, width=290)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        # Заголовок панели
        self._section_label(parent, "УПРАВЛЕНИЕ")

        # Кнопка загрузки
        self._btn(parent, "📂  Загрузить изображение  [Ctrl+O]", self._load_image, color=ACCENT)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=12)
        self._section_label(parent, "КОМАНДА (на русском)")

        # Поле ввода команды
        entry_frame = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER,
                               highlightthickness=1, padx=2, pady=2)
        entry_frame.pack(fill=tk.X, padx=14, pady=(0, 8))

        self.cmd_var = tk.StringVar()
        self.cmd_entry = tk.Entry(
            entry_frame, textvariable=self.cmd_var,
            bg=BG_CARD, fg=TEXT_PRIMARY,
            insertbackground=ACCENT,
            relief=tk.FLAT, font=FONT_BODY,
            width=28
        )
        self.cmd_entry.pack(fill=tk.X, padx=6, pady=6)
        self.cmd_entry.bind("<Return>", lambda e: self._execute_command())

        # Кнопка выполнения
        self._btn(parent, "▶  Выполнить команду", self._execute_command, color=SUCCESS)

        # Быстрые команды
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=12)
        self._section_label(parent, "БЫСТРЫЕ КОМАНДЫ")

        quick_cmds = [
            ("Сделай чёрно-белым", "grayscale"),
            ("Поверни на 90°", "rotate"),
            ("Найди границы", "edges"),
            ("Размой изображение", "blur"),
            ("Инвертируй цвета", "invert"),
            ("Увеличь резкость", "sharpen"),
            ("Выдели красный", "red"),
        ]
        for label, cmd in quick_cmds:
            self._quick_btn(parent, label, cmd)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=12)
        self._section_label(parent, "ДЕЙСТВИЯ")

        self._btn(parent, "↺  Сбросить к оригиналу  [Ctrl+R]", self._reset_image, color=WARNING)
        self._btn(parent, "💾  Сохранить результат  [Ctrl+S]", self._save_image, color=TEXT_SECONDARY)

        # JSON-блок
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=12)
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

        # Оригинал
        orig_frame = tk.Frame(images_frame, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(orig_frame, text="ИСХОДНОЕ", bg=BG_PANEL, fg=TEXT_SECONDARY, font=FONT_SMALL).pack(pady=(8, 0))
        self.orig_canvas = tk.Canvas(orig_frame, bg=BG_CARD, highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Результат
        res_frame = tk.Frame(images_frame, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        res_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(res_frame, text="РЕЗУЛЬТАТ", bg=BG_PANEL, fg=SUCCESS, font=FONT_SMALL).pack(pady=(8, 0))
        self.res_canvas = tk.Canvas(res_frame, bg=BG_CARD, highlightthickness=0)
        self.res_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Прогресс-бар
        self.progress = ttk.Progressbar(parent, mode="indeterminate", length=400)

        # Стиль прогресс-бара
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BG_CARD, background=ACCENT, thickness=4)

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

        # Теги для цветов
        self.log_text.tag_config("success", foreground=SUCCESS)
        self.log_text.tag_config("error", foreground=ERROR)
        self.log_text.tag_config("warning", foreground=WARNING)
        self.log_text.tag_config("info", foreground=TEXT_SECONDARY)
        self.log_text.tag_config("accent", foreground=ACCENT)
        self.log_text.tag_config("time", foreground=TEXT_DIM)

        # Кнопка очистки журнала
        self._btn(parent, "🗑  Очистить журнал", self._clear_log, color=TEXT_DIM)

    # ──────────────────────────────────────────────
    # Вспомогательные виджеты
    # ──────────────────────────────────────────────

    def _section_label(self, parent, text):
        tk.Label(
            parent, text=text, bg=BG_PANEL if parent.cget("bg") == str(BG_PANEL) else parent.cget("bg"),
            fg=TEXT_DIM, font=("Consolas", 8, "bold"), anchor="w"
        ).pack(fill=tk.X, padx=14, pady=(10, 4))

    def _btn(self, parent, text, command, color=ACCENT):
        btn = tk.Button(
            parent, text=text, command=command,
            bg=BG_CARD, fg=color, activebackground=BG_DARK,
            activeforeground=ACCENT_HOVER,
            relief=tk.FLAT, font=FONT_BODY,
            cursor="hand2", padx=10, pady=8,
            anchor="w"
        )
        btn.pack(fill=tk.X, padx=14, pady=2)

        def on_enter(e):
            btn.configure(bg=BG_DARK)

        def on_leave(e):
            btn.configure(bg=BG_CARD)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def _quick_btn(self, parent, label, cmd_text):
        def _run():
            self.cmd_var.set(label)
            self._execute_command()

        btn = tk.Button(
            parent, text=f"  {label}", command=_run,
            bg=BG_DARK, fg=TEXT_SECONDARY,
            activebackground=BG_CARD, activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT, font=FONT_SMALL,
            cursor="hand2", padx=8, pady=5, anchor="w"
        )
        btn.pack(fill=tk.X, padx=14, pady=1)

        def on_enter(e):
            btn.configure(bg=BG_CARD, fg=TEXT_PRIMARY)

        def on_leave(e):
            btn.configure(bg=BG_DARK, fg=TEXT_SECONDARY)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # ──────────────────────────────────────────────
    # Логирование
    # ──────────────────────────────────────────────

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

    # ──────────────────────────────────────────────
    # Отображение изображений
    # ──────────────────────────────────────────────

    def _show_image(self, canvas: tk.Canvas, bgr_img: np.ndarray):
        """Конвертирует BGR → RGB → PhotoImage и отображает в canvas."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        canvas.update_idletasks()
        cw = canvas.winfo_width() or 450
        ch = canvas.winfo_height() or 380

        pil.thumbnail((cw, ch), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.image = tk_img  # предотвращаем сборку мусора
        x = cw // 2
        y = ch // 2
        canvas.create_image(x, y, anchor=tk.CENTER, image=tk_img)

    # ──────────────────────────────────────────────
    # Основные действия
    # ──────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("Все файлы", "*.*")
            ]
        )
        if not path:
            return

        try:
            img = self.processor.load(path)
            self.image_loaded = True
            self._show_image(self.orig_canvas, img)
            self._show_image(self.res_canvas, img)
            name = path.split("/")[-1]
            self._log(f"📁 Загружено: {name}", "success")
            self._log(f"   Размер: {img.shape[1]}×{img.shape[0]} px", "info")
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

        thread = threading.Thread(target=self._process_in_thread, args=(text,), daemon=True)
        thread.start()

    def _process_in_thread(self, text: str):
        try:
            # Шаг 1: LLM
            self._log("🤖 Отправка в Qwen...", "info")
            command = self.llm.send_request(text)

            import json
            json_str = json.dumps(command, ensure_ascii=False, indent=2)
            self.root.after(0, self._set_json_display, json_str)
            self._log(f"📡 JSON: {json_str.strip()}", "accent")

            # Шаг 2: OpenCV
            action = command.get("action", "unknown")
            if action == "unknown":
                self._log("⚠️ Команда не распознана LLM.", "warning")
                return

            self._log(f"⚙️ OpenCV → {action}", "info")
            result_img, status_msg = self.processor.execute(command)

            # Шаг 3: Обновление GUI в главном потоке
            self.root.after(0, self._update_result, result_img, status_msg)

        except ConnectionError as e:
            self.root.after(0, messagebox.showerror, "Ошибка подключения", str(e))
            self._log(f"❌ LM Studio недоступен: {e}", "error")
        except Exception as e:
            self._log(f"❌ Ошибка: {e}", "error")
            logger.exception("Ошибка обработки команды")
        finally:
            self.root.after(0, self._stop_progress)

    def _update_result(self, img: np.ndarray, status: str):
        if img is not None:
            self._show_image(self.res_canvas, img)
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
            self._show_image(self.res_canvas, img)
            self._log("↺ Изображение сброшено к оригиналу.", "warning")
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
