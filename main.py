"""
ImageVision AI
==============
Управление обработкой изображений с помощью естественного языка.

Архитектура: Пользователь → Текст → Qwen 3.5 9B (LM Studio) → JSON → OpenCV → Результат

Запуск:
    python main.py
"""

import tkinter as tk
import logging
import sys

# Reconfigure stdout/stderr to use UTF-8 to prevent encoding crashes on Windows
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except Exception:
            pass

# ─── Настройка логирования ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("imagevision.log", encoding="utf-8"),
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Запуск ImageVision AI")

    root = tk.Tk()

    # Устанавливаем иконку окна
    try:
        import os, sys
        # Поддержка PyInstaller (frozen) и обычного запуска
        base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(base, 'icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass

    # Импортируем GUI после настройки логирования
    from gui import ImageVisionApp  # noqa: E402
    app = ImageVisionApp(root)  # noqa: F841

    root.mainloop()
    logger.info("Приложение закрыто")


if __name__ == "__main__":
    main()
