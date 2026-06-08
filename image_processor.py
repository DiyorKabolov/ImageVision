import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Выполняет операции над изображениями с помощью OpenCV."""

    def __init__(self):
        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.save_path: Path = Path("result.png")

    # ──────────────────────────────────────────────
    # Загрузка / выгрузка
    # ──────────────────────────────────────────────

    def load(self, path: str) -> np.ndarray:
        """Загружает изображение с диска (поддерживает пути с кириллицей)."""
        try:
            with open(path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {path}: {e}")
            img = None

        if img is None:
            raise FileNotFoundError(f"Не удалось открыть файл: {path}")
        self.original = img.copy()
        self.current = img.copy()
        self.save_path = Path(path).parent / ("result_" + Path(path).name)
        logger.info(f"Изображение загружено: {path}, размер: {img.shape}")
        return self.current

    def reset(self) -> np.ndarray:
        """Возвращает оригинал."""
        if self.original is None:
            raise RuntimeError("Изображение не загружено.")
        self.current = self.original.copy()
        return self.current

    def _require_image(self):
        if self.current is None:
            raise RuntimeError("Сначала загрузите изображение.")

    # ──────────────────────────────────────────────
    # Повороты
    # ──────────────────────────────────────────────

    def rotate_90(self) -> np.ndarray:
        """Поворачивает изображение на 90° по часовой стрелке."""
        self._require_image()
        self.current = cv2.rotate(self.current, cv2.ROTATE_90_CLOCKWISE)
        return self.current

    def rotate_180(self) -> np.ndarray:
        """Поворачивает изображение на 180°."""
        self._require_image()
        self.current = cv2.rotate(self.current, cv2.ROTATE_180)
        return self.current

    def rotate_270(self) -> np.ndarray:
        """Поворачивает изображение на 270° (или 90° против часовой)."""
        self._require_image()
        self.current = cv2.rotate(self.current, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.current

    # ──────────────────────────────────────────────
    # Отражения
    # ──────────────────────────────────────────────

    def flip_horizontal(self) -> np.ndarray:
        """Отражает по горизонтали (зеркало слева–направо)."""
        self._require_image()
        self.current = cv2.flip(self.current, 1)
        return self.current

    def flip_vertical(self) -> np.ndarray:
        """Отражает по вертикали (зеркало сверху–вниз)."""
        self._require_image()
        self.current = cv2.flip(self.current, 0)
        return self.current

    # ──────────────────────────────────────────────
    # Цветовые преобразования
    # ──────────────────────────────────────────────

    def grayscale(self) -> np.ndarray:
        """Переводит изображение в оттенки серого."""
        self._require_image()
        gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        self.current = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self.current

    def extract_red(self) -> np.ndarray:
        """Оставляет только красный канал (BGR → только B=0, G=0)."""
        self._require_image()
        result = np.zeros_like(self.current)
        result[:, :, 2] = self.current[:, :, 2]  # RED канал в BGR — индекс 2
        self.current = result
        return self.current

    def extract_green(self) -> np.ndarray:
        """Оставляет только зелёный канал."""
        self._require_image()
        result = np.zeros_like(self.current)
        result[:, :, 1] = self.current[:, :, 1]  # GREEN канал — индекс 1
        self.current = result
        return self.current

    def extract_blue(self) -> np.ndarray:
        """Оставляет только синий канал."""
        self._require_image()
        result = np.zeros_like(self.current)
        result[:, :, 0] = self.current[:, :, 0]  # BLUE канал — индекс 0
        self.current = result
        return self.current

    def invert(self) -> np.ndarray:
        """Инвертирует цвета (негатив)."""
        self._require_image()
        self.current = cv2.bitwise_not(self.current)
        return self.current

    # ──────────────────────────────────────────────
    # Масштабирование
    # ──────────────────────────────────────────────

    def resize_up(self) -> np.ndarray:
        """Увеличивает изображение в 2 раза."""
        self._require_image()
        h, w = self.current.shape[:2]
        self.current = cv2.resize(self.current, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        return self.current

    def resize_down(self) -> np.ndarray:
        """Уменьшает изображение в 2 раза."""
        self._require_image()
        h, w = self.current.shape[:2]
        self.current = cv2.resize(self.current, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        return self.current

    # ──────────────────────────────────────────────
    # Фильтры
    # ──────────────────────────────────────────────

    def blur(self) -> np.ndarray:
        """Применяет Гауссово размытие."""
        self._require_image()
        self.current = cv2.GaussianBlur(self.current, (15, 15), 0)
        return self.current

    def brightness(self) -> np.ndarray:
        """Увеличивает яркость изображения."""
        self._require_image()
        self.current = cv2.convertScaleAbs(self.current, alpha=1.0, beta=60)
        return self.current

    def contrast(self) -> np.ndarray:
        """Увеличивает контраст изображения."""
        self._require_image()
        self.current = cv2.convertScaleAbs(self.current, alpha=1.8, beta=0)
        return self.current

    def edges(self) -> np.ndarray:
        """Находит границы объектов (Canny edge detection)."""
        self._require_image()
        gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, threshold1=50, threshold2=150)
        self.current = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        return self.current

    def sharpen(self) -> np.ndarray:
        """Увеличивает резкость изображения."""
        self._require_image()
        kernel = np.array([
            [0, -1,  0],
            [-1,  5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        self.current = cv2.filter2D(self.current, -1, kernel)
        return self.current

    # ──────────────────────────────────────────────
    # Сохранение
    # ──────────────────────────────────────────────

    def save(self, path: str | None = None) -> str:
        """Сохраняет текущее изображение на диск (поддерживает пути с кириллицей)."""
        self._require_image()
        out = Path(path) if path else self.save_path
        
        # Получаем расширение для кодирования (например, '.png')
        ext = out.suffix
        if not ext:
            ext = ".png"

        success, encoded_img = cv2.imencode(ext, self.current)
        if not success:
            raise RuntimeError(f"Не удалось закодировать изображение в формат {ext}")

        with open(out, "wb") as f:
            f.write(encoded_img)

        logger.info(f"Изображение сохранено: {out}")
        return str(out)

    # ──────────────────────────────────────────────
    # Диспетчер команд
    # ──────────────────────────────────────────────

    ACTION_MAP = {
        "rotate_90":  "rotate_90",
        "rotate_180": "rotate_180",
        "rotate_270": "rotate_270",
        "flip_h":     "flip_horizontal",
        "flip_v":     "flip_vertical",
        "grayscale":  "grayscale",
        "red":        "extract_red",
        "green":      "extract_green",
        "blue":       "extract_blue",
        "resize_up":  "resize_up",
        "resize_down":"resize_down",
        "blur":       "blur",
        "brightness": "brightness",
        "contrast":   "contrast",
        "edges":      "edges",
        "invert":     "invert",
        "sharpen":    "sharpen",
        "save":       "save",
    }

    def execute(self, command: dict) -> tuple[np.ndarray | None, str]:
        """
        Выполняет действие из команды LLM.
        Возвращает (изображение | None, описание результата).
        """
        action = command.get("action", "unknown")

        if action == "unknown":
            return self.current, "⚠️ Команда не распознана. Попробуйте перефразировать."

        method_name = self.ACTION_MAP.get(action)
        if method_name is None:
            return self.current, f"⚠️ Неизвестное действие: '{action}'"

        method = getattr(self, method_name, None)
        if method is None:
            return self.current, f"⚠️ Метод '{method_name}' не найден."

        if action == "save":
            path = method()
            return self.current, f"✅ Сохранено: {path}"
        else:
            result = method()
            return result, f"✅ Выполнено: {action}"
