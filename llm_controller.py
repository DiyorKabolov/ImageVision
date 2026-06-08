import json
import re
import logging
from openai import OpenAI
from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMController:
    """Управляет взаимодействием с локальной LLM через LM Studio."""

    VALID_ACTIONS = {
        "rotate_90", "rotate_180", "rotate_270",
        "flip_h", "flip_v",
        "grayscale",
        "red", "green", "blue",
        "resize_up", "resize_down",
        "blur", "brightness", "contrast",
        "edges", "invert", "sharpen",
        "save", "unknown"
    }

    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = "local-model"  # LM Studio игнорирует имя, используется загруженная модель
        self.last_raw_response = ""
        self.last_command = {}

    def send_request(self, user_text: str) -> dict:
        """Отправляет текст пользователя в LLM и возвращает распознанную команду."""
        logger.info(f"Отправка запроса в LLM: '{user_text}'")
        
        # 1. Проверяем доступность сервера и загрузку моделей
        try:
            models_response = self.client.models.list()
            loaded_models = [m.id for m in models_response.data]
            if not loaded_models:
                error_msg = "Сервер LM Studio работает, но ни одна модель не загружена. Пожалуйста, выберите и загрузите модель в верхней панели LM Studio."
                logger.error(error_msg)
                raise ConnectionError(error_msg)
            logger.info(f"Текущие загруженные модели: {loaded_models}")
        except ConnectionError as ce:
            raise ce
        except Exception as e:
            # Если не удалось получить список моделей, значит сервер вообще не отвечает/не запущен
            error_msg = f"Сервер LM Studio не запущен на {self.client.base_url} или не отвечает. Убедитесь, что вы запустили локальный сервер (Start Server) в LM Studio на порту 1234."
            logger.error(f"{error_msg} Детали: {e}")
            raise ConnectionError(error_msg)

        # 2. Отправляем запрос к LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.1,
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            if raw is None:
                raw = ""
            raw = raw.strip()
            self.last_raw_response = raw
            logger.info(f"Сырой ответ LLM: {raw}")
            
            if not raw:
                logger.warning("Получен пустой ответ от LLM.")
                return {"action": "unknown"}
                
            command = self._extract_json(raw)
            command = self._validate_command(command)
            self.last_command = command
            return command
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа LLM: {e}")
            raise ConnectionError(f"Ошибка при общении с моделью в LM Studio:\n{e}")

    def _extract_json(self, text: str) -> dict:
        """Извлекает JSON из ответа модели, защищаясь от «мусора»."""
        # Убираем markdown-блоки если модель всё же их добавила
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

        # Пробуем напрямую
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Ищем JSON-объект от первой фигурной скобки до последней
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Не удалось разобрать JSON из ответа: '{text}'")
        return {"action": "unknown"}

    def _validate_command(self, command: dict) -> dict:
        """Проверяет, что команда содержит допустимое действие."""
        if not isinstance(command, dict):
            return {"action": "unknown"}

        action = command.get("action", "unknown")
        if action not in self.VALID_ACTIONS:
            logger.warning(f"Неизвестное действие от LLM: '{action}'")
            command["action"] = "unknown"

        return command
