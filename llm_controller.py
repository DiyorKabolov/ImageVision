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
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = "local-model"
        self.last_raw_response = ""
        self.last_command = {}

    def update_server(self, base_url: str):
        """Пересоздаёт клиент с новым адресом сервера."""
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)
        logger.info(f"Адрес сервера обновлён: {base_url}")

    def send_request(self, user_text: str) -> dict:
        logger.info(f"Отправка запроса в LLM: '{user_text}'")
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
            raw = response.choices[0].message.content.strip()
            self.last_raw_response = raw
            logger.info(f"Сырой ответ LLM: {raw}")
            command = self._extract_json(raw)
            command = self._validate_command(command)
            self.last_command = command
            return command
        except Exception as e:
            logger.error(f"Ошибка при обращении к LLM: {e}")
            raise ConnectionError(f"Не удалось подключиться к серверу:\n{self.base_url}\n\n{e}")

    def _extract_json(self, text: str) -> dict:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"Не удалось разобрать JSON из ответа: '{text}'")
        return {"action": "unknown"}

    def _validate_command(self, command: dict) -> dict:
        if not isinstance(command, dict):
            return {"action": "unknown"}
        action = command.get("action", "unknown")
        if action not in self.VALID_ACTIONS:
            logger.warning(f"Неизвестное действие от LLM: '{action}'")
            command["action"] = "unknown"
        return command
