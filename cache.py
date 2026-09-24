"""
Модуль для кеширования ответов LLM.

Кеш позволяет избежать повторных запросов к LLM для одинаковых вопросов,
что экономит время и деньги на API-запросы.
"""

import hashlib
import json
import os
from typing import Any, Mapping, Optional
from pathlib import Path

# Версия формата ключа. Ключи текущей версии выглядят как "v2:<sha256>".
# Старые записи (голый sha256 от одного лишь запроса) не начинаются с этого
# префикса, поэтому никогда не совпадут с новыми ключами.
# Если поменяется формат ключа или промпт — увеличьте версию.
CACHE_KEY_VERSION = "v2"


class ResponseCache:
    """
    Простой кеш для хранения ответов LLM.

    Использует словарь Python для хранения пар (ключ, ответ).
    Ключ строится из запроса и контекста (фильтр по источнику, модель LLM),
    поэтому один и тот же вопрос при разных условиях кешируется отдельно.
    Кеш сохраняется в JSON-файл и загружается обратно при запуске.
    """

    def __init__(self, cache_file: str = "cache.json"):
        """
        Инициализация кеша.

        Args:
            cache_file: Путь к файлу для сохранения кеша на диск
        """
        self.cache_file = Path(cache_file)
        self.cache = {}

        # Загружаем существующий кеш, если файл есть
        self._load_cache()

    def _get_cache_key(
        self, query: str, context: Optional[Mapping[str, Any]] = None
    ) -> str:
        """
        Создает уникальный ключ (хеш) для запроса в заданном контексте.

        Используем SHA-256 для создания стабильного хеша. Одинаковые запрос
        и контекст всегда дадут одинаковый ключ; любое отличие в запросе
        или в контексте даст другой ключ.

        Args:
            query: Пользовательский запрос
            context: Всё, от чего зависит ответ, кроме самого текста вопроса
                (например, {"source_filter": None, "model": "gpt-4o-mini"}).
                None и отсутствие фильтра различаются: None хранится как null.

        Returns:
            Строка вида "v2:<sha256>" для использования как ключ кеша
        """
        # Нормализуем запрос: убираем лишние пробелы и приводим к нижнему регистру
        normalized_query = " ".join(query.lower().split())

        # sort_keys делает результат независимым от порядка ключей в контексте
        payload = json.dumps(
            {"query": normalized_query, "context": dict(context or {})},
            sort_keys=True,
            ensure_ascii=False,
        )

        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        return f"{CACHE_KEY_VERSION}:{digest}"

    def get(
        self, query: str, context: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        """
        Получает ответ из кеша, если он есть.

        Args:
            query: Пользовательский запрос
            context: Контекст запроса (см. _get_cache_key)

        Returns:
            Закешированный ответ или None, если ответа нет в кеше
        """
        cache_key = self._get_cache_key(query, context)

        if cache_key in self.cache:
            print(f"✓ Найден ответ в кеше для запроса: '{query[:50]}...'")
            return self.cache[cache_key]

        print(f"✗ Ответ не найден в кеше, выполняем RAG поиск...")
        return None

    def set(
        self, query: str, response: str, context: Optional[Mapping[str, Any]] = None
    ) -> None:
        """
        Сохраняет ответ в кеш.

        Args:
            query: Пользовательский запрос
            response: Ответ от LLM
            context: Контекст запроса (должен совпадать с переданным в get)
        """
        cache_key = self._get_cache_key(query, context)
        self.cache[cache_key] = response

        # Автоматически сохраняем кеш на диск после каждого добавления
        self._save_cache()

        print(f"✓ Ответ сохранен в кеше")

    def _save_cache(self) -> None:
        """
        Сохраняет кеш в JSON файл.

        Это позволяет сохранить кеш между запусками программы.
        Пишем во временный файл и подменяем им cache.json одной операцией,
        чтобы прерванная запись не оставила наполовину записанный кеш.
        """
        tmp_file = self.cache_file.with_name(self.cache_file.name + ".tmp")
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, self.cache_file)
        except Exception as e:
            print(f"⚠ Предупреждение: не удалось сохранить кеш: {e}")

    def _load_cache(self) -> None:
        """
        Загружает кеш из JSON файла, если он существует.

        Повреждённый файл не роняет программу — кеш просто начинается пустым.
        Записи без префикса текущей версии ключа (старый формат) пропускаются.
        """
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("ожидался JSON-объект")
        except Exception as e:
            print(f"⚠ Предупреждение: не удалось загрузить кеш: {e}")
            self.cache = {}
            return

        prefix = f"{CACHE_KEY_VERSION}:"
        self.cache = {
            key: value for key, value in data.items()
            if key.startswith(prefix) and isinstance(value, str)
        }
        print(f"✓ Загружен кеш с {len(self.cache)} записями")

        skipped = len(data) - len(self.cache)
        if skipped:
            print(f"  (пропущено {skipped} записей устаревшего или неверного формата)")

    def clear(self) -> None:
        """
        Очищает весь кеш.
        """
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("✓ Кеш очищен")

    def size(self) -> int:
        """
        Возвращает количество записей в кеше.
        """
        return len(self.cache)

