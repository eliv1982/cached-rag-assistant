"""Общие фикстуры. Все тесты офлайн: OpenAI и ChromaDB заменены заглушками."""

import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import rag


def make_reply(text):
    """Заглушка ответа chat.completions.create с текстом `text`."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Тесты не зависят от окружения разработчика и не могут выйти в сеть."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_SOURCE_FILTER", raising=False)

    def blocked(*args, **kwargs):
        raise RuntimeError("Сетевые соединения в тестах запрещены")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def fake_store():
    """Векторное хранилище-заглушка: search() возвращает один фрагмент."""
    store = MagicMock()
    store.search.return_value = [("Python создан Гвидо ван Россумом.", "Python Основы", 0.42)]
    return store


@pytest.fixture
def assistant(fake_store, monkeypatch):
    """Настоящий RAGAssistant с заглушкой вместо клиента OpenAI."""
    monkeypatch.setattr(rag, "OpenAI", MagicMock())
    rag_assistant = rag.RAGAssistant(embedding_store=fake_store, api_key="test-key")
    rag_assistant.client.chat.completions.create.return_value = make_reply("Ответ модели")
    return rag_assistant
