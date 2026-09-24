from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import embeddings
import main
import rag
from cache import ResponseCache
from conftest import make_reply
from rag import NO_CONTEXT_MESSAGE


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(str(tmp_path / "cache.json"))


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Не читаем реальный .env и не пишем в репозиторий: работаем во временной папке."""
    monkeypatch.setattr(main, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def clients(monkeypatch):
    """Заглушки конструкторов ChromaDB и OpenAI (в embeddings и в rag)."""
    chroma_client = MagicMock()
    chroma_client.get_or_create_collection.return_value.count.return_value = 5
    mocks = SimpleNamespace(
        chroma=MagicMock(return_value=chroma_client),
        embeddings_openai=MagicMock(),
        rag_openai=MagicMock(),
    )
    monkeypatch.setattr(embeddings.chromadb, "PersistentClient", mocks.chroma)
    monkeypatch.setattr(embeddings, "OpenAI", mocks.embeddings_openai)
    monkeypatch.setattr(rag, "OpenAI", mocks.rag_openai)
    return mocks


class FakeApiError(Exception):
    status_code = 429


# --- Запуск и конфигурация -----------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key_fails_before_any_initialization(monkeypatch, sandbox, clients, tmp_path, value):
    if value is not None:
        monkeypatch.setenv("OPENAI_API_KEY", value)

    with pytest.raises(main.ConfigError, match="OPENAI_API_KEY"):
        main.initialize_system()

    clients.chroma.assert_not_called()
    clients.embeddings_openai.assert_not_called()
    clients.rag_openai.assert_not_called()
    assert list(tmp_path.iterdir()) == []  # ни cache.json, ни chroma_db не созданы


def test_main_exits_with_error_code_when_key_is_missing(sandbox, clients, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out
    clients.chroma.assert_not_called()


class FakeAuthError(Exception):
    status_code = 401


@pytest.mark.parametrize(
    "error, expected",
    [
        (FakeAuthError("Incorrect API key provided: sk-test-secret"), "(FakeAuthError, HTTP 401)"),
        (RuntimeError("Incorrect API key provided: sk-test-secret"), "(RuntimeError)"),
    ],
)
def test_main_startup_failure_does_not_leak_exception_details(
    monkeypatch, sandbox, clients, capsys, error, expected
):
    """Реальный путь main() -> initialize_system() -> первый запрос эмбеддингов."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clients.chroma.return_value.get_or_create_collection.return_value.count.return_value = 0
    clients.embeddings_openai.return_value.embeddings.create.side_effect = error

    main.main()  # неожиданные ошибки не пробрасываются наружу

    captured = capsys.readouterr()
    assert "Критическая ошибка " + expected in captured.out
    assert "sk-test-secret" not in captured.out
    assert "sk-test-secret" not in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_interactive_loop_unexpected_error_does_not_leak_exception_details(
    monkeypatch, assistant, cache, capsys
):
    """Сбой вне generate_response (здесь — чтение кеша) не должен печатать текст исключения."""
    monkeypatch.setattr(
        cache, "get", MagicMock(side_effect=FakeAuthError("Incorrect API key provided: sk-test-secret"))
    )
    answers = iter(["вопрос", "exit"])
    monkeypatch.setattr("builtins.input", lambda *args: next(answers))

    main.interactive_mode(assistant, cache)  # цикл переживает ошибку и доходит до выхода

    captured = capsys.readouterr()
    assert "Ошибка (FakeAuthError, HTTP 401)" in captured.out
    assert "До свидания" in captured.out
    assert "sk-test-secret" not in captured.out
    assert "sk-test-secret" not in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_initialize_system_uses_documented_defaults(monkeypatch, sandbox, clients):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    store, assistant, _ = main.initialize_system()

    assert assistant.model == "gpt-4o-mini"
    assert store.embedding_model == "text-embedding-3-small"
    clients.rag_openai.assert_called_once_with(api_key="test-key")


def test_source_filter_from_env(monkeypatch):
    assert main.source_filter_from_env() is None
    monkeypatch.setenv("RAG_SOURCE_FILTER", "  Python Основы ")
    assert main.source_filter_from_env() == "Python Основы"
    monkeypatch.setenv("RAG_SOURCE_FILTER", "   ")
    assert main.source_filter_from_env() is None


# --- answer_question: кеш, фильтр, ошибки --------------------------------

def test_offline_end_to_end_cache_flow(assistant, fake_store, cache):
    create = assistant.client.chat.completions.create

    # 1. промах кеша -> поиск -> ответ LLM -> запись в кеш
    first = main.answer_question("Что такое Python?", assistant, cache)
    assert first == "Ответ модели"
    assert (fake_store.search.call_count, create.call_count, cache.size()) == (1, 1, 1)

    # 2. тот же вопрос в том же контексте -> попадание в кеш, ни поиска, ни LLM
    again = main.answer_question("  что такое   python? ", assistant, cache)
    assert again == first
    assert (fake_store.search.call_count, create.call_count, cache.size()) == (1, 1, 1)

    # 3. другой source_filter -> промах, фильтр доходит до поиска
    main.answer_question("Что такое Python?", assistant, cache, source_filter="Python Основы")
    assert (fake_store.search.call_count, create.call_count, cache.size()) == (2, 2, 2)
    fake_store.search.assert_called_with("Что такое Python?", top_k=3, source="Python Основы")

    # 4. повтор с тем же фильтром -> попадание
    main.answer_question("Что такое Python?", assistant, cache, source_filter="Python Основы")
    assert (fake_store.search.call_count, create.call_count) == (2, 2)

    # 5. другая модель -> промах
    assistant.model = "gpt-4o"
    main.answer_question("Что такое Python?", assistant, cache)
    assert (fake_store.search.call_count, create.call_count, cache.size()) == (3, 3, 3)

    # 6. кеш пережил перезапуск
    assert ResponseCache(str(cache.cache_file)).size() == 3


def test_generation_failure_is_not_cached(assistant, cache):
    create = assistant.client.chat.completions.create
    create.side_effect = [FakeApiError("boom, key sk-secret-123"), make_reply("Настоящий ответ")]

    failed = main.answer_question("вопрос", assistant, cache)

    assert "FakeApiError" in failed and "HTTP 429" in failed
    assert "sk-secret-123" not in failed  # текст исключения не показываем
    assert cache.size() == 0
    assert not cache.cache_file.exists()

    # временный сбой: повтор снова идёт в LLM (а не в закешированную ошибку) и кешируется
    retried = main.answer_question("вопрос", assistant, cache)
    assert retried == "Настоящий ответ"
    assert create.call_count == 2
    assert cache.size() == 1


def test_search_failure_is_not_cached(assistant, fake_store, cache):
    fake_store.search.side_effect = RuntimeError("chroma недоступна")

    main.answer_question("вопрос", assistant, cache)

    assistant.client.chat.completions.create.assert_not_called()
    assert cache.size() == 0


def test_no_retrieval_results_skips_llm_and_is_not_cached(assistant, fake_store, cache):
    fake_store.search.return_value = []

    answer = main.answer_question("вопрос", assistant, cache)

    assert answer == NO_CONTEXT_MESSAGE
    assistant.client.chat.completions.create.assert_not_called()
    assert cache.size() == 0
