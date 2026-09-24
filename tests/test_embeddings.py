from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import embeddings
from embeddings import EmbeddingStore


@pytest.fixture
def collection(monkeypatch):
    """Заглушка коллекции Chroma; ни ChromaDB, ни OpenAI не создаются по-настоящему."""
    client = MagicMock()
    monkeypatch.setattr(embeddings.chromadb, "PersistentClient", MagicMock(return_value=client))
    monkeypatch.setattr(embeddings, "OpenAI", MagicMock())

    col = client.get_or_create_collection.return_value
    col.count.return_value = 5
    col.query.return_value = {
        "documents": [["фрагмент A", "фрагмент B"]],
        "metadatas": [[{"source": "Python Основы"}, {"source": "Векторные базы данных"}]],
        "distances": [[0.3, 0.9]],
    }
    return col


@pytest.fixture
def store(collection, tmp_path):
    s = EmbeddingStore(persist_directory=str(tmp_path / "chroma_db"), api_key="test-key")
    s.openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
    )
    return s


class LoopGuard(str):
    """Строка, которая падает, если её режут слишком много раз (защита от зависания)."""

    def __getitem__(self, item):
        self.slices = getattr(self, "slices", 0) + 1
        if self.slices > 1000:
            raise AssertionError("_create_chunks не завершается")
        return super().__getitem__(item)


# --- Чанки ---------------------------------------------------------------

@pytest.mark.parametrize(
    "chunk_size, overlap",
    [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 11)],
)
def test_create_chunks_rejects_invalid_parameters(store, chunk_size, overlap):
    with pytest.raises(ValueError):
        store._create_chunks("текст", chunk_size=chunk_size, overlap=overlap)


def test_overlap_not_smaller_than_chunk_size_cannot_loop_forever(store):
    # Без проверки окно не сдвигалось бы вперёд; LoopGuard превращает зависание в ошибку
    with pytest.raises(ValueError):
        store._create_chunks(LoopGuard("a" * 100), chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        store._create_chunks(LoopGuard("a" * 100), chunk_size=10, overlap=50)


def test_chunks_overlap_by_requested_amount(store):
    chunks = store._create_chunks("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
    for previous, following in zip(chunks, chunks[1:]):
        assert previous[-3:] == following[:3]


def test_chunks_without_overlap_do_not_share_text(store):
    assert store._create_chunks("abcdefghij", chunk_size=4, overlap=0) == ["abcd", "efgh", "ij"]


def test_text_that_fits_in_one_chunk_has_no_duplicate_tail(store):
    assert store._create_chunks("abc", chunk_size=10, overlap=3) == ["abc"]
    assert store._create_chunks("a" * 10, chunk_size=10, overlap=3) == ["a" * 10]


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_text_gives_no_chunks(store, text):
    assert store._create_chunks(text) == []


# --- Поиск ---------------------------------------------------------------

@pytest.mark.parametrize("query", ["", "   ", None])
def test_search_rejects_empty_query(store, collection, query):
    with pytest.raises(ValueError):
        store.search(query)
    store.openai_client.embeddings.create.assert_not_called()
    collection.query.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -3])
def test_search_rejects_non_positive_top_k(store, collection, top_k):
    with pytest.raises(ValueError):
        store.search("вопрос", top_k=top_k)
    store.openai_client.embeddings.create.assert_not_called()
    collection.query.assert_not_called()


def test_search_passes_source_filter_to_chroma(store, collection):
    store.search("вопрос", top_k=2, source="Python Основы")

    kwargs = collection.query.call_args.kwargs
    assert kwargs["where"] == {"source": "Python Основы"}
    assert kwargs["n_results"] == 2


def test_search_without_source_has_no_where_clause(store, collection):
    store.search("вопрос")
    assert "where" not in collection.query.call_args.kwargs


def test_search_limits_n_results_to_collection_size(store, collection):
    collection.count.return_value = 2
    store.search("вопрос", top_k=10)
    assert collection.query.call_args.kwargs["n_results"] == 2


def test_search_returns_text_source_and_raw_distance(store):
    assert store.search("вопрос") == [
        ("фрагмент A", "Python Основы", 0.3),
        ("фрагмент B", "Векторные базы данных", 0.9),
    ]


def test_search_in_empty_collection_returns_nothing_without_api_call(store, collection):
    collection.count.return_value = 0
    assert store.search("вопрос") == []
    store.openai_client.embeddings.create.assert_not_called()
