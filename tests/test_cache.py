import hashlib
import json

import pytest

import cache as cache_module
from cache import ResponseCache

CTX = {"source_filter": None, "model": "gpt-4o-mini"}


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(str(tmp_path / "cache.json"))


def test_query_is_normalized(cache):
    cache.set("Что такое Python?", "ответ", CTX)
    assert cache.get("  что   ТАКОЕ python? ", CTX) == "ответ"


def test_same_query_filter_and_model_hit(cache):
    ctx = {"source_filter": "Python Основы", "model": "gpt-4o-mini"}
    cache.set("вопрос", "ответ", ctx)
    # порядок ключей в контексте не важен
    assert cache.get("вопрос", {"model": "gpt-4o-mini", "source_filter": "Python Основы"}) == "ответ"


def test_different_source_filter_misses(cache):
    cache.set("вопрос", "ответ A", {"source_filter": "Python Основы", "model": "m"})
    assert cache.get("вопрос", {"source_filter": "Векторные базы данных", "model": "m"}) is None


def test_filtered_and_unfiltered_are_separate(cache):
    cache.set("вопрос", "по всей базе", {"source_filter": None, "model": "m"})
    assert cache.get("вопрос", {"source_filter": "Python Основы", "model": "m"}) is None

    cache.set("вопрос", "по источнику", {"source_filter": "Python Основы", "model": "m"})
    assert cache.get("вопрос", {"source_filter": None, "model": "m"}) == "по всей базе"
    # строка "None" — не то же самое, что отсутствие фильтра
    assert cache.get("вопрос", {"source_filter": "None", "model": "m"}) is None


def test_different_model_misses(cache):
    cache.set("вопрос", "ответ", {"source_filter": None, "model": "gpt-4o-mini"})
    assert cache.get("вопрос", {"source_filter": None, "model": "gpt-4o"}) is None


def test_legacy_query_only_key_does_not_match(tmp_path):
    path = tmp_path / "cache.json"
    legacy_key = hashlib.sha256("что такое python?".encode("utf-8")).hexdigest()
    path.write_text(json.dumps({legacy_key: "старый ответ"}), encoding="utf-8")

    cache = ResponseCache(str(path))

    assert cache.size() == 0  # запись старого формата не загружена
    assert cache.get("Что такое Python?") is None
    assert cache.get("Что такое Python?", CTX) is None
    assert cache._get_cache_key("Что такое Python?") != legacy_key
    assert cache._get_cache_key("Что такое Python?", CTX) != legacy_key


def test_cache_persists_between_instances(tmp_path):
    path = str(tmp_path / "cache.json")
    ResponseCache(path).set("вопрос", "ответ", CTX)

    reloaded = ResponseCache(path)
    assert reloaded.size() == 1
    assert reloaded.get("вопрос", CTX) == "ответ"


@pytest.mark.parametrize("content", ["{not json", "", '["список"]', '"строка"'])
def test_malformed_cache_file_recovers(tmp_path, content):
    path = tmp_path / "cache.json"
    path.write_text(content, encoding="utf-8")

    cache = ResponseCache(str(path))
    assert cache.size() == 0

    # после восстановления кеш работает и сохраняется
    cache.set("вопрос", "ответ", CTX)
    assert ResponseCache(str(path)).get("вопрос", CTX) == "ответ"


def test_write_leaves_no_temp_file(tmp_path, cache):
    cache.set("вопрос", "ответ", CTX)
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]


def test_interrupted_write_keeps_previous_cache_file(tmp_path, cache, monkeypatch):
    cache.set("первый", "ответ 1", CTX)

    def boom(*args, **kwargs):
        raise OSError("диск отвалился")

    monkeypatch.setattr(cache_module.json, "dump", boom)
    cache.set("второй", "ответ 2", CTX)  # ошибка записи не должна ронять программу
    monkeypatch.undo()

    on_disk = ResponseCache(str(tmp_path / "cache.json"))
    assert on_disk.get("первый", CTX) == "ответ 1"
    assert on_disk.get("второй", CTX) is None


def test_clear_removes_entries_and_file(tmp_path, cache):
    cache.set("вопрос", "ответ", CTX)
    cache.clear()
    assert cache.size() == 0
    assert not (tmp_path / "cache.json").exists()
