# 🤖 RAG-ассистент с ChromaDB и кешированием

**cached-rag-assistant** — минимальный, но полностью рабочий пример ИИ-ассистента с технологией RAG (Retrieval-Augmented Generation), локальным векторным хранилищем ChromaDB и кешированием ответов.

**Репозиторий:** [github.com/eliv1982/cached-rag-assistant](https://github.com/eliv1982/cached-rag-assistant)

*English summary:* Python RAG assistant with response caching, ChromaDB vector search, OpenAI embeddings/LLM, and optional metadata `source` filter via `RAG_SOURCE_FILTER` in `.env`.

## 📋 Описание

Этот проект демонстрирует, как работает RAG-система:

1. **Пользователь задает вопрос**
2. **Проверка кеша** - если ответ уже есть, возвращается из кеша
3. **Поиск релевантных документов** - векторный поиск в ChromaDB
4. **Формирование контекста** - сборка найденных фрагментов
5. **Генерация ответа** - запрос к LLM (OpenAI) с контекстом
6. **Кеширование ответа** - сохранение для будущих запросов
7. **Возврат ответа пользователю**

## 🎯 Особенности

- ✅ **Локальное векторное хранилище** с ChromaDB
- ✅ **OpenAI эмбеддинги** для высокого качества поиска
- ✅ **Семантический поиск** по документам
- ✅ **Кеширование ответов** для экономии API-запросов
- ✅ **Простая архитектура** с понятным кодом
- ✅ **Подробные комментарии** для обучения
- ✅ **Два режима работы**: интерактивный и демонстрационный
- ✅ **Фильтр по источнику** (метаданные Chroma): переменная `RAG_SOURCE_FILTER` в `.env` — поиск только среди чанков с заданным `source`

## 📦 Установка

### 1. Клонирование и переход в директорию

```bash
git clone https://github.com/eliv1982/cached-rag-assistant.git
cd cached-rag-assistant
```

### 2. Создание виртуального окружения (рекомендуется)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка окружения (`.env`)

Скопируйте пример и подставьте ключ:

```bash
# Windows (cmd/PowerShell)
copy env.example .env

# Linux / macOS
cp env.example .env
```

Минимально в `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Опционально** — поиск только по документам с указанным именем источника (должно совпадать с первым элементом кортежа в `get_sample_documents()`, например `Python Основы`):

```env
# RAG_SOURCE_FILTER=Python Основы
```

Закомментируйте или удалите строку, чтобы искать по всей коллекции. Подробнее см. `env.example`.

Получить API ключ: [platform.openai.com](https://platform.openai.com/api-keys)

## 🚀 Запуск

```bash
python main.py
```

При первом запуске программа:
- Создаст локальную базу данных ChromaDB
- Загрузит модель для создания эмбеддингов
- Добавит примеры документов в базу
- Предложит выбрать режим работы

### Режимы работы

**1. Интерактивный режим** (по умолчанию)
- Задавайте свои вопросы ассистенту
- Получайте ответы на основе базы знаний
- Команды: `exit`, `cache`, `clear_cache`, `stats`

**2. Демонстрационный режим**
- Автоматически выполняет заготовленные примеры
- Показывает работу RAG и кеша
- После демонстрации можно перейти в интерактивный режим

## 📁 Структура проекта

```
.
├── main.py           # Точка входа, интерактивный режим
├── rag.py            # Логика RAG (поиск + генерация)
├── cache.py          # Кеширование ответов LLM
├── embeddings.py     # Работа с векторным хранилищем ChromaDB
├── requirements.txt  # Зависимости проекта
├── README.md         # Документация (этот файл)
├── env.example       # Пример переменных окружения
├── .env              # Конфигурация (создается вручную, не коммитится)
├── cache.json        # Файл кеша (создается автоматически)
└── chroma_db/        # База данных ChromaDB (создается автоматически)
```

## 🔧 Архитектура

### Модуль `cache.py`

Реализует простое кеширование на основе хеша запроса:

```python
cache = ResponseCache()

# Проверка кеша
answer = cache.get(query)

# Сохранение в кеш
cache.set(query, answer)
```

**Зачем кеш?**
- Экономия денег на API-запросах
- Мгновенные ответы на повторные вопросы
- Снижение нагрузки на LLM

### Модуль `embeddings.py`

Работа с векторным хранилищем:

```python
store = EmbeddingStore()

# Добавление документов
store.add_documents(documents)

# Поиск релевантных фрагментов (опционально: только metadata source)
results = store.search(query, top_k=5, source="Python Основы")  # source=None — вся коллекция
```

**Используемые технологии:**
- ChromaDB - векторная база данных
- OpenAI Embeddings (text-embedding-3-small) - создание эмбеддингов
- Chunking - разбивка текстов на фрагменты с перекрытием

### Модуль `rag.py`

Основная логика RAG:

```python
assistant = RAGAssistant(embedding_store)

# Генерация ответа с RAG (source_filter=None — без фильтра по источнику)
answer, sources = assistant.generate_response(query, top_k=5, source_filter=None)
```

**Процесс RAG:**
1. Векторный поиск релевантных документов
2. Формирование контекста из найденных фрагментов
3. Создание промпта для LLM
4. Генерация ответа с учетом контекста

### Модуль `main.py`

Оркестрация всех компонентов:

```python
# Инициализация
cache = ResponseCache()
store = EmbeddingStore()
assistant = RAGAssistant(store)

# Ответ на вопрос
cached = cache.get(query)
if not cached:
    answer = assistant.generate_response(query)
    cache.set(query, answer)
```

## 💡 Примеры вопросов

Система содержит знания о:
- Python и его особенностях
- Машинном обучении и RAG
- Векторных базах данных

Примеры вопросов:
- "Что такое Python?"
- "Расскажи про RAG"
- "Для чего нужны векторные базы данных?"
- "Какие есть типы машинного обучения?"

## 🛠️ Кастомизация

### Добавление своих документов

Отредактируйте функцию `get_sample_documents()` в `embeddings.py`:

```python
documents = [
    ("Название документа", "Текст документа..."),
    ("Другой документ", "Еще текст..."),
]
```

### Изменение модели LLM

В `main.py`:

```python
rag_assistant = RAGAssistant(
    embedding_store=embedding_store,
    model="gpt-4",  # или другая модель
    temperature=0.5
)
```

### Изменение параметров поиска

```python
# Количество фрагментов для контекста (в main.py — аргумент generate_response)
answer, _ = assistant.generate_response(query, top_k=5)

# Или фильтр по источнику в коде (в приложении также читается RAG_SOURCE_FILTER из .env)
answer, _ = assistant.generate_response(query, top_k=5, source_filter="Python Основы")

# Размер чанков
store = EmbeddingStore()
store._create_chunks(text, chunk_size=1000, overlap=100)
```

### Использование другой модели эмбеддингов

В `main.py`:

```python
embedding_store = EmbeddingStore(
    collection_name="rag_documents",
    persist_directory="./chroma_db",
    embedding_model="text-embedding-3-large",  # более мощная модель
    api_key=api_key
)
```

Доступные модели OpenAI:
- `text-embedding-3-small` - быстрая и недорогая (по умолчанию)
- `text-embedding-3-large` - более качественная, но дороже
- `text-embedding-ada-002` - старая модель (legacy)

## 📊 Технические детали

### Эмбеддинги

- **Модель**: text-embedding-3-small (OpenAI)
- **Размерность**: 1536
- **Языки**: поддержка множества языков, включая русский
- **Преимущества**: 
  - Высокое качество семантического поиска
  - Отличная работа с многоязычными текстами
  - Не требует загрузки локальной модели
- **Недостатки**:
  - Требует API запросы (платно)
  - Нужно интернет-соединение

### Chunking

- **Размер чанка**: 500 символов
- **Перекрытие**: 50 символов
- **Причина**: сохранение контекста на границах

### ChromaDB

- **Тип**: персистентное хранилище
- **Метрика**: cosine similarity
- **Расположение**: ./chroma_db/

### Кеш

- **Тип**: JSON файл
- **Ключ**: SHA-256 хеш нормализованного запроса
- **Расположение**: ./cache.json

## ⚠️ Известные ограничения

1. **API ключ OpenAI** требуется для работы. Используется как для генерации ответов, так и для создания эмбеддингов. Запросы платные.

2. **Стоимость API**: 
   - Создание эмбеддингов: ~$0.02 за 1M токенов (text-embedding-3-small)
   - Генерация ответов: ~$0.50-$15 за 1M токенов (в зависимости от модели)
   - При первой загрузке документов будет сделано несколько запросов для создания эмбеддингов

3. **Кеш** простой и не учитывает изменения в базе знаний. При обновлении документов очистите кеш командой `clear_cache`

4. **Интернет-соединение** необходимо для всех операций (эмбеддинги и генерация ответов)

## 🎓 Обучающие цели

Этот проект создан для демонстрации:

- Как работает RAG (Retrieval-Augmented Generation)
- Как использовать векторные базы данных (ChromaDB)
- Как создавать эмбеддинги с OpenAI API
- Как кешировать ответы LLM
- Как организовать архитектуру AI-ассистента
- Как интегрировать OpenAI для всех AI-задач

Код максимально прокомментирован для понимания каждого шага.

## 📚 Дополнительные ресурсы

- [ChromaDB документация](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [RAG концепция](https://arxiv.org/abs/2005.11401)
- [OpenAI Pricing](https://openai.com/api/pricing/)

## 📝 Лицензия

Учебный проект, свободен для использования и модификации.

## 🤝 Вопросы и улучшения

Проект создан в учебных целях. Вы можете свободно модифицировать его под свои нужды.

---

**Приятного обучения! 🚀**

