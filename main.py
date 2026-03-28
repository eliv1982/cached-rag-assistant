"""
Главный файл для запуска RAG-ассистента.

Это точка входа в приложение. Здесь происходит:
1. Загрузка конфигурации
2. Инициализация всех компонентов (кеш, векторная база, RAG)
3. Добавление примеров документов (при первом запуске)
4. Интерактивный цикл общения с пользователем
"""

import os
from typing import Optional

from dotenv import load_dotenv
from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache


def source_filter_from_env() -> Optional[str]:
    """Читает RAG_SOURCE_FILTER из окружения (точное совпадение с metadata source)."""
    raw = os.getenv("RAG_SOURCE_FILTER")
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def initialize_system():
    """
    Инициализирует все компоненты RAG-системы.
    
    Returns:
        Кортеж (embedding_store, rag_assistant, cache)
    """
    print("=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ RAG-АССИСТЕНТА")
    print("=" * 70)
    
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Проверяем наличие API ключа OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  ВНИМАНИЕ: Не найден OPENAI_API_KEY в переменных окружения!")
        print("   Создайте файл .env и добавьте туда: OPENAI_API_KEY=your_key_here")
        print("   Или установите переменную окружения в системе.")
        print()
    
    # 1. Инициализируем кеш для хранения ответов
    print("\n[1/3] Инициализация кеша...")
    cache = ResponseCache(cache_file="cache.json")
    
    # 2. Инициализируем векторное хранилище ChromaDB
    print("\n[2/3] Инициализация векторного хранилища...")
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory="./chroma_db",
        embedding_model="text-embedding-3-small",  # Модель OpenAI для эмбеддингов
        api_key=api_key
    )
    
    # Проверяем, нужно ли добавить примеры документов
    if embedding_store.collection.count() == 0:
        print("\n📝 База данных пуста. Добавляем примеры документов...")
        sample_docs = get_sample_documents()
        embedding_store.add_documents(sample_docs)
    else:
        print(f"✓ В базе уже есть {embedding_store.collection.count()} документов")
    
    # 3. Инициализируем RAG-ассистента
    print("\n[3/3] Инициализация RAG-ассистента...")
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print("\n" + "=" * 70)
    print("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
    print("=" * 70)
    
    return embedding_store, rag_assistant, cache


def answer_question(
    query: str,
    rag_assistant: RAGAssistant,
    cache: ResponseCache,
    source_filter: Optional[str] = None,
) -> str:
    """
    Отвечает на вопрос пользователя с использованием кеша и RAG.
    
    Логика работы:
    1. Проверяем кеш - если ответ есть, возвращаем его
    2. Если ответа нет, выполняем RAG (поиск + генерация)
    3. Сохраняем новый ответ в кеш
    4. Возвращаем ответ
    
    Args:
        query: Вопрос пользователя
        rag_assistant: Экземпляр RAG-ассистента
        cache: Экземпляр кеша
        source_filter: Опционально — только чанки с metadata source равным строке
        
    Returns:
        Ответ на вопрос
    """
    print("\n" + "=" * 70)
    print(f"❓ ВОПРОС: {query}")
    print("=" * 70)
    
    # Шаг 1: Проверяем кеш
    print("\n[Шаг 1] Проверка кеша...")
    cached_answer = cache.get(query)
    
    if cached_answer:
        # Ответ найден в кеше - возвращаем его
        print("\n💾 Ответ из кеша:")
        print("-" * 70)
        print(cached_answer)
        print("-" * 70)
        return cached_answer
    
    # Шаг 2: Ответа нет в кеше - выполняем RAG
    print("\n[Шаг 2] Выполнение RAG (поиск + генерация)...")
    
    try:
        answer, search_results = rag_assistant.generate_response(
            query=query,
            top_k=3,
            verbose=True,
            source_filter=source_filter,
        )
        
        # Шаг 3: Сохраняем ответ в кеш
        print("\n[Шаг 3] Сохранение ответа в кеш...")
        cache.set(query, answer)
        
        # Выводим финальный ответ
        print("\n💡 ОТВЕТ:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        
        return answer
        
    except Exception as e:
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        print(f"\n❌ {error_msg}")
        return error_msg


def interactive_mode(
    rag_assistant: RAGAssistant,
    cache: ResponseCache,
    source_filter: Optional[str] = None,
):
    """
    Интерактивный режим общения с ассистентом.
    
    Пользователь может задавать вопросы в цикле до тех пор,
    пока не введет команду выхода.
    """
    print("\n" + "=" * 70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nВы можете задавать вопросы ассистенту.")
    print("Для выхода введите: exit, quit, выход или q")
    if source_filter:
        print(f"\n🔎 Активен фильтр по источнику (metadata): {source_filter!r}")
    print("\nДоступные команды:")
    print("  • cache - показать информацию о кеше")
    print("  • clear_cache - очистить кеш")
    print("  • stats - показать статистику системы")
    print()
    
    while True:
        try:
            # Получаем ввод от пользователя
            user_input = input("\n👤 Вы: ").strip()
            
            # Проверяем команды выхода
            if user_input.lower() in ['exit', 'quit', 'выход', 'q', '']:
                print("\n👋 До свидания!")
                break
            
            # Обрабатываем специальные команды
            if user_input.lower() == 'cache':
                print(f"\n📊 Кеш содержит {cache.size()} записей")
                continue
            
            if user_input.lower() == 'clear_cache':
                cache.clear()
                print("\n✓ Кеш очищен")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n📊 СТАТИСТИКА СИСТЕМЫ:")
                print(f"  • Документов в ChromaDB: {rag_assistant.embedding_store.collection.count()}")
                print(f"  • Записей в кеше: {cache.size()}")
                print(f"  • Модель LLM: {rag_assistant.model}")
                continue
            
            # Обрабатываем вопрос пользователя
            answer_question(
                user_input, rag_assistant, cache, source_filter=source_filter
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")


def demo_mode(
    rag_assistant: RAGAssistant,
    cache: ResponseCache,
    source_filter: Optional[str] = None,
):
    """
    Демонстрационный режим с заранее заготовленными вопросами.
    
    Показывает работу системы на примерах, включая использование кеша.
    """
    print("\n" + "=" * 70)
    print("🎬 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nСейчас будет продемонстрирована работа RAG-ассистента")
    print("на нескольких примерах вопросов.\n")
    
    # Список демо-вопросов
    demo_questions = [
        "Что такое Python и для чего он используется?",
        "Расскажи про RAG и как он работает",
        "Что такое векторные базы данных?",
        "Что такое Python и для чего он используется?"  # Повторный вопрос для демонстрации кеша
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#' * 70}")
        print(f"ВОПРОС {i} из {len(demo_questions)}")
        print(f"{'#' * 70}")
        
        answer_question(
            question, rag_assistant, cache, source_filter=source_filter
        )
        
        # Пауза между вопросами (кроме последнего)
        if i < len(demo_questions):
            input("\n[Нажмите Enter для следующего вопроса...]")
    
    print("\n\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)


def main():
    """
    Главная функция приложения.
    """
    try:
        # Инициализируем систему
        embedding_store, rag_assistant, cache = initialize_system()
        source_filter = source_filter_from_env()
        
        # Выбор режима работы
        print("\n" + "=" * 70)
        print("ВЫБОР РЕЖИМА РАБОТЫ")
        print("=" * 70)
        print("\n1. Интерактивный режим - задавайте свои вопросы")
        print("2. Демонстрационный режим - готовые примеры вопросов")
        print()
        
        mode = input("Выберите режим (1 или 2, по умолчанию 1): ").strip()
        
        if mode == '2':
            demo_mode(rag_assistant, cache, source_filter=source_filter)
            
            # Предложить перейти в интерактивный режим
            print("\n" + "=" * 70)
            continue_interactive = input("\nПерейти в интерактивный режим? (y/n): ").strip().lower()
            if continue_interactive in ['y', 'yes', 'д', 'да', '']:
                interactive_mode(rag_assistant, cache, source_filter=source_filter)
        else:
            interactive_mode(rag_assistant, cache, source_filter=source_filter)
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

