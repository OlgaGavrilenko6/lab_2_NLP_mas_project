## Агенты и роли

### Router (router_node)

Определяет тип запроса пользователя (intent) и инициирует передачу управления (handoff) к специализированному агенту.

Выполняет retrieval по заметкам (RAG-lite): load_notes + simple_retrieve_notes.

### Planner (planner_node)

Строит краткий план решения (5–10 шагов) в формате JSON (Pydantic).

План используется специализированными агентами как “скелет” ответа.

### Gather Tools (gather_tools_node)

ReAct-агент, задача которого — добрать недостающую информацию через инструменты.

Использует focus (что именно нужно добрать), который формирует reviewer.

Добавляет результаты инструментов в tool_context (контекст наблюдений).

## Специализированные агенты

conceptual_agent_node: теоретические вопросы по MAS/LLM-агентам.

architecture_agent_node: архитектура систем, дизайн state, handoff, memory.

coding_agent_node: реализация/код (обязателен исполнимый код).

daily_agent_node: повседневные задачи (планы, дедлайны, расчёты).

literature_agent_node: запросы/критерии литературы и структура обзора.

### Reviewer (reviewer_node)

Проверяет, достаточно ли данных для качественного ответа.

Если данных не хватает: need_more=true и формирует focus — что добрать инструментами.

Если хватает: need_more=false и (опционально) возвращает улучшенную версию ответа.

### Finalize (finalize_node)

Формирует final_answer, сохраняет в историю, формирует memory_summary.

## Реализованные паттерны МАС

### Router + специализированные агенты
Router определяет intent и маршрутизирует запрос к нужному агенту (coding, daily, …). Это базовый паттерн распределения задач по ролям.

### Planner–Executor
Planner создаёт план (структуру действий), а специализированный агент выступает как “executor”, который пишет ответ с опорой на план и контекст.

### Supervisor-like loop (Reviewer + Tool loop)
Reviewer выступает “контролёром качества”: если данных не хватает, он запускает цикл добора:
reviewer → gather_tools → (специализированный агент) → reviewer → ...
Цикл ограничен max_rounds, чтобы избежать зацикливания.

## Mermaid-диаграмма
flowchart TD
    U[User query] --> R[Router]
    R --> P[Planner]

    P -->|intent=conceptual| A1[Conceptual agent]
    P -->|intent=architecture| A2[Architecture agent]
    P -->|intent=coding| A3[Coding agent]
    P -->|intent=daily| A4[Daily agent]
    P -->|intent=literature| A5[Literature agent]

    A1 --> RV[Reviewer]
    A2 --> RV
    A3 --> RV
    A4 --> RV
    A5 --> RV

    RV -->|need_more=true| GT[Gather tools]
    GT -->|handoff back| A1
    GT -->|handoff back| A2
    GT -->|handoff back| A3
    GT -->|handoff back| A4
    GT -->|handoff back| A5

    RV -->|need_more=false| F[Finalize]
    F --> OUT[Final answer]

## Handoff:

router_node фиксирует handoff в handoff_log (router → intent-агент).

reviewer_node инициирует возврат к инструментам (handoff reviewer → gather_tools).

После добора gather_tools_node возвращает управление обратно к агенту (через маршрут графа).

## Tool calling

Инструменты реализованы как LangChain tools и вызываются ReAct-агентами:

coding_agent_node вызывает search_user_notes, save_user_note, calc по необходимости.

daily_agent_node вызывает days_until, calc, search_user_notes, save_user_note.

literature_agent_node вызывает search_user_notes, save_user_note.

gather_tools_node выбирает набор tools по intent и добирает информацию по focus.

## Memory

### Оперативная память (session memory):

history: список последних сообщений пользователя/ассистента.

trim_history: ограничивает историю до N последних элементов.

### Долговременная память (persistent memory):

data/notes.json: локальное хранилище заметок.

load_notes/save_notes/append_note: чтение/запись заметок.

simple_retrieve_notes: retrieval (RAG-lite) по токенам запроса.

### Как память влияет на работу:

Router учитывает memory_hits при выборе intent.

Planner учитывает memory_hits при построении плана.

Специализированные агенты используют memory_hits и history_tail как контекст.

Агент может сохранять полезные результаты в notes, чтобы система “обучалась” под пользователя.