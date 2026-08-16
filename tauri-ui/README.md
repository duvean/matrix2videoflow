# Matrix2VideoFlow Tauri UI (MVP)

Это стартовая версия нового фронтенда на **Tauri + React**, чтобы уйти от ограничений PySide UI.

## Что уже учтено из требований

- Перенесён UI-каркас на Tauri.
- Все основные секции окна сделаны как **перетаскиваемые и ресайзабельные** панели через `react-grid-layout`.
- В блоке масок:
  - убрана кнопка `Toggle`;
  - у каждой маски есть отдельная иконка **выбора** и **видимости**;
  - `Inpaint selected frames` работает только по выбранным маскам (в MVP — имитация команды).
- У общего progress bar нет подписи (один бар под inpaint/export).
- В превью реализовано поведение: если видео дошло до конца, нажатие `Space` запускает проигрывание с начала.
- Таймлайн автоскроллится синхронно с текущим временем видео.
- Node graph сделан более информативным (отображены ключевые узлы пайплайна).

## Быстрый запуск через Docker Compose

Из корня репозитория:

```bash
docker compose up --build
```

После старта UI будет доступен на:

- http://localhost:1420

Остановка:

```bash
docker compose down
```

> В контейнере запускается веб-версия нового фронтенда (Vite dev server). Это самый простой путь «запустил и смотри UI».

## Что установить для desktop Tauri (локальный запуск без Docker)

### 1) Системные зависимости Tauri

Официальный список: https://tauri.app/start/prerequisites/

Минимально:
- **Node.js 20+**
- **Rust (stable)** и `cargo`
- Платформенные зависимости:
  - Linux: WebKitGTK, build-essential и др.
  - macOS: Xcode Command Line Tools
  - Windows: MSVC Build Tools + WebView2

### 2) Зависимости фронта

```bash
cd tauri-ui
npm install
```

## Запуск в режиме разработки (desktop Tauri)

```bash
cd tauri-ui
npm run tauri dev
```

Что произойдёт:
1. Поднимется Vite dev server на `http://localhost:1420`.
2. Tauri откроет desktop-окно с новым UI.

## Production-сборка

```bash
cd tauri-ui
npm run tauri build
```

Артефакты появятся в `src-tauri/target/release/bundle/...`.

## Следующий шаг интеграции с текущим Python-ядром

Сейчас UI — каркас. Чтобы подключить существующую логику `core/`:

1. Поднять локальный backend-bridge (например, FastAPI + WebSocket или gRPC) рядом с Python-процессором.
2. В Tauri вызывать bridge-команды:
   - загрузка батчей;
   - split на кадры;
   - inpaint selected masks;
   - экспорт.
3. Состояние проекта держать в TS-store (например, Zustand), а long-running операции — через job-очередь с прогрессом.

Такой путь позволит сохранить текущие Python-алгоритмы и быстро заменить только presentation layer.
