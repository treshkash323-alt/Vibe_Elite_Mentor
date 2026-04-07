import os
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl, Field
from readability import Document
from openai import OpenAI

# =========================
# INIT
# =========================

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

app = FastAPI(title="Elite Mentor AI MVP v0.5")

# =========================
# CONFIG
# =========================

MIN_TEXT_LENGTH = 50
MAX_CONTEXT_CHARS = 8000

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K_CHUNKS = 4
MIN_WORD_LEN = 3

# =========================
# SCHEMAS
# =========================

class IngestRequest(BaseModel):
    url: HttpUrl


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    url: HttpUrl | None = None
    debug: bool = False


# =========================
# MEMORY (RAM)
# =========================

stored_pages: dict[str, dict] = {}
last_ingested_url: str | None = None

# =========================
# HELPERS
# =========================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [w for w in words if len(w) >= MIN_WORD_LEN]


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    if not text:
        return []

    chunks = []
    start = 0
    idx = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "index": idx,
                    "start": start,
                    "end": min(end, text_length),
                    "text": chunk_text,
                }
            )
            idx += 1

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


def score_chunk(chunk_text: str, question: str, title: str | None = None) -> int:
    question_words = tokenize(question)
    chunk_words = set(tokenize(chunk_text))
    title_words = set(tokenize(title or ""))

    score = 0

    for word in question_words:
        if word in chunk_words:
            score += 3
        if word in title_words:
            score += 1

    overlap_count = sum(1 for word in question_words if word in chunk_words)
    score += overlap_count

    return score


def select_relevant_chunks(question: str, page: dict, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    chunks = page.get("chunks", [])
    title = page.get("title")

    if not chunks:
        return []

    scored_chunks = []
    for chunk in chunks:
        score = score_chunk(chunk["text"], question, title=title)
        scored_chunks.append(
            {
                "index": chunk["index"],
                "score": score,
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"],
            }
        )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:top_k]
    top_chunks.sort(key=lambda x: x["index"])

    return top_chunks


def build_context(question: str, page: dict) -> tuple[str, list[dict]]:
    selected_chunks = select_relevant_chunks(question, page, top_k=TOP_K_CHUNKS)

    if not selected_chunks:
        raw_text = page.get("text", "")[:MAX_CONTEXT_CHARS]
        return raw_text, []

    context_parts = []
    total_len = 0
    used_chunks = []

    for chunk in selected_chunks:
        chunk_text = chunk["text"]
        if total_len + len(chunk_text) > MAX_CONTEXT_CHARS:
            continue

        context_parts.append(f"[CHUNK {chunk['index']}]\n{chunk_text}")
        total_len += len(chunk_text)
        used_chunks.append(chunk)

    if not context_parts:
        raw_text = page.get("text", "")[:MAX_CONTEXT_CHARS]
        return raw_text, []

    context = "\n\n".join(context_parts)
    return context, used_chunks


async def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }

    async with httpx.AsyncClient(
        timeout=20.0,
        verify=False,
        follow_redirects=True,
        headers=headers,
    ) as client_http:
        response = await client_http.get(url)
        response.raise_for_status()
        return response.text


def extract_text(html: str) -> tuple[str | None, str]:
    doc = Document(html)
    title = doc.short_title()

    article_html = doc.summary()
    soup = BeautifulSoup(article_html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = clean_text(text)

    return title, text


def get_page_or_raise(url: str | None = None) -> dict:
    global last_ingested_url

    if url:
        page = stored_pages.get(url)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found in memory")
        return page

    if last_ingested_url and last_ingested_url in stored_pages:
        return stored_pages[last_ingested_url]

    raise HTTPException(status_code=400, detail="No page loaded")


def build_messages(context: str, question: str) -> list[dict]:
    system = (
        "Ты AI-наставник. "
        "Отвечай только по предоставленному тексту страницы. "
        "Не выдумывай факты и не дополняй ответ внешними знаниями. "
        "Если в тексте страницы нет точного ответа, так и напиши: "
        "'В тексте страницы нет точного ответа на этот вопрос.' "
        "Сначала дай короткий ответ в 1-3 предложениях. "
        "Потом, если уместно, дай 2-4 коротких пункта по содержанию текста."
    )

    user = (
        f"Контекст страницы:\n{context}\n\n"
        f"Вопрос пользователя:\n{question}\n\n"
        "Ответь строго по этому контексту."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =========================
# UI
# =========================

UI_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Elite Mentor AI</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(15, 23, 42, 0.82);
      --panel-2: rgba(30, 41, 59, 0.7);
      --text: #e5eefc;
      --muted: #94a3b8;
      --accent: #60a5fa;
      --accent-2: #22c55e;
      --danger: #ef4444;
      --border: rgba(148, 163, 184, 0.18);
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      --radius: 20px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(34,197,94,0.12), transparent 24%),
        linear-gradient(180deg, #08101d 0%, #0b1020 45%, #0a0f1b 100%);
      min-height: 100vh;
    }

    .app {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 20px;
      padding: 20px;
      min-height: 100vh;
    }

    .sidebar, .main {
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
    }

    .sidebar {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }

    .brand h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }

    .brand p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }

    .card {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
    }

    .card h2 {
      margin: 0 0 10px;
      font-size: 15px;
      color: #dbeafe;
    }

    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }

    input[type="text"], textarea, select {
      width: 100%;
      background: rgba(2, 6, 23, 0.72);
      color: var(--text);
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      outline: none;
    }

    textarea {
      min-height: 100px;
      resize: vertical;
    }

    input[type="text"]:focus, textarea:focus, select:focus {
      border-color: rgba(96,165,250,0.7);
      box-shadow: 0 0 0 3px rgba(96,165,250,0.15);
    }

    .button-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
      align-items: center;
    }

    button {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      cursor: pointer;
      font: inherit;
      font-weight: 600;
      transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

    .btn-primary {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      color: white;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28);
    }

    .btn-secondary {
      background: rgba(148,163,184,0.14);
      color: var(--text);
      border: 1px solid var(--border);
    }

    .btn-success {
      background: linear-gradient(135deg, #22c55e, #16a34a);
      color: white;
      box-shadow: 0 10px 24px rgba(22, 163, 74, 0.25);
    }

    #micBtn {
      min-width: 52px;
      padding: 12px 14px;
      border-radius: 14px;
      background: #1f2937;
      color: white;
      border: 1px solid rgba(148,163,184,0.18);
      font-size: 18px;
    }

    #micBtn:hover {
      background: #374151;
    }

    .pages-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 280px;
      overflow: auto;
    }

    .page-item {
      background: rgba(2, 6, 23, 0.55);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 12px;
      cursor: pointer;
      transition: border-color 0.15s ease, background 0.15s ease;
    }

    .page-item:hover,
    .page-item.active {
      border-color: rgba(96,165,250,0.65);
      background: rgba(15, 23, 42, 0.95);
    }

    .page-title {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
      color: #dbeafe;
    }

    .page-url {
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }

    .meta {
      margin-top: 6px;
      font-size: 12px;
      color: #93c5fd;
    }

    .main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: calc(100vh - 40px);
      overflow: hidden;
    }

    .topbar {
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }

    .topbar h2 {
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.02em;
    }

    .topbar .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }

    .status {
      font-size: 13px;
      color: #bfdbfe;
      background: rgba(59,130,246,0.12);
      border: 1px solid rgba(96,165,250,0.25);
      padding: 8px 12px;
      border-radius: 999px;
      white-space: nowrap;
    }

    .chat {
      padding: 24px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .empty-state {
      margin: auto;
      max-width: 720px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed rgba(148,163,184,0.2);
      border-radius: 22px;
      padding: 28px;
      background: rgba(15,23,42,0.25);
    }

    .empty-state h3 {
      margin: 0 0 10px;
      color: #dbeafe;
      font-size: 22px;
    }

    .message {
      display: flex;
      gap: 12px;
      max-width: 960px;
    }

    .message.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }

    .avatar {
      flex: 0 0 42px;
      width: 42px;
      height: 42px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      font-size: 18px;
      background: rgba(59,130,246,0.18);
      border: 1px solid rgba(96,165,250,0.24);
    }

    .message.user .avatar {
      background: rgba(34,197,94,0.16);
      border-color: rgba(34,197,94,0.28);
    }

    .bubble {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.92);
      line-height: 1.55;
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    .message.user .bubble {
      background: rgba(16, 24, 40, 0.96);
    }

    .message-meta {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }

    .composer {
      padding: 18px 20px;
      border-top: 1px solid var(--border);
      display: grid;
      gap: 12px;
      background: rgba(7, 12, 24, 0.72);
    }

    .composer-top {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }

    .composer-top .left {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    .hint {
      color: var(--muted);
      font-size: 13px;
    }

    .error-box, .success-box {
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 14px;
      white-space: pre-wrap;
    }

    .error-box {
      background: rgba(239,68,68,0.12);
      border: 1px solid rgba(239,68,68,0.22);
      color: #fecaca;
    }

    .success-box {
      background: rgba(34,197,94,0.1);
      border: 1px solid rgba(34,197,94,0.22);
      color: #bbf7d0;
    }

    .footer-note {
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      padding-top: 2px;
    }

    @media (max-width: 1100px) {
      .app {
        grid-template-columns: 1fr;
      }

      .main {
        min-height: 75vh;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>Elite Mentor AI</h1>
        <p>RAG-light chat по тексту веб-страниц</p>
      </div>

      <div class="card">
        <h2>1. Загрузка страницы</h2>
        <label for="urlInput">URL страницы</label>
        <input id="urlInput" type="text" placeholder="https://example.com" />
        <div class="button-row">
          <button id="ingestBtn" class="btn-primary">Загрузить страницу</button>
          <button id="refreshPagesBtn" class="btn-secondary">Обновить список</button>
        </div>
      </div>

      <div id="feedbackArea"></div>

      <div class="card">
        <h2>2. Страницы в памяти</h2>
        <div id="pagesList" class="pages-list"></div>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div>
          <h2>Чат с контекстом страницы</h2>
          <div class="subtitle" id="pageInfo">Пока не выбрана страница</div>
        </div>
        <div class="status" id="statusBadge">Готов к работе</div>
      </div>

      <section id="chat" class="chat">
        <div class="empty-state" id="emptyState">
          <h3>Загрузи страницу и начинай диалог</h3>
          <p>
            Сначала добавь URL слева, потом выбери страницу и задай вопрос.
            Интерфейс отправляет запросы прямо в твой локальный FastAPI backend.
          </p>
        </div>
      </section>

      <section class="composer">
        <div class="composer-top">
          <div class="left">
            <label style="margin:0; display:flex; align-items:center; gap:8px;">
              <input id="debugCheckbox" type="checkbox" />
              <span class="hint">Показать debug retrieval</span>
            </label>
          </div>
          <div class="hint">Enter — отправить, Shift+Enter — новая строка</div>
        </div>

        <textarea id="questionInput" placeholder="Например: О чём эта страница?"></textarea>

        <div class="button-row">
          <button id="askBtn" class="btn-success">Отправить вопрос</button>
          <button id="micBtn" title="Голосовой ввод">🎤</button>
          <button id="clearChatBtn" class="btn-secondary">Очистить чат</button>
        </div>

        <div class="footer-note">
          UI работает поверх тех же endpoint'ов: /ingest, /ask, /pages
        </div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      selectedUrl: null,
      messages: [],
      pages: []
    };

    const chatEl = document.getElementById("chat");
    const emptyStateEl = document.getElementById("emptyState");
    const pagesListEl = document.getElementById("pagesList");
    const feedbackAreaEl = document.getElementById("feedbackArea");
    const statusBadgeEl = document.getElementById("statusBadge");
    const pageInfoEl = document.getElementById("pageInfo");

    const urlInputEl = document.getElementById("urlInput");
    const questionInputEl = document.getElementById("questionInput");
    const debugCheckboxEl = document.getElementById("debugCheckbox");

    const ingestBtnEl = document.getElementById("ingestBtn");
    const refreshPagesBtnEl = document.getElementById("refreshPagesBtn");
    const askBtnEl = document.getElementById("askBtn");
    const micBtnEl = document.getElementById("micBtn");
    const clearChatBtnEl = document.getElementById("clearChatBtn");

    function setStatus(text) {
      statusBadgeEl.textContent = text;
    }

    function showFeedback(type, text) {
      feedbackAreaEl.innerHTML = "";
      const div = document.createElement("div");
      div.className = type === "error" ? "error-box" : "success-box";
      div.textContent = text;
      feedbackAreaEl.appendChild(div);
    }

    function clearFeedback() {
      feedbackAreaEl.innerHTML = "";
    }

    function addMessage(role, content, meta = "") {
      state.messages.push({ role, content, meta });
      renderChat();
    }

    function renderChat() {
      chatEl.innerHTML = "";

      if (state.messages.length === 0) {
        chatEl.appendChild(emptyStateEl);
        return;
      }

      state.messages.forEach((msg) => {
        const wrapper = document.createElement("div");
        wrapper.className = `message ${msg.role}`;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = msg.role === "user" ? "🧑" : "🤖";

        const body = document.createElement("div");

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = msg.content;

        body.appendChild(bubble);

        if (msg.meta) {
          const meta = document.createElement("div");
          meta.className = "message-meta";
          meta.textContent = msg.meta;
          body.appendChild(meta);
        }

        wrapper.appendChild(avatar);
        wrapper.appendChild(body);
        chatEl.appendChild(wrapper);
      });

      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function updatePageInfo() {
      if (!state.selectedUrl) {
        pageInfoEl.textContent = "Пока не выбрана страница";
        return;
      }

      const page = state.pages.find(p => p.url === state.selectedUrl);
      if (!page) {
        pageInfoEl.textContent = state.selectedUrl;
        return;
      }

      const title = page.title || "Без заголовка";
      pageInfoEl.textContent = `${title} · ${page.url}`;
    }

    function renderPages() {
      pagesListEl.innerHTML = "";

      if (!state.pages.length) {
        const empty = document.createElement("div");
        empty.className = "page-item";
        empty.innerHTML = '<div class="page-title">Пока пусто</div><div class="page-url">Сначала сделай ingest страницы</div>';
        pagesListEl.appendChild(empty);
        return;
      }

      state.pages.forEach((page) => {
        const item = document.createElement("div");
        item.className = "page-item" + (state.selectedUrl === page.url ? " active" : "");

        item.innerHTML = `
          <div class="page-title">${escapeHtml(page.title || "Без заголовка")}</div>
          <div class="page-url">${escapeHtml(page.url)}</div>
          <div class="meta">Текст: ${page.text_length} · Чанки: ${page.chunks_count}</div>
        `;

        item.addEventListener("click", () => {
          state.selectedUrl = page.url;
          updatePageInfo();
          renderPages();
          setStatus("Выбрана страница");
        });

        pagesListEl.appendChild(item);
      });
    }

    async function refreshPages() {
      try {
        const res = await fetch("/pages");
        const data = await res.json();
        state.pages = data.pages || [];

        if (!state.selectedUrl && data.last_ingested_url) {
          state.selectedUrl = data.last_ingested_url;
        }

        if (state.selectedUrl) {
          const exists = state.pages.some(p => p.url === state.selectedUrl);
          if (!exists && state.pages.length) {
            state.selectedUrl = state.pages[0].url;
          }
        }

        renderPages();
        updatePageInfo();
      } catch (err) {
        showFeedback("error", "Не удалось получить список страниц.");
      }
    }

    async function ingestPage() {
      const url = urlInputEl.value.trim();
      if (!url) {
        showFeedback("error", "Введи URL страницы.");
        return;
      }

      clearFeedback();
      setStatus("Загрузка страницы...");

      ingestBtnEl.disabled = true;

      try {
        const res = await fetch("/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url })
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || "Ошибка ingest");
        }

        state.selectedUrl = data.url;
        await refreshPages();

        showFeedback(
          "success",
          `Страница загружена.\\nURL: ${data.url}\\nЧанки: ${data.chunks_count}\nДлина текста: ${data.text_length}`
        );
        setStatus("Страница загружена");
      } catch (err) {
        showFeedback("error", `Ошибка загрузки:\\n${err.message}`);
        setStatus("Ошибка загрузки");
      } finally {
        ingestBtnEl.disabled = false;
      }
    }

    async function askQuestion() {
      const question = questionInputEl.value.trim();

      if (!question) {
        showFeedback("error", "Введи вопрос.");
        return;
      }

      if (!state.selectedUrl) {
        showFeedback("error", "Сначала выбери или загрузи страницу.");
        return;
      }

      clearFeedback();
      setStatus("Формирую ответ...");

      addMessage("user", question, state.selectedUrl);
      questionInputEl.value = "";
      askBtnEl.disabled = true;

      try {
        const payload = {
          question,
          url: state.selectedUrl,
          debug: debugCheckboxEl.checked
        };

        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || "Ошибка ask");
        }

        let meta = `Контекст: ${data.context_length} · Чанки: ${data.selected_chunks_count}/${data.total_chunks_in_page}`;

        if (data.selected_chunk_indexes) {
          meta += ` · Индексы: [${data.selected_chunk_indexes.join(", ")}]`;
        }

        let content = data.answer;

        if (data.debug && data.debug.used_chunks) {
    const debugText = data.debug.used_chunks
        .map(chunk =>
            `\\n[DEBUG] chunk=${chunk.index}, score=${chunk.score}, range=${chunk.start}-${chunk.end}\n${chunk.preview}`
        )
        .join("\\n");

    content += `\\n\\n${debugText}`;
}

        addMessage("assistant", content, meta);
        setStatus("Ответ готов");
      } catch (err) {
        addMessage("assistant", `Ошибка: ${err.message}`, "Ошибка backend/API");
        setStatus("Ошибка ответа");
      } finally {
        askBtnEl.disabled = false;
      }
    }

    function clearChat() {
      state.messages = [];
      renderChat();
      setStatus("Чат очищен");
    }

    function escapeHtml(str) {
      return str
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    ingestBtnEl.addEventListener("click", ingestPage);
    refreshPagesBtnEl.addEventListener("click", refreshPages);
    askBtnEl.addEventListener("click", askQuestion);
    clearChatBtnEl.addEventListener("click", clearChat);

    questionInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
      }
    });

    refreshPages();
    renderChat();
    updatePageInfo();

    // ===============================
    // VOICE INPUT (STT)
    // ===============================

    let recognition = null;
    let isListening = false;

    function updateMicUI() {
      micBtnEl.style.background = isListening ? "#ef4444" : "#1f2937";
      micBtnEl.textContent = isListening ? "⏺" : "🎤";
      micBtnEl.title = isListening ? "Остановить запись" : "Голосовой ввод";
    }

    function initSpeech() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

      if (!SpeechRecognition) {
        showFeedback("error", "Этот браузер не поддерживает голосовой ввод. Открой интерфейс в Chrome или Edge.");
        return null;
      }

      const rec = new SpeechRecognition();
      rec.lang = "ru-RU";
      rec.continuous = false;
      rec.interimResults = false;

      rec.onstart = function () {
        isListening = true;
        updateMicUI();
        setStatus("Слушаю...");
      };

      rec.onresult = function (event) {
        const text = event.results[0][0].transcript;
        questionInputEl.value = text;
        askQuestion();
      };

      rec.onerror = function (event) {
        isListening = false;
        updateMicUI();
        showFeedback("error", "Ошибка голосового ввода: " + event.error);
        setStatus("Ошибка микрофона");
      };

      rec.onend = function () {
        isListening = false;
        updateMicUI();
        setStatus("Готов к работе");
      };

      return rec;
    }

    function toggleMic() {
      if (!recognition) {
        recognition = initSpeech();
        if (!recognition) return;
      }

      try {
        if (isListening) {
          recognition.stop();
        } else {
          recognition.start();
        }
      } catch (err) {
        showFeedback("error", "Не удалось запустить голосовой ввод.");
      }
    }

    micBtnEl.addEventListener("click", toggleMic);
    updateMicUI();
  </script>
</body>
</html>
"""

# =========================
# ROUTES
# =========================

@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


@app.get("/api")
def api_root():
    return {
        "status": "ok",
        "service": "Elite Mentor AI MVP v0.5"
    }


@app.get("/pages")
def list_pages():
    return {
        "count": len(stored_pages),
        "last_ingested_url": last_ingested_url,
        "pages": [
            {
                "url": url,
                "title": data.get("title"),
                "text_length": len(data.get("text", "")),
                "chunks_count": len(data.get("chunks", [])),
            }
            for url, data in stored_pages.items()
        ]
    }


@app.post("/ingest")
async def ingest(payload: IngestRequest):
    global last_ingested_url

    url = str(payload.url)

    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        html = await fetch_html(url)
        title, text = extract_text(html)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse page: {e}")

    if len(text) < MIN_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="Too little content")

    chunks = split_into_chunks(text)

    stored_pages[url] = {
        "url": url,
        "title": title,
        "text": text,
        "chunks": chunks,
    }
    last_ingested_url = url

    return {
        "status": "ingested",
        "url": url,
        "title": title,
        "text_length": len(text),
        "chunks_count": len(chunks),
        "preview": text[:500],
    }


@app.post("/ask")
def ask(payload: AskRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not set")

    page = get_page_or_raise(str(payload.url) if payload.url else None)

    context, used_chunks = build_context(payload.question, page)
    messages = build_messages(context=context, question=payload.question)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=700,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {e}")

    answer = response.choices[0].message.content
    if not answer:
        answer = "Модель не вернула ответ."

    result = {
        "answer": answer,
        "source_url": page["url"],
        "title": page.get("title"),
        "context_length": len(context),
        "selected_chunk_indexes": [chunk["index"] for chunk in used_chunks],
        "selected_chunks_count": len(used_chunks),
        "total_chunks_in_page": len(page.get("chunks", [])),
    }

    if payload.debug:
        result["debug"] = {
            "used_chunks": [
                {
                    "index": chunk["index"],
                    "score": chunk["score"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "preview": chunk["text"][:200],
                }
                for chunk in used_chunks
            ]
        }

    return result