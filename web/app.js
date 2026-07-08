const state = {
  env: null,
  run: null,
  pollTimer: null,
  messages: [],
};

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function setBadge(element, text, tone) {
  element.textContent = text;
  element.className = `badge ${tone}`;
}

async function loadEnv() {
  const payload = await api("/api/env");
  state.env = payload.env;
  renderHeader();
}

function renderHeader() {
  const env = state.env || {};
  setBadge($("#envBadge"), env.ready ? "Env ready" : "Env missing", env.ready ? "ok" : "bad");
  setBadge($("#modelBadge"), `${env.model_count || 0} models`, env.model_count ? "ok" : "warn");
  setBadge($("#localBadge"), env.local_model_configured ? "Local ready" : "Local off", env.local_model_configured ? "ok" : "neutral");
  if (env.ready) {
    $("#statusLine").textContent = "Backend chooses runtime, routing, and token caps automatically";
  } else if (!env.openai_package_available && !env.local_model_configured) {
    $("#statusLine").textContent = "Install the Python openai package or configure LOCAL_MODEL_COMMAND";
  } else {
    $("#statusLine").textContent = "Set FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS locally";
  }
  $("#sendBtn").disabled = !env.ready || isRunning();
}

function isRunning() {
  return state.run && ["running", "cancelling"].includes(state.run.status);
}

function addMessage(role, content, meta = "", results = null) {
  state.messages.push({ role, content, meta, results });
  renderTranscript();
}

function updateAssistantMessage(content, meta = "", results = null) {
  const last = state.messages[state.messages.length - 1];
  if (last && last.role === "assistant") {
    last.content = content;
    last.meta = meta;
    last.results = results;
  } else {
    state.messages.push({ role: "assistant", content, meta, results });
  }
  renderTranscript();
}

function renderTranscript() {
  const transcript = $("#transcript");
  transcript.textContent = "";
  if (!state.messages.length) {
    const empty = document.createElement("section");
    empty.className = "empty-state";
    const title = document.createElement("h2");
    title.textContent = "Ready for a prompt";
    const copy = document.createElement("p");
    copy.textContent = "Paste one task prompt or a Track 1 tasks JSON array.";
    empty.append(title, copy);
    transcript.append(empty);
    return;
  }

  state.messages.forEach((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;
    article.append(bubble);

    if (message.results && message.results.length) {
      const list = document.createElement("div");
      list.className = "result-list";
      message.results.forEach((result) => {
        const card = document.createElement("div");
        card.className = "result-card";
        const id = document.createElement("strong");
        id.textContent = result.task_id || "task";
        const answer = document.createElement("p");
        answer.textContent = result.answer || "";
        card.append(id, answer);
        list.append(card);
      });
      article.append(list);
    }

    if (message.meta) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = message.meta;
      article.append(meta);
    }
    transcript.append(article);
  });
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function autoGrow() {
  const input = $("#promptInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 220)}px`;
}

async function startRun(prompt) {
  addMessage("user", prompt);
  updateAssistantMessage("Working...");
  $("#promptInput").value = "";
  autoGrow();
  state.run = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
  renderRun();
  startPolling();
}

async function cancelRun() {
  state.run = await api("/api/run/cancel", { method: "POST", body: "{}" });
  renderRun();
}

function startPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    state.run = await api("/api/run/status");
    renderRun();
    if (!isRunning()) {
      clearInterval(state.pollTimer);
    }
  }, 900);
}

function renderRun() {
  const run = state.run || { status: "idle", logs: [], results: [] };
  const running = isRunning();
  $("#sendBtn").disabled = running || !(state.env && state.env.ready);
  $("#stopBtn").disabled = !running;

  if (run.status === "idle") {
    return;
  }

  const prediction = run.prediction || {};
  const summary = run.token_summary || {};
  const validation = run.results_validation || {};
  const warningCount = (validation.errors || []).length + (validation.warnings || []).length;
  const scale = prediction.scale ? `scale ${prediction.scale}` : "estimating";
  const tokens = summary.found ? `${summary.total_tokens} Fireworks tokens` : "tokens pending";
  const categories = prediction.categories
    ? Object.entries(prediction.categories).map(([key, value]) => `${key} ${value}`).join(", ")
    : "";
  const meta = [titleCase(run.status), scale, tokens, categories, warningCount ? `${warningCount} warning(s)` : ""]
    .filter(Boolean)
    .join(" | ");

  if (running) {
    updateAssistantMessage("Working...", meta);
    return;
  }

  if (run.status === "succeeded") {
    const results = run.results || [];
    const nonEmpty = results.filter((result) => (result.answer || "").trim());
    if (nonEmpty.length === 1 && results.length === 1) {
      updateAssistantMessage(nonEmpty[0].answer.trim(), meta);
    } else {
      updateAssistantMessage(`Returned ${nonEmpty.length} answer(s).`, meta, results);
    }
  } else if (run.status === "cancelled") {
    updateAssistantMessage("Stopped.", meta);
  } else {
    const error = run.error || "Run failed.";
    const details = failureDetails(run, validation);
    updateAssistantMessage(`${error}${details}`, meta, run.results || []);
  }
}

function failureDetails(run, validation) {
  const lines = [
    ...(validation.errors || []),
    ...(validation.warnings || []),
  ];
  const stderrLines = (run.stderr || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-4);
  lines.push(...stderrLines);
  return lines.length ? `\n${lines.join("\n")}` : "";
}

function titleCase(value) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : "";
}

function bindEvents() {
  $("#promptInput").addEventListener("input", autoGrow);
  $("#promptInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#promptForm").requestSubmit();
    }
  });
  $("#promptForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const prompt = $("#promptInput").value.trim();
    if (!prompt || isRunning()) return;
    try {
      await startRun(prompt);
    } catch (error) {
      updateAssistantMessage(error.message || String(error));
      renderRun();
    }
  });
  $("#stopBtn").addEventListener("click", () => cancelRun().catch((error) => updateAssistantMessage(error.message)));
}

bindEvents();
loadEnv().catch((error) => {
  setBadge($("#envBadge"), "Env error", "bad");
  $("#statusLine").textContent = error.message || String(error);
});
