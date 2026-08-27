(() => {
  "use strict";

  const state = {
    dataset: null,
    tasks: [],
    selectedTaskId: null,
    selectedFrame: 0,
    filter: "all",
    query: "",
    drafts: new Map(),
    saveStates: new Map(),
  };

  const el = (id) => document.getElementById(id);
  const app = el("app");
  const taskList = el("task-list");
  const search = el("search");
  const filterBar = el("filters");
  const reviewForm = el("review-form");
  const noteField = el("note-field");
  const reviewNote = el("review-note");
  const saveButton = el("save-review");
  const saveStatus = el("save-status");
  const mainImage = el("main-image");
  const imagePlaceholder = el("image-placeholder");
  const taskRail = el("task-rail");
  const drawerBackdrop = el("drawer-backdrop");

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatSeconds(value) {
    return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(3)} s` : "—";
  }

  function formatClock(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function verdictFor(task) {
    return task?.review?.verdict || "unreviewed";
  }

  function verdictLabel(verdict) {
    return verdict === "correct" ? "正确" : verdict === "incorrect" ? "未正确" : "未审阅";
  }

  function currentTask() {
    return state.tasks.find((task) => task.task_id === state.selectedTaskId) || null;
  }

  function savedDraft(task) {
    return {
      verdict: verdictFor(task),
      note: task?.review?.note || "",
    };
  }

  function draftFor(task) {
    if (!task) return { verdict: "unreviewed", note: "" };
    if (!state.drafts.has(task.task_id)) state.drafts.set(task.task_id, savedDraft(task));
    return state.drafts.get(task.task_id);
  }

  function isDirty(task) {
    if (!task) return false;
    const draft = draftFor(task);
    const saved = savedDraft(task);
    return draft.verdict !== saved.verdict || draft.note !== saved.note;
  }

  function visibleTasks() {
    const query = state.query.trim().toLocaleLowerCase("zh-CN");
    return state.tasks.filter((task) => {
      const statusMatch = state.filter === "all" || verdictFor(task) === state.filter;
      if (!statusMatch) return false;
      if (!query) return true;
      const haystack = [task.task_id, String(task.task_id).padStart(4, "0"), task.case_id, task.app, task.task].join(" ").toLocaleLowerCase("zh-CN");
      return haystack.includes(query);
    });
  }

  function setQueryParams() {
    if (state.selectedTaskId === null) return;
    const url = new URL(window.location.href);
    url.searchParams.set("task", String(state.selectedTaskId));
    url.searchParams.set("frame", String(state.selectedFrame));
    window.history.replaceState(null, "", url);
  }

  function renderProgress() {
    const total = state.tasks.length;
    const correct = state.tasks.filter((task) => verdictFor(task) === "correct").length;
    const incorrect = state.tasks.filter((task) => verdictFor(task) === "incorrect").length;
    const reviewed = correct + incorrect;
    const unreviewed = total - reviewed;
    const width = total ? (reviewed / total) * 100 : 0;
    const root = el("progress-summary");
    root.replaceChildren();
    const line = make("div", "progress-line");
    const reviewedText = make("strong", "", `已审阅 ${reviewed} / ${total}`);
    const counts = make("span");
    const c = make("span", "count-correct", `正确 ${correct}`);
    const i = make("span", "count-incorrect", ` · 未正确 ${incorrect}`);
    const u = make("span", "", ` · 未审阅 ${unreviewed}`);
    counts.append(c, i, u);
    line.append(reviewedText, counts);
    const track = make("div", "progress-track");
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", String(total));
    track.setAttribute("aria-valuenow", String(reviewed));
    const fill = make("span", "progress-fill");
    fill.style.width = `${width}%`;
    track.append(fill);
    root.append(line, track);
  }

  function renderTaskList() {
    const tasks = visibleTasks();
    taskList.replaceChildren();
    el("task-empty").hidden = tasks.length !== 0;
    for (const task of tasks) {
      const verdict = verdictFor(task);
      const button = make("button", `task-row${task.task_id === state.selectedTaskId ? " selected" : ""}`);
      button.type = "button";
      button.dataset.taskId = String(task.task_id);
      button.setAttribute("aria-current", task.task_id === state.selectedTaskId ? "true" : "false");
      const id = make("span", "task-id mono", String(task.task_id).padStart(4, "0"));
      const main = make("span", "task-row-main");
      const meta = make("span", "task-row-meta");
      const appName = make("span", "", task.app || "未知 APP");
      const status = make("span", `verdict-word ${verdict}`, verdictLabel(verdict));
      meta.append(appName, status);
      const text = make("span", "task-row-text", task.task);
      const frameCount = make("span", "task-row-meta", `${task.frames.length} 帧 · ${task.case_id || "无用例编号"}`);
      main.append(meta, text, frameCount);
      button.append(id, main);
      button.addEventListener("click", () => {
        selectTask(task.task_id, 0);
        closeDrawer();
      });
      taskList.append(button);
    }
  }

  function renderMetadata(task) {
    const root = el("task-metadata");
    root.replaceChildren();
    const rows = [
      ["APP", task.app || "—"],
      ["CASE ID", task.case_id || "—"],
      ["预期步数", task.expected_steps ?? "—"],
      ["记录步数", task.recorded_steps ?? "—"],
      ["模型记录状态", task.model_status || "—"],
      ["任务总时长", formatSeconds(task.total_seconds)],
    ];
    for (const [label, value] of rows) {
      const row = make("div");
      row.append(make("dt", "", String(label)), make("dd", "", String(value)));
      root.append(row);
    }
  }

  function renderTimings(frame, task) {
    const timingList = el("timing-list");
    timingList.replaceChildren();
    const rows = [
      ["TTFT", formatSeconds(frame?.ttft_seconds)],
      ["模型生成总时长", formatSeconds(frame?.model_total_seconds)],
      ["请求端到端", formatSeconds(frame?.request_e2e_seconds)],
      ["单步端到端", formatSeconds(frame?.step_e2e_seconds)],
      ["任务总时长", formatSeconds(task?.total_seconds)],
    ];
    for (const [label, value] of rows) {
      const row = make("div");
      row.append(make("dt", "", label), make("dd", "", value));
      timingList.append(row);
    }
  }

  function renderContactSheet(task) {
    const root = el("contact-sheet");
    root.replaceChildren();
    task.frames.forEach((frame, index) => {
      const button = make("button", `frame-thumb${index === state.selectedFrame ? " selected" : ""}`);
      button.type = "button";
      button.setAttribute("aria-label", `查看第 ${index + 1} 帧，动作 ${frame.action_type}`);
      button.setAttribute("aria-pressed", index === state.selectedFrame ? "true" : "false");
      if (frame.image) {
        const image = document.createElement("img");
        image.loading = "lazy";
        image.src = frame.image;
        image.alt = `任务 ${task.task_id} 第 ${index + 1} 帧缩略图`;
        image.addEventListener("error", () => {
          const replacement = make("span", "thumb-missing", "图片缺失");
          image.replaceWith(replacement);
        }, { once: true });
        button.append(image);
      } else {
        button.append(make("span", "thumb-missing", "图片缺失"));
      }
      button.append(make("span", "", `${String(index + 1).padStart(2, "0")} · ${frame.action_type}`));
      button.addEventListener("click", () => selectFrame(index));
      root.append(button);
    });
    const selected = root.querySelector(".selected");
    if (selected) selected.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function renderFrame(task) {
    const count = task.frames.length;
    state.selectedFrame = Math.max(0, Math.min(state.selectedFrame, Math.max(0, count - 1)));
    const frame = task.frames[state.selectedFrame] || null;
    el("frame-counter").textContent = count ? `${String(state.selectedFrame + 1).padStart(2, "0")} / ${String(count).padStart(2, "0")}` : "00 / 00";
    el("action-label").textContent = frame?.action_type || "NO FRAME";
    el("prev-frame").disabled = state.selectedFrame <= 0;
    el("next-frame").disabled = state.selectedFrame >= count - 1;
    el("capture-badge").hidden = !String(frame?.capture_mode || "").toLowerCase().includes("secure");
    el("action-json").textContent = frame && Object.keys(frame.action || {}).length ? JSON.stringify(frame.action, null, 2) : "—";
    el("step-time").textContent = frame?.timestamp ? `记录时间 ${formatClock(frame.timestamp)}` : "";
    mainImage.hidden = true;
    mainImage.removeAttribute("src");
    imagePlaceholder.hidden = false;
    imagePlaceholder.textContent = frame?.image ? "正在加载截图…" : `截图不可用：${frame?.filename || "未记录文件"}`;
    if (frame?.image) {
      mainImage.alt = `任务 ${task.task_id} 第 ${state.selectedFrame + 1} 帧：${task.task}`;
      mainImage.onload = () => {
        imagePlaceholder.hidden = true;
        mainImage.hidden = false;
      };
      mainImage.onerror = () => {
        mainImage.hidden = true;
        imagePlaceholder.hidden = false;
        imagePlaceholder.textContent = `图片加载失败：${frame.filename}`;
      };
      mainImage.src = frame.image;
    }
    renderContactSheet(task);
    renderTimings(frame, task);
  }

  function renderSaveState(task) {
    const saveState = state.saveStates.get(task.task_id);
    saveStatus.className = "save-status";
    if (saveState?.kind === "saving") {
      saveStatus.textContent = "保存中…";
    } else if (saveState?.kind === "error") {
      saveStatus.textContent = `保存失败：${saveState.message}`;
      saveStatus.classList.add("error");
    } else if (isDirty(task)) {
      saveStatus.textContent = "未保存";
      saveStatus.classList.add("dirty");
    } else if (saveState?.kind === "saved") {
      saveStatus.textContent = `已保存 ${saveState.time}`;
    } else if (task.review?.reviewed_at) {
      saveStatus.textContent = `已有标注 ${formatClock(task.review.reviewed_at)}`;
    } else {
      saveStatus.textContent = "尚未修改";
    }
    saveButton.disabled = !state.dataset?.annotation_writable || saveState?.kind === "saving";
  }

  function renderReview(task) {
    const draft = draftFor(task);
    el("task-text").textContent = task.task;
    el("task-sop").textContent = task.sop || "—";
    renderMetadata(task);
    const radio = reviewForm.querySelector(`input[name="verdict"][value="${draft.verdict}"]`);
    if (radio) radio.checked = true;
    reviewNote.value = draft.note;
    noteField.hidden = draft.verdict !== "incorrect";
    renderSaveState(task);
  }

  function renderSelected() {
    const task = currentTask();
    if (!task) return;
    el("evidence-title").textContent = `TASK ${String(task.task_id).padStart(4, "0")}`;
    renderFrame(task);
    renderReview(task);
    renderTaskList();
    setQueryParams();
  }

  function selectTask(taskId, frameIndex = 0) {
    if (!state.tasks.some((task) => task.task_id === taskId)) return;
    state.selectedTaskId = taskId;
    state.selectedFrame = frameIndex;
    renderSelected();
  }

  function selectFrame(index) {
    const task = currentTask();
    if (!task || index < 0 || index >= task.frames.length) return;
    state.selectedFrame = index;
    renderFrame(task);
    setQueryParams();
  }

  function selectAdjacentTask(offset) {
    const tasks = visibleTasks();
    if (!tasks.length) return;
    let index = tasks.findIndex((task) => task.task_id === state.selectedTaskId);
    if (index < 0) index = offset > 0 ? -1 : tasks.length;
    const next = Math.max(0, Math.min(tasks.length - 1, index + offset));
    selectTask(tasks[next].task_id, 0);
  }

  function updateDraft(partial) {
    const task = currentTask();
    if (!task) return;
    const draft = { ...draftFor(task), ...partial };
    state.drafts.set(task.task_id, draft);
    noteField.hidden = draft.verdict !== "incorrect";
    renderSaveState(task);
  }

  async function saveCurrent() {
    const task = currentTask();
    if (!task || !state.dataset?.annotation_writable) return;
    const draft = draftFor(task);
    state.saveStates.set(task.task_id, { kind: "saving" });
    renderSaveState(task);
    try {
      const response = await fetch(`/api/reviews/${task.task_id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
      task.review = payload.review || {};
      state.drafts.set(task.task_id, savedDraft(task));
      state.saveStates.set(task.task_id, { kind: "saved", time: formatClock(payload.updated_at) || formatClock(new Date()) });
      renderProgress();
      renderTaskList();
      renderReview(task);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      state.saveStates.set(task.task_id, { kind: "error", message });
      renderSaveState(task);
      showToast(`标注未保存：${message}`);
    }
  }

  function showToast(message) {
    const toast = el("toast");
    toast.textContent = message;
    toast.hidden = false;
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => { toast.hidden = true; }, 6500);
  }

  function openDrawer() {
    taskRail.classList.add("open");
    drawerBackdrop.hidden = false;
  }

  function closeDrawer() {
    taskRail.classList.remove("open");
    drawerBackdrop.hidden = true;
  }

  function renderLoading() {
    taskList.replaceChildren(...Array.from({ length: 8 }, () => make("div", "loading-row")));
  }

  async function loadDataset() {
    renderLoading();
    try {
      const response = await fetch("/api/tasks", { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
      state.dataset = payload;
      state.tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
      el("dataset-name").textContent = payload.dataset || "未知数据集";
      el("dataset-path").textContent = payload.dataset_path || "";
      if (payload.annotation_error) {
        el("annotation-warning").hidden = false;
        el("annotation-warning").textContent = payload.annotation_error;
      }
      if (!state.tasks.length) throw new Error(`未发现任务目录：${payload.dataset_path}`);
      renderProgress();
      const params = new URL(window.location.href).searchParams;
      const queryTask = Number.parseInt(params.get("task") || "", 10);
      const queryFrame = Number.parseInt(params.get("frame") || "0", 10);
      const initial = state.tasks.some((task) => task.task_id === queryTask) ? queryTask : state.tasks[0].task_id;
      selectTask(initial, Number.isFinite(queryFrame) ? queryFrame : 0);
      app.setAttribute("aria-busy", "false");
    } catch (error) {
      app.setAttribute("aria-busy", "false");
      el("fatal-state").hidden = false;
      el("fatal-state").textContent = `无法载入审阅数据。\n${error instanceof Error ? error.message : String(error)}\n请确认输出目录存在且当前用户有读取权限。`;
      taskList.replaceChildren();
    }
  }

  search.addEventListener("input", () => {
    state.query = search.value;
    renderTaskList();
  });

  filterBar.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    state.filter = button.dataset.filter;
    filterBar.querySelectorAll("button[data-filter]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-checked", active ? "true" : "false");
    });
    renderTaskList();
  });

  reviewForm.addEventListener("change", (event) => {
    if (event.target.matches('input[name="verdict"]')) updateDraft({ verdict: event.target.value });
  });
  reviewNote.addEventListener("input", () => updateDraft({ note: reviewNote.value }));
  reviewForm.addEventListener("submit", (event) => { event.preventDefault(); saveCurrent(); });
  el("prev-frame").addEventListener("click", () => selectFrame(state.selectedFrame - 1));
  el("next-frame").addEventListener("click", () => selectFrame(state.selectedFrame + 1));
  el("prev-task").addEventListener("click", () => selectAdjacentTask(-1));
  el("next-task").addEventListener("click", () => selectAdjacentTask(1));
  el("open-tasks").addEventListener("click", openDrawer);
  el("close-tasks").addEventListener("click", closeDrawer);
  drawerBackdrop.addEventListener("click", closeDrawer);
  el("jump-review").addEventListener("click", () => el("review-panel").scrollIntoView({ behavior: "smooth" }));

  document.addEventListener("keydown", (event) => {
    const typing = event.target.matches("input, textarea, select, [contenteditable=true]");
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      saveCurrent();
      return;
    }
    if (typing) return;
    if (event.key === "/") {
      event.preventDefault();
      search.focus();
    } else if (event.key.toLowerCase() === "j") {
      selectAdjacentTask(1);
    } else if (event.key.toLowerCase() === "k") {
      selectAdjacentTask(-1);
    } else if (event.key === "ArrowLeft") {
      selectFrame(state.selectedFrame - 1);
    } else if (event.key === "ArrowRight") {
      selectFrame(state.selectedFrame + 1);
    } else if (event.key.toLowerCase() === "c") {
      const radio = reviewForm.querySelector('input[value="correct"]');
      radio.checked = true;
      updateDraft({ verdict: "correct" });
    } else if (event.key.toLowerCase() === "x") {
      const radio = reviewForm.querySelector('input[value="incorrect"]');
      radio.checked = true;
      updateDraft({ verdict: "incorrect" });
      reviewNote.focus();
    }
  });

  window.addEventListener("beforeunload", (event) => {
    if (!state.tasks.some((task) => isDirty(task))) return;
    event.preventDefault();
    event.returnValue = "";
  });

  loadDataset();
})();
