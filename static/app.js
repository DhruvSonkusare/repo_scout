"use strict";

const EXAMPLES = [
  "Extract tables from scanned PDFs in Python",
  "Build a WhatsApp chatbot for my shop in Python",
  "Add full-text search to a Postgres app",
];
// The API answers a search in one response, so these steps follow a typical search's timing, not its real progress.
const SEARCH_STEPS = [
  { text: "Planning search…", seconds: 0 },
  { text: "Searching GitHub…", seconds: 3 },
  { text: "Ranking…", seconds: 6 },
];
const WEAK_SCORE = 50;
const STRONG_SCORE = 80;
const POLL_MS = 3000;
const IN_PROGRESS = new Set(["pending", "cloning", "indexing"]);
const VERDICT = /## Does it solve your problem\?\s*\**\s*(Fully|Partly|No)\b/i;
// Repo text and model replies are untrusted, so they can't add images or forms to the page.
const SANITIZE_OPTIONS = { FORBID_TAGS: ["img", "style", "form", "input", "iframe", "video", "audio"] };
const TOAST_MS = 3000;
const ERROR_TOAST_MS = 6000;
const MAX_TOASTS = 3;
const MAX_STAGGERED_CARDS = 8;
const COPY_FEEDBACK_MS = 1500;
const SMOOTH_SCROLL_MS = 500;

const $ = (id) => document.getElementById(id);

/** Build an element. Children become text nodes, so untrusted strings can't add HTML. */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else node[key] = value;
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function setText(id, text) {
  const node = $(id);
  node.textContent = text || "";
  node.hidden = !text;
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** How long to wait for a CSS exit transition, from the --normal token. No wait with motion off. */
function exitDuration() {
  if (reducedMotion()) return 0;
  return parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--normal")) || 0;
}

function toast(message, kind = "success") {
  const node = el("div", { class: `toast ${kind}` }, message);
  const list = $("toasts");
  list.append(node);
  while (list.children.length > MAX_TOASTS) list.firstElementChild.remove();
  setTimeout(() => {
    node.classList.add("leaving");
    setTimeout(() => node.remove(), exitDuration());
  }, kind === "error" ? ERROR_TOAST_MS : TOAST_MS);
}

/** Show an error where it belongs, and as a toast in case that spot is out of view. */
function showError(id, message) {
  setText(id, message);
  toast(message, "error");
}

function skeletonLine(width, tall = false) {
  const line = el("div", { class: `skeleton-line${tall ? " tall" : ""}` });
  line.style.width = width;
  return line;
}

function skeletonCard() {
  return el(
    "div",
    { class: "skeleton skeleton-card" },
    skeletonLine("40%", true),
    skeletonLine("25%"),
    skeletonLine("90%"),
    skeletonLine("75%"),
    skeletonLine("55%"),
  );
}

function guideSkeleton() {
  const node = el(
    "div",
    { class: "skeleton" },
    skeletonLine("30%", true),
    skeletonLine("85%"),
    skeletonLine("70%"),
    skeletonLine("35%", true),
    skeletonLine("95%"),
    skeletonLine("80%"),
    skeletonLine("60%"),
  );
  node.setAttribute("aria-hidden", "true");
  return node;
}

/** Fill an empty guide area with a skeleton. A guide already on screen stays until its replacement arrives. */
function showGuideSkeleton(target) {
  if (!target.textContent.trim() && !target.querySelector(".skeleton")) target.replaceChildren(guideSkeleton());
}

function removeSkeleton(target) {
  target.querySelector(".skeleton")?.remove();
}

/** Show how far indexing has got. Before the chunk count is known, the bar just shows activity. */
function showProgress(id, repo) {
  const bar = $(id);
  const known = repo.status === "indexing" && repo.chunk_count > 0;
  bar.hidden = false;
  bar.classList.toggle("indeterminate", !known);
  if (known) {
    const fraction = repo.chunks_embedded / repo.chunk_count;
    bar.firstElementChild.style.transform = `scaleX(${fraction})`;
    bar.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
  } else {
    bar.firstElementChild.style.transform = "";
    bar.removeAttribute("aria-valuenow");
  }
}

function hideProgress(id) {
  $(id).hidden = true;
}

function typingIndicator() {
  return el(
    "div",
    { class: "typing" },
    el("span", { class: "typing-dot" }),
    el("span", { class: "typing-dot" }),
    el("span", { class: "typing-dot" }),
    el("span", { class: "visually-hidden" }, "The answer is being written."),
  );
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function errorMessage(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(" ");
  return `The server returned an error (HTTP ${status}).`;
}

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("Couldn't reach the server. Is it running?", 0);
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(errorMessage(body, response.status), response.status);
  return body;
}

function postJson(path, body) {
  return api(path, { method: "POST", body });
}

function safeUrl(url) {
  try {
    return ["http:", "https:"].includes(new URL(url).protocol) ? url : "#";
  } catch {
    return "#";
  }
}

function addCopyButtons(root) {
  for (const pre of root.querySelectorAll("pre")) {
    const code = pre.querySelector("code") || pre;
    const button = el("button", { class: "button copy-button", type: "button" }, "Copy");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
        button.classList.add("copied");
        toast("Copied");
      } catch {
        button.textContent = "Couldn't copy";
        button.classList.add("copy-failed");
        toast("Couldn't copy that. Your browser blocked it.", "error");
      }
      setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove("copied", "copy-failed");
      }, COPY_FEEDBACK_MS);
    });
    pre.append(button);
  }
}

/** Render Markdown safely. While text is still streaming, skip the slower finishing touches. */
function renderMarkdown(target, text, { finished = true } = {}) {
  if (!window.marked || !window.DOMPurify) {
    target.classList.add("plain");
    target.textContent = text;
    return;
  }
  target.innerHTML = DOMPurify.sanitize(marked.parse(text), SANITIZE_OPTIONS);
  for (const link of target.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  if (!finished) return;
  if (window.hljs) {
    for (const block of target.querySelectorAll("pre code")) hljs.highlightElement(block);
  }
  addCopyButtons(target);
}

function formatDate(value) {
  return value ? new Date(value).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "unknown";
}

function scoreBadge(score) {
  if (score === null || score === undefined) return el("span", { class: "badge" }, "Not assessed");
  const level = score >= STRONG_SCORE ? "good" : score >= WEAK_SCORE ? "mid" : "bad";
  return el("span", { class: `badge ${level}` }, `${score}/100`);
}

/* ------------------------------ Find a repo ------------------------------ */

const find = {
  search: null, // the search on screen
  result: null, // the result in the solution view
  // Each navigation gets a new number, so slow requests from an earlier view can't change this one.
  run: 0,
};

function showFindView(name) {
  find.run += 1;
  $("search-view").hidden = name !== "search";
  $("results-view").hidden = name !== "results";
  $("solution-view").hidden = name !== "solution";
  window.scrollTo(0, 0);
}

async function loadSearchHistory() {
  try {
    const searches = await api("/api/finder/searches");
    const items = searches.map((search) => {
      const isActive = find.search && find.search.id === search.id;
      const button = el(
        "button",
        { class: `side-item${isActive ? " active" : ""}`, type: "button", title: search.problem, onclick: () => openSearch(search.id) },
        el("span", { class: "side-item-text" }, search.problem),
      );
      return el("li", {}, button);
    });
    $("search-history").replaceChildren(...items);
    $("search-history-empty").hidden = searches.length > 0;
  } catch (error) {
    setText("search-error", error.message);
  }
}

async function openSearch(id) {
  const run = ++find.run;
  try {
    const search = await api(`/api/finder/searches/${id}`);
    if (run === find.run) showResults(search);
  } catch (error) {
    if (run !== find.run) return;
    showFindView("search");
    setText("search-error", error.message);
  }
}

function showResults(search) {
  find.search = search;
  showFindView("results");
  $("results-problem").textContent = search.problem;
  const queries = (search.plan.queries || []).join(", ");
  $("results-plan").textContent = `Searched GitHub for: ${queries}${search.plan.language ? ` (language: ${search.plan.language})` : ""}`;
  renderResults(search.results);
  loadSearchHistory();
}

function renderResults(results) {
  const list = $("results-list");
  if (results.length === 0) {
    list.replaceChildren(el("p", { class: "empty" }, "GitHub found nothing recent for this. Try describing the problem differently."));
    return;
  }
  const strong = results.filter((result) => result.score !== null && result.score >= WEAK_SCORE);
  const weak = results.filter((result) => result.score === null || result.score < WEAK_SCORE);
  const nodes = strong.map((result, index) => {
    const card = resultCard(result);
    // Cards fade in one after another, and CSS turns this number into a delay.
    card.style.setProperty("--i", String(Math.min(index, MAX_STAGGERED_CARDS)));
    return card;
  });
  if (strong.length === 0) {
    nodes.push(el("p", { class: "empty" }, "Nothing here fits well, so treat these with caution."));
  }
  if (weak.length > 0) {
    const open = strong.length === 0;
    nodes.push(el(
      "details",
      { class: "weaker", open },
      el("summary", {}, `${weak.length} weaker or unassessed ${weak.length === 1 ? "match" : "matches"}`),
      ...weak.map(resultCard),
    ));
  }
  list.replaceChildren(...nodes);
}

function resultCard(result) {
  const meta = [
    `★ ${result.stars.toLocaleString()}`,
    result.language || "language unknown",
    `updated ${formatDate(result.last_pushed)}`,
  ].join(" · ");
  const steps = result.how_to_start.length
    ? el("div", {}, el("span", { class: "card-label" }, "How to start"), el("ol", {}, ...result.how_to_start.map((step) => el("li", {}, step))))
    : null;
  return el(
    "article",
    { class: "card" },
    el("div", { class: "card-head" }, el("h3", { class: "card-title" }, result.full_name), scoreBadge(result.score)),
    el("p", { class: "card-meta" }, meta),
    result.description ? el("p", {}, result.description) : null,
    el("p", {}, el("span", { class: "card-label" }, "Why it fits: "), result.why_it_fits),
    el("p", {}, el("span", { class: "card-label" }, "Caveats: "), result.caveats),
    steps,
    el(
      "div",
      { class: "button-row" },
      el("a", { class: "button", href: safeUrl(result.url), target: "_blank", rel: "noopener noreferrer" }, "Open on GitHub"),
      el(
        "button",
        { class: "button primary", type: "button", onclick: () => openSolution(result) },
        result.problem_guide ? "Show guide" : "Index & explain",
      ),
    ),
  );
}

async function runSearch(event) {
  event.preventDefault();
  const problem = $("problem").value.trim();
  if (problem.length < 3) return;
  const run = ++find.run;
  setText("search-error", "");
  $("search-button").disabled = true;
  const steps = $("search-steps");
  const items = SEARCH_STEPS.map((step) => el("li", {}, step.text));
  steps.replaceChildren(...items);
  steps.hidden = false;
  $("search-skeleton").replaceChildren(skeletonCard(), skeletonCard(), skeletonCard());
  $("search-skeleton").hidden = false;
  const started = Date.now();
  const paintSteps = () => {
    const elapsed = (Date.now() - started) / 1000;
    const current = SEARCH_STEPS.filter((step) => elapsed >= step.seconds).length - 1;
    items.forEach((item, index) => {
      item.className = index < current ? "done" : index === current ? "current" : "";
    });
  };
  paintSteps();
  const timer = setInterval(paintSteps, 500);
  try {
    const search = await postJson("/api/finder/search", { problem });
    if (run === find.run) showResults(search);
  } catch (error) {
    if (run === find.run) showError("search-error", error.message);
  } finally {
    clearInterval(timer);
    steps.hidden = true;
    $("search-skeleton").hidden = true;
    $("search-skeleton").replaceChildren();
    $("search-button").disabled = false;
  }
}

function delay(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

/** Poll until indexing ends, showing progress. Returns the repo, or null if the user navigated away. */
async function waitForIndexing(repoId, run) {
  for (;;) {
    const repo = await api(`/api/repos/${repoId}`);
    if (run !== find.run) return null;
    if (!IN_PROGRESS.has(repo.status)) {
      hideProgress("solution-progress");
      return repo;
    }
    showProgress("solution-progress", repo);
    let message = "Queued for indexing…";
    if (repo.status === "cloning") message = `Cloning ${repo.name}…`;
    if (repo.status === "indexing") message = `Indexing ${repo.name}: ${repo.chunks_embedded} of ${repo.chunk_count} chunks…`;
    setText("solution-status", message);
    await delay(POLL_MS);
    if (run !== find.run) return null;
  }
}

function renderVerification(verification) {
  const panel = $("verification");
  panel.hidden = !verification;
  panel.open = false;
  if (!verification) return;
  const count = verification.changes.length;
  const labels = {
    verified: verification.changed
      ? `Checked against the code: ${count} ${count === 1 ? "correction" : "corrections"}`
      : "Checked against the code: no corrections",
    failed: "The check against the code didn't finish, so treat this guide with care",
    skipped: "Not checked against the code: no source files to check",
  };
  $("verification-summary").textContent = labels[verification.status] || verification.status;
  const sections = [];
  if (count > 0) {
    sections.push(el("h3", {}, "What changed"), el("ul", {}, ...verification.changes.map((change) => el("li", {}, change))));
  }
  if (verification.files_checked.length > 0) {
    sections.push(el("h3", {}, "Files checked"), el("ul", {}, ...verification.files_checked.map((file) => el("li", {}, el("code", {}, file)))));
  }
  if (verification.changed) {
    sections.push(el("details", {}, el("summary", {}, "Show the draft before checking"), el("pre", {}, verification.draft)));
  }
  $("verification-body").replaceChildren(...sections);
}

function renderSolutionGuide(result) {
  const verdict = (result.problem_guide.match(VERDICT) || [])[1];
  const badge = $("solution-verdict");
  badge.hidden = !verdict;
  if (verdict) {
    const name = verdict[0].toUpperCase() + verdict.slice(1).toLowerCase();
    badge.className = `badge ${{ Fully: "good", Partly: "mid", No: "bad" }[name]}`;
    badge.textContent = name;
  }
  renderVerification(result.guide_verification);
  renderMarkdown($("solution-guide"), result.problem_guide);
  $("ask-follow-up").hidden = !result.repo_id;
  $("regenerate-guide").hidden = false;
}

function showSolutionError(message, canReindex) {
  $("solution-error-text").textContent = message;
  $("solution-retry").hidden = !canReindex;
  $("solution-error").hidden = false;
  hideProgress("solution-progress");
  removeSkeleton($("solution-guide"));
  toast(message, "error");
}

async function writeGuide(result, run) {
  setText("solution-status", "Writing the guide, then checking it against the code. This can take a minute…");
  const answer = await postJson(`/api/finder/results/${result.id}/guide`, undefined);
  if (run !== find.run) return;
  result.problem_guide = answer.guide;
  result.guide_verification = answer.verification;
  result.repo_id = answer.repo_id;
  setText("solution-status", "");
  renderSolutionGuide(result);
  toast("Guide ready");
}

/** Index the result if needed, then write its guide. reindex rebuilds a repo that failed or is out of date. */
async function buildGuide(result, { reindex = false } = {}) {
  const run = find.run;
  $("solution-error").hidden = true;
  $("regenerate-guide").hidden = true;
  showGuideSkeleton($("solution-guide"));
  try {
    setText("solution-status", "Starting…");
    let repo = await postJson(`/api/finder/results/${result.id}/index`, undefined);
    if (run !== find.run) return;
    result.repo_id = repo.id;
    if (reindex) repo = await postJson(`/api/repos/${repo.id}/reindex`, undefined);
    if (run === find.run && IN_PROGRESS.has(repo.status)) toast("Indexing started");
    repo = await waitForIndexing(repo.id, run);
    if (!repo) return;
    if (repo.status === "failed") {
      setText("solution-status", "");
      showSolutionError(`Indexing failed: ${repo.error || "see the server log."}`, true);
      return;
    }
    await writeGuide(result, run);
  } catch (error) {
    if (run !== find.run) return;
    setText("solution-status", "");
    showSolutionError(error.message, error.message.startsWith("Re-index needed"));
    $("regenerate-guide").hidden = !result.repo_id;
  }
}

function openSolution(result) {
  showFindView("solution");
  find.result = result;
  $("solution-problem").textContent = find.search.problem;
  const link = $("solution-repo-link");
  link.textContent = result.full_name;
  link.href = safeUrl(result.url);
  $("solution-verdict").hidden = true;
  $("verification").hidden = true;
  $("solution-error").hidden = true;
  $("solution-guide").replaceChildren();
  hideProgress("solution-progress");
  setText("solution-status", "");
  $("ask-follow-up").hidden = !result.repo_id;
  $("regenerate-guide").hidden = true;
  if (result.problem_guide) renderSolutionGuide(result);
  else buildGuide(result);
}

/* ------------------------------ My repos ------------------------------ */

const repos = {
  list: [],
  selectedId: null,
  conversationId: null,
  streaming: null, // the AbortController of the answer being streamed
  pollTimer: null,
};

function selectedRepo() {
  return repos.list.find((repo) => repo.id === repos.selectedId) || null;
}

function statusBadge(repo) {
  const level = repo.status === "ready" ? "good" : repo.status === "failed" ? "bad" : "mid";
  return el("span", { class: `badge ${level}` }, repo.status);
}

function renderRepoList() {
  const items = repos.list.map((repo) => el(
    "li",
    {},
    el(
      "button",
      { class: `side-item${repo.id === repos.selectedId ? " active" : ""}`, type: "button", title: repo.name, onclick: () => selectRepo(repo.id) },
      el("span", { class: "side-item-text" }, repo.name),
      statusBadge(repo),
    ),
  ));
  $("repo-list").replaceChildren(...items);
  $("repo-list-empty").hidden = repos.list.length > 0;
}

function repoSummary(repo) {
  if (repo.status === "failed") return "Indexing failed";
  if (repo.status === "indexing") return `Indexing: ${repo.chunks_embedded} of ${repo.chunk_count} chunks`;
  if (IN_PROGRESS.has(repo.status)) return repo.status === "cloning" ? "Cloning…" : "Queued for indexing…";
  return `${repo.file_count} files, ${repo.chunk_count} chunks`;
}

function renderRepoHeader() {
  const repo = selectedRepo();
  $("no-repo-selected").hidden = Boolean(repo);
  $("repo-view").hidden = !repo;
  if (!repo) {
    hideProgress("repo-progress");
    return;
  }
  const title = $("repo-title");
  title.textContent = repo.name;
  title.href = safeUrl(repo.url);
  $("repo-meta").textContent = repoSummary(repo);
  setText("repo-error", repo.status === "failed" ? repo.error || "Indexing failed. See the server log." : "");
  const busy = IN_PROGRESS.has(repo.status);
  if (busy) showProgress("repo-progress", repo);
  else hideProgress("repo-progress");
  $("reindex-repo").disabled = busy;
  $("delete-repo").disabled = busy;
  const ready = repo.status === "ready";
  $("ask-button").disabled = !ready || Boolean(repos.streaming);
  $("question").disabled = !ready;
  $("question").placeholder = ready ? `Ask about ${repo.name}` : "You can ask once indexing finishes.";
  $("generate-onboarding").disabled = !ready;
}

/** Refresh the list, and keep polling while anything is being indexed. */
async function refreshRepos() {
  clearTimeout(repos.pollTimer);
  try {
    const wasReady = selectedRepo()?.status === "ready";
    repos.list = await api("/api/repos");
    renderRepoList();
    renderRepoHeader();
    if (!wasReady && selectedRepo()?.status === "ready") loadConversations();
  } catch (error) {
    setText("add-repo-error", error.message);
  }
  if (repos.list.some((repo) => IN_PROGRESS.has(repo.status))) {
    repos.pollTimer = setTimeout(refreshRepos, POLL_MS);
  }
}

async function selectRepo(id, { newChat = false } = {}) {
  repos.streaming?.abort();
  repos.selectedId = id;
  repos.conversationId = null;
  closeDrawer();
  $("messages").replaceChildren();
  renderRepoList();
  renderRepoHeader();
  showRepoTab("chat");
  await loadConversations({ newChat });
  loadOnboardingGuide();
}

async function loadConversations({ newChat = true } = {}) {
  const repoId = repos.selectedId;
  if (repoId === null) return;
  try {
    const conversations = await api(`/api/repos/${repoId}/conversations`);
    if (repoId !== repos.selectedId) return;
    const options = conversations.map((item) => el("option", { value: String(item.id) }, item.title));
    $("conversation-select").replaceChildren(el("option", { value: "" }, "New chat"), ...options);
    $("conversation-select").value = repos.conversationId ? String(repos.conversationId) : "";
    if (!newChat && !repos.conversationId && conversations.length > 0) {
      await openConversation(conversations[0].id);
    }
  } catch (error) {
    setText("repo-error", error.message);
  }
}

async function openConversation(id) {
  repos.streaming?.abort();
  repos.conversationId = id;
  $("conversation-select").value = String(id);
  const list = $("messages");
  list.replaceChildren();
  try {
    const messages = await api(`/api/conversations/${id}/messages`);
    if (repos.conversationId !== id) return;
    list.replaceChildren(...messages.map((message) => (
      message.role === "user" ? userMessage(message.content) : savedAnswer(message)
    )));
    list.lastElementChild?.scrollIntoView({ block: "end" });
  } catch (error) {
    setText("repo-error", error.message);
  }
}

function startNewChat() {
  repos.streaming?.abort();
  repos.conversationId = null;
  $("conversation-select").value = "";
  $("messages").replaceChildren();
  $("question").focus();
}

function userMessage(text) {
  return el("div", { class: "message user" }, text);
}

function sourceChips(sources) {
  if (!sources.length) return null;
  const repoId = repos.selectedId;
  return el("div", { class: "source-chips" }, ...sources.map((source) => el(
    "button",
    { class: "source-chip", type: "button", onclick: () => openDrawer(repoId, source) },
    `[${source.n}] ${source.path}:L${source.start_line}-${source.end_line}`,
  )));
}

function modelNote(model) {
  return model ? el("p", { class: "message-model" }, `Answered by ${model}`) : null;
}

function savedAnswer(message) {
  const body = el("div", { class: "markdown" });
  renderMarkdown(body, message.content);
  return el("div", { class: "message ai" }, body, sourceChips(message.sources), modelNote(message.model));
}

/** Call onEvent for each NDJSON line of a streamed response. A read can end in the middle of a line. */
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
}

/** Follow the chat downwards: smoothly for a new message, at once for each piece of a streaming answer. */
function scrollIfNearBottom(wasNear, smooth = false) {
  if (!wasNear) return;
  const behavior = smooth && !reducedMotion() ? "smooth" : "auto";
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior });
}

function isNearBottom() {
  return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 120;
}

async function askQuestion(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  const repo = selectedRepo();
  if (!question || !repo || repos.streaming) return;
  const controller = new AbortController();
  repos.streaming = controller;
  $("ask-button").disabled = true;
  $("question").value = "";

  const list = $("messages");
  const body = el("div", { class: "markdown" });
  const typing = typingIndicator();
  const answer = el("div", { class: "message ai" }, body, typing);
  list.append(userMessage(question), answer);
  scrollIfNearBottom(true, true);
  // A smooth scroll takes a moment, and the page isn't near the bottom until it ends.
  const stillScrollingUntil = Date.now() + SMOOTH_SCROLL_MS;

  let text = "";
  let finished = false;
  let paintQueued = false;
  const showError = (message) => answer.append(el("p", { class: "message-error" }, message));
  const paint = () => {
    paintQueued = false;
    const wasNear = Date.now() < stillScrollingUntil || isNearBottom();
    renderMarkdown(body, text, { finished: false });
    scrollIfNearBottom(wasNear);
  };
  const handleEvent = (item) => {
    if (item.type === "meta") {
      repos.conversationId = item.conversation_id;
      answer.sources = item.sources;
    } else if (item.type === "token") {
      text += item.text;
      if (!paintQueued) {
        // Rendering once per frame keeps long answers smooth.
        paintQueued = true;
        requestAnimationFrame(paint);
      }
    } else if (item.type === "done") {
      finished = true;
      answer.model = item.model;
    } else if (item.type === "error") {
      finished = true;
      showError(item.message);
    }
  };

  try {
    const response = await fetch(`/api/repos/${repo.id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: repos.conversationId }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      throw new ApiError(errorMessage(errorBody, response.status), response.status);
    }
    await readEvents(response, handleEvent);
    if (!finished) showError("The answer stopped early, so it wasn't saved. Try again.");
  } catch (error) {
    if (error.name === "AbortError") return;
    showError(error instanceof ApiError ? error.message : "Couldn't reach the server. Is it running?");
  } finally {
    if (repos.streaming === controller) repos.streaming = null;
    typing.remove();
    if (text) renderMarkdown(body, text);
    if (!controller.signal.aborted) {
      const chips = sourceChips(answer.sources || []);
      if (chips) answer.append(chips);
      const note = modelNote(answer.model);
      if (note) answer.append(note);
      renderRepoHeader();
      await loadConversations();
      $("question").focus();
    }
  }
}

function showRepoTab(name) {
  const isChat = name === "chat";
  $("chat-panel").hidden = !isChat;
  $("guide-panel").hidden = isChat;
  for (const [id, active] of [["subtab-chat", isChat], ["subtab-guide", !isChat]]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-selected", String(active));
  }
}

async function loadOnboardingGuide() {
  const repoId = repos.selectedId;
  setText("onboarding-error", "");
  setText("onboarding-status", "");
  $("onboarding-guide").replaceChildren();
  $("generate-onboarding").textContent = "Write the guide";
  try {
    const answer = await api(`/api/repos/${repoId}/guide`);
    if (repoId !== repos.selectedId) return;
    renderMarkdown($("onboarding-guide"), answer.guide);
    $("generate-onboarding").textContent = "Write it again";
  } catch (error) {
    if (repoId === repos.selectedId && error.status !== 404) showError("onboarding-error", error.message);
  }
}

async function generateOnboardingGuide() {
  const repoId = repos.selectedId;
  const button = $("generate-onboarding");
  button.disabled = true;
  setText("onboarding-error", "");
  setText("onboarding-status", "Writing the guide. This can take a minute…");
  showGuideSkeleton($("onboarding-guide"));
  try {
    const answer = await postJson(`/api/repos/${repoId}/guide`, undefined);
    if (repoId !== repos.selectedId) return;
    renderMarkdown($("onboarding-guide"), answer.guide);
    button.textContent = "Write it again";
    toast("Guide ready");
  } catch (error) {
    if (repoId === repos.selectedId) showError("onboarding-error", error.message);
  } finally {
    if (repoId === repos.selectedId) {
      removeSkeleton($("onboarding-guide"));
      setText("onboarding-status", "");
      renderRepoHeader();
    }
  }
}

async function addRepo(event) {
  event.preventDefault();
  setText("add-repo-error", "");
  try {
    const repo = await postJson("/api/repos", { url: $("repo-url").value.trim() });
    $("repo-url").value = "";
    toast(IN_PROGRESS.has(repo.status) ? "Indexing started" : `${repo.name} is already added`);
    await refreshRepos();
    await selectRepo(repo.id);
  } catch (error) {
    showError("add-repo-error", error.message);
  }
}

async function reindexRepo() {
  setText("repo-error", "");
  try {
    await postJson(`/api/repos/${repos.selectedId}/reindex`, undefined);
    toast("Indexing started");
    await refreshRepos();
    await loadOnboardingGuide();
  } catch (error) {
    showError("repo-error", error.message);
  }
}

async function deleteRepo() {
  const repo = selectedRepo();
  if (!repo || !window.confirm(`Delete ${repo.name}, with its clone, index and chats?`)) return;
  try {
    await api(`/api/repos/${repo.id}`, { method: "DELETE" });
    repos.selectedId = null;
    closeDrawer();
    toast(`Deleted ${repo.name}`);
    await refreshRepos();
  } catch (error) {
    showError("repo-error", error.message);
  }
}

/* ------------------------------ File drawer ------------------------------ */

let drawerRequest = 0;
let drawerHideTimer = null;
let drawerOpener = null;

/** Slide the drawer in. It starts off-screen, so the browser has to lay it out before the transition can run. */
function slideDrawerIn() {
  clearTimeout(drawerHideTimer);
  const drawer = $("drawer");
  const backdrop = $("drawer-backdrop");
  drawer.hidden = false;
  backdrop.hidden = false;
  void drawer.offsetWidth;
  drawer.classList.add("open");
  backdrop.classList.add("open");
}

async function openDrawer(repoId, source) {
  const request = ++drawerRequest;
  drawerOpener = document.activeElement;
  slideDrawerIn();
  $("drawer-path").textContent = `${source.path}:L${source.start_line}-${source.end_line}`;
  setText("drawer-error", "");
  const lines = $("drawer-lines");
  lines.replaceChildren();
  $("drawer-close").focus();
  try {
    const params = new URLSearchParams({ path: source.path });
    const file = await api(`/api/repos/${repoId}/file?${params}`);
    if (request !== drawerRequest) return;
    const rows = file.content.split("\n").map((line, index) => {
      const number = index + 1;
      const cited = number >= source.start_line && number <= source.end_line;
      return el(
        "div",
        { class: `code-line${cited ? " cited" : ""}` },
        el("span", { class: "line-number" }, String(number)),
        el("span", {}, line),
      );
    });
    lines.replaceChildren(...rows);
    rows[source.start_line - 1]?.scrollIntoView({ block: "center" });
  } catch (error) {
    if (request === drawerRequest) setText("drawer-error", error.message);
  }
}

function closeDrawer() {
  drawerRequest += 1;
  const drawer = $("drawer");
  const backdrop = $("drawer-backdrop");
  if (drawer.hidden) return;
  drawer.classList.remove("open");
  backdrop.classList.remove("open");
  clearTimeout(drawerHideTimer);
  drawerHideTimer = setTimeout(() => {
    drawer.hidden = true;
    backdrop.hidden = true;
  }, exitDuration());
  // Keyboard users carry on from the chip they opened the file with.
  if (drawerOpener?.isConnected) drawerOpener.focus();
  drawerOpener = null;
}

/* ------------------------------ Page setup ------------------------------ */

function showPage(name) {
  const isFind = name === "find";
  $("find-page").hidden = !isFind;
  $("repos-page").hidden = isFind;
  for (const [id, active] of [["tab-find", isFind], ["tab-repos", !isFind]]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-selected", String(active));
  }
  closeDrawer();
  if (!isFind) refreshRepos();
}

async function askFollowUp() {
  const { repo_id: repoId } = find.result;
  showPage("repos");
  await refreshRepos();
  await selectRepo(repoId, { newChat: true });
  $("question").focus();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const body = await response.json();
    if (!response.ok) {
      const problems = [body.db !== "ok" ? `Database: ${body.db}` : null, body.models !== "ok" ? body.models : null];
      setText("health-banner", problems.filter(Boolean).join(" "));
    }
  } catch {
    setText("health-banner", "Couldn't reach the server. Is it running?");
  }
}

function init() {
  $("example-chips").replaceChildren(...EXAMPLES.map((example) => el(
    "button",
    { class: "chip", type: "button", onclick: () => { $("problem").value = example; $("problem").focus(); } },
    example,
  )));
  $("tab-find").addEventListener("click", () => showPage("find"));
  $("tab-repos").addEventListener("click", () => showPage("repos"));
  $("search-form").addEventListener("submit", runSearch);
  $("new-search").addEventListener("click", () => {
    showFindView("search");
    $("problem").value = "";
    $("problem").focus();
  });
  $("back-to-results").addEventListener("click", () => {
    showFindView("results");
    renderResults(find.search.results);
  });
  $("ask-follow-up").addEventListener("click", askFollowUp);
  $("regenerate-guide").addEventListener("click", () => buildGuide(find.result));
  $("solution-retry").addEventListener("click", () => buildGuide(find.result, { reindex: true }));

  $("add-repo-form").addEventListener("submit", addRepo);
  $("reindex-repo").addEventListener("click", reindexRepo);
  $("delete-repo").addEventListener("click", deleteRepo);
  $("subtab-chat").addEventListener("click", () => showRepoTab("chat"));
  $("subtab-guide").addEventListener("click", () => showRepoTab("guide"));
  $("generate-onboarding").addEventListener("click", generateOnboardingGuide);
  $("new-chat").addEventListener("click", startNewChat);
  $("conversation-select").addEventListener("change", (event) => {
    if (event.target.value) openConversation(Number(event.target.value));
    else startNewChat();
  });
  $("ask-form").addEventListener("submit", askQuestion);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("ask-form").requestSubmit();
    }
  });
  $("drawer-close").addEventListener("click", closeDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });

  checkHealth();
  loadSearchHistory();
}

init();
