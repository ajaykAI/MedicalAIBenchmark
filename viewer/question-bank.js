const state = {
  rounds: new Map(),
  selectedRoundSlug: null,
  benchmark: null,
  models: new Map(),
  selectedModelSlug: null,
  objectiveResults: new Map(),
  objectiveQids: new Map(),
  limitations: new Map(),
  search: "",
  dataset: "all",
  category: "all",
  showAnswers: false,
};

const elements = {
  subtitle: document.querySelector("#bank-subtitle"),
  roundSelector: document.querySelector("#round-selector"),
  search: document.querySelector("#search"),
  dataset: document.querySelector("#dataset-filter"),
  category: document.querySelector("#category-filter"),
  model: document.querySelector("#model-filter"),
  showAnswers: document.querySelector("#show-answers"),
  clear: document.querySelector("#clear-filters"),
  resultCount: document.querySelector("#result-count"),
  viewport: document.querySelector("#question-viewport"),
  list: document.querySelector("#question-list"),
  empty: document.querySelector("#empty-state"),
  objectiveTemplate: document.querySelector("#objective-template"),
  healthTemplate: document.querySelector("#health-template"),
};

function tag(text, modifier = "") {
  const span = document.createElement("span");
  span.className = `tag ${modifier}`.trim();
  span.textContent = text;
  return span;
}

function resultBadge(text, modifier = "") {
  const span = document.createElement("span");
  span.className = `result-badge ${modifier}`.trim();
  span.textContent = text;
  return span;
}

function selectedModel() {
  return state.models.get(state.selectedModelSlug) || null;
}

function runBadgeText(pass, answer, confidence) {
  const confidenceText = Number.isInteger(confidence)
    ? ` · ${confidence}% confidence`
    : "";
  return `${pass} · ${answer}${confidenceText}`;
}

function renderObjective(caseData) {
  const fragment = elements.objectiveTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".case-card");
  const tags = fragment.querySelector(".case-card__tags");
  const qid = state.objectiveQids.get(caseData.id);
  tags.append(tag(caseData.dataset, `tag--${caseData.dataset.toLowerCase()}`));
  tags.append(tag(caseData.category));
  tags.append(tag(`qID ${qid}`, "tag--qid"));
  if (caseData.source_round) {
    const roundName = state.rounds.get(caseData.source_round)?.display_name || caseData.source_round;
    tags.append(tag(`${roundName} · local qID ${caseData.source_qid}`));
  }
  fragment.querySelector(".case-card__id").textContent = caseData.source_round
    ? `${caseData.source_round} · ${caseData.id.split(":").at(-1)}`
    : caseData.id;
  fragment.querySelector(".case-card__question").textContent = caseData.question;

  const limitation = state.limitations.get(caseData.id);
  const choices = fragment.querySelector(".choices");
  Object.entries(caseData.choices).forEach(([letter, text]) => {
    const item = document.createElement("li");
    item.className = "choice";
    if (state.showAnswers && letter === caseData.correct_answer) {
      item.classList.add(limitation ? "choice--source-key" : "choice--correct");
    }
    const marker = document.createElement("span");
    marker.className = "choice__marker";
    marker.textContent = letter;
    const content = document.createElement("span");
    content.textContent = text;
    item.append(marker, content);
    choices.append(item);
  });

  const limitationBox = fragment.querySelector(".case-limitation");
  if (limitation) {
    const title = document.createElement("strong");
    title.textContent = limitation.title;
    const reason = document.createElement("span");
    reason.textContent = `${limitation.reason} This item is excluded from the adjusted view.`;
    limitationBox.append(title, reason);
    limitationBox.hidden = false;
    card.classList.add("case-card--limited");
  }

  const answer = fragment.querySelector(".answer-key");
  answer.hidden = !state.showAnswers;
  answer.classList.toggle("answer-key--excluded", Boolean(limitation));
  answer.textContent = limitation
    ? `Source key: ${caseData.correct_answer} · excluded from adjusted scoring`
    : `Answer key: ${caseData.correct_answer}`;

  const model = selectedModel();
  const modelResult = state.objectiveResults.get(caseData.id);
  const caseResults = fragment.querySelector(".case-results");
  if (model && modelResult && state.showAnswers) {
    const heading = document.createElement("span");
    heading.className = "case-results__label";
    heading.textContent = `${model.display_name} · two passes`;
    const badges = document.createElement("div");
    badges.className = "case-results__badges";
    badges.append(
      resultBadge(
        runBadgeText("Pass 1", modelResult.run_1_answer, modelResult.run_1_confidence),
        modelResult.run_1_correct ? "result-badge--correct" : "result-badge--incorrect"
      ),
      resultBadge(
        runBadgeText("Pass 2", modelResult.run_2_answer, modelResult.run_2_confidence),
        modelResult.run_2_correct ? "result-badge--correct" : "result-badge--incorrect"
      ),
      resultBadge(
        modelResult.consistent ? "Same answer twice" : "Answer changed",
        modelResult.consistent ? "result-badge--consistent" : "result-badge--changed"
      )
    );
    if (modelResult.excluded_from_adjusted) {
      badges.append(resultBadge("Excluded from adjusted score", "result-badge--excluded"));
    }
    caseResults.append(heading, badges);
    caseResults.hidden = false;
  }
  card.dataset.caseId = caseData.id;
  return fragment;
}

function renderHealth(caseData) {
  const fragment = elements.healthTemplate.content.cloneNode(true);
  const tags = fragment.querySelector(".case-card__tags");
  tags.append(tag(caseData.dataset, "tag--healthbench"));
  tags.append(tag(caseData.category));
  fragment.querySelector(".case-card__id").textContent = caseData.id;

  const conversation = fragment.querySelector(".conversation");
  caseData.conversation.forEach((message) => {
    const row = document.createElement("div");
    row.className = `message message--${message.role}`;
    const role = document.createElement("span");
    role.className = "message__role";
    role.textContent = message.role;
    const content = document.createElement("p");
    content.textContent = message.content;
    row.append(role, content);
    conversation.append(row);
  });

  const rubricList = fragment.querySelector(".rubrics__list");
  caseData.rubrics.forEach((rubric) => {
    const row = document.createElement("div");
    row.className = "rubric";
    const points = document.createElement("span");
    points.className = rubric.points >= 0
      ? "rubric__points rubric__points--positive"
      : "rubric__points rubric__points--negative";
    points.textContent = `${rubric.points > 0 ? "+" : ""}${rubric.points}`;
    const criterion = document.createElement("p");
    criterion.textContent = rubric.criterion;
    row.append(points, criterion);
    rubricList.append(row);
  });

  const model = selectedModel();
  const resultNote = fragment.querySelector(".health-result-note");
  if (model?.summary?.healthbench) {
    resultNote.textContent = `${model.display_name} completed this case in both passes · rubric grading not performed`;
    resultNote.hidden = false;
  }
  return fragment;
}

function searchableText(caseData) {
  const qid = state.objectiveQids.get(caseData.id);
  const conversation = (caseData.conversation || []).map((message) => message.content);
  const rubrics = (caseData.rubrics || []).map((rubric) => rubric.criterion);
  return [
    caseData.id,
    qid ? `qid ${qid}` : "",
    caseData.dataset,
    caseData.category,
    caseData.question,
    ...(caseData.choices ? Object.values(caseData.choices) : []),
    ...conversation,
    ...rubrics,
  ].join(" ").toLowerCase();
}

function filteredCases() {
  const query = state.search.trim().toLowerCase();
  return state.benchmark.cases.filter((caseData) => {
    if (state.dataset !== "all" && caseData.dataset !== state.dataset) return false;
    if (state.category !== "all" && caseData.category !== state.category) return false;
    return !query || searchableText(caseData).includes(query);
  });
}

function render({ resetScroll = false } = {}) {
  if (!state.benchmark) return;
  const cases = filteredCases();
  const fragment = document.createDocumentFragment();
  cases.forEach((caseData) => {
    fragment.append(
      caseData.dataset === "HealthBench"
        ? renderHealth(caseData)
        : renderObjective(caseData)
    );
  });
  elements.list.replaceChildren(fragment);
  elements.resultCount.textContent = `Showing ${cases.length} of ${state.benchmark.cases.length} cases`;
  elements.empty.hidden = cases.length !== 0;
  if (resetScroll) elements.viewport.scrollTop = 0;
}

function populateFilters() {
  const cases = state.benchmark.cases;
  const objectiveCases = cases.filter((caseData) => caseData.choices);
  state.objectiveQids = new Map(
    objectiveCases.map((caseData, index) => [caseData.id, index + 1])
  );
  const availableDatasets = new Set(cases.map((caseData) => caseData.dataset));
  [...elements.dataset.options].forEach((option) => {
    option.disabled = option.value !== "all" && !availableDatasets.has(option.value);
  });
  while (elements.category.options.length > 1) elements.category.remove(1);
  [...new Set(cases.map((caseData) => caseData.category))].sort().forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    elements.category.append(option);
  });
}

function populateModels() {
  elements.model.replaceChildren();
  state.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.slug;
    option.textContent = model.display_name;
    elements.model.append(option);
  });
  elements.model.value = state.selectedModelSlug;
}

function selectModel(slug) {
  if (!state.models.has(slug)) return;
  state.selectedModelSlug = slug;
  const model = selectedModel();
  state.objectiveResults = new Map(
    model.objective_results.map((row) => [row.id, row])
  );
  render();
}

function fetchJson(url, label) {
  return fetch(url, { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error(`${label} HTTP ${response.status}`);
    return response.json();
  });
}

function updateRoundButtons() {
  elements.roundSelector.querySelectorAll("button").forEach((button) => {
    const selected = button.dataset.round === state.selectedRoundSlug;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function populateRoundSelector() {
  elements.roundSelector.replaceChildren();
  state.rounds.forEach((round) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "round-selector__button";
    button.dataset.round = round.slug;
    const name = document.createElement("strong");
    name.textContent = round.display_name;
    button.append(name);
    if (round.slug === "combined") {
      const detail = document.createElement("span");
      detail.textContent = "80 questions";
      button.append(detail);
    }
    button.addEventListener("click", () => loadRound(round.slug));
    elements.roundSelector.append(button);
  });
  updateRoundButtons();
}

function resetFilters() {
  state.search = "";
  state.dataset = "all";
  state.category = "all";
  state.showAnswers = false;
  elements.search.value = "";
  elements.dataset.value = "all";
  elements.category.value = "all";
  elements.showAnswers.checked = false;
}

async function loadRound(roundSlug) {
  if (!state.rounds.has(roundSlug)) return;
  state.selectedRoundSlug = roundSlug;
  updateRoundButtons();
  elements.resultCount.textContent = "Loading questions…";
  try {
    const query = encodeURIComponent(roundSlug);
    const [benchmark, resultsPayload] = await Promise.all([
      fetchJson(`/api/benchmark?round=${query}`, "Benchmark"),
      fetchJson(`/api/results?round=${query}`, "Results"),
    ]);
    if (state.selectedRoundSlug !== roundSlug) return;
    state.benchmark = benchmark;
    state.models = new Map(resultsPayload.models.map((model) => [model.slug, model]));
    state.limitations = new Map(
      resultsPayload.known_limitations.map((limitation) => [limitation.id, limitation])
    );
    state.selectedModelSlug = resultsPayload.default_model_slug;
    resetFilters();
    populateFilters();
    populateModels();
    selectModel(state.selectedModelSlug);
    elements.subtitle.textContent = `${resultsPayload.round.display_name} · ${benchmark.cases.length} searchable cases in a bounded scroll view.`;
    document.title = `${resultsPayload.round.display_name} Question Bank · Medical AI Benchmark`;
    render({ resetScroll: true });
  } catch (error) {
    elements.resultCount.textContent = `Could not load questions: ${error.message}`;
    elements.list.replaceChildren();
    elements.empty.hidden = false;
  }
}

elements.search.addEventListener("input", (event) => {
  state.search = event.target.value;
  render({ resetScroll: true });
});
elements.dataset.addEventListener("change", (event) => {
  state.dataset = event.target.value;
  render({ resetScroll: true });
});
elements.category.addEventListener("change", (event) => {
  state.category = event.target.value;
  render({ resetScroll: true });
});
elements.model.addEventListener("change", (event) => selectModel(event.target.value));
elements.showAnswers.addEventListener("change", (event) => {
  state.showAnswers = event.target.checked;
  render();
});
elements.clear.addEventListener("click", () => {
  resetFilters();
  render({ resetScroll: true });
});

fetchJson("/api/rounds", "Rounds")
  .then((roundsPayload) => {
    state.rounds = new Map(roundsPayload.rounds.map((round) => [round.slug, round]));
    state.selectedRoundSlug = roundsPayload.default_round_slug;
    populateRoundSelector();
    return loadRound(roundsPayload.default_round_slug);
  })
  .catch((error) => {
    elements.resultCount.textContent = `Could not load questions: ${error.message}`;
    elements.empty.hidden = false;
  });
