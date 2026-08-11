const state = {
  rounds: new Map(),
  selectedRoundSlug: null,
  benchmark: null,
  resultsPayload: null,
  models: new Map(),
  selectedModelSlug: null,
  scoreView: "adjusted",
};

const elements = {
  heroSubtitle: document.querySelector("#hero-subtitle"),
  roundSelector: document.querySelector("#round-selector"),
  totalCount: document.querySelector("#total-count"),
  medqaCount: document.querySelector("#medqa-count"),
  medmcqaCount: document.querySelector("#medmcqa-count"),
  healthbenchCount: document.querySelector("#healthbench-count"),
  evaluation: document.querySelector("#evaluation"),
  resultsTitle: document.querySelector("#results-title"),
  resultsDescription: document.querySelector("#results-description"),
  qualityTitle: document.querySelector("#quality-title"),
  qualityDescription: document.querySelector("#quality-description"),
  evaluationTitle: document.querySelector("#evaluation-title"),
  evaluationDescription: document.querySelector("#evaluation-description"),
  evaluationScope: document.querySelector("#evaluation-scope"),
  evaluationModel: document.querySelector("#evaluation-model"),
  rawView: document.querySelector("#raw-view"),
  adjustedView: document.querySelector("#adjusted-view"),
  modelComparison: document.querySelector("#model-comparison"),
  chartSource: document.querySelector("#chart-source"),
  accuracyChart: document.querySelector("#accuracy-chart"),
  repeatabilityChart: document.querySelector("#repeatability-chart"),
  run1Accuracy: document.querySelector("#run-1-accuracy"),
  run1Detail: document.querySelector("#run-1-detail"),
  run2Accuracy: document.querySelector("#run-2-accuracy"),
  run2Detail: document.querySelector("#run-2-detail"),
  consistency: document.querySelector("#consistency"),
  consistencyDetail: document.querySelector("#consistency-detail"),
  stableCorrect: document.querySelector("#stable-correct"),
  stableCorrectDetail: document.querySelector("#stable-correct-detail"),
  datasetResults: document.querySelector("#dataset-results"),
  coverageIcon: document.querySelector("#coverage-icon"),
  coverageLabel: document.querySelector("#coverage-label"),
  coverageTitle: document.querySelector("#coverage-title"),
  coverageDescription: document.querySelector("#coverage-description"),
  ruleCard: document.querySelector("#rule-card"),
  ruleTitle: document.querySelector("#rule-title"),
  ruleRequirement: document.querySelector("#rule-requirement"),
  ruleProhibited: document.querySelector("#rule-prohibited"),
  ruleEnforcement: document.querySelector("#rule-enforcement"),
};

function percentage(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function chartPercent(value) {
  return Number((Number(value) * 100).toFixed(1));
}

function chartTrace(name, values, color, categories) {
  return {
    type: "bar",
    name,
    x: categories,
    y: values,
    text: values.map((value) => `${value.toFixed(1)}%`),
    textposition: "outside",
    textfont: { color: "#f3f5f7", size: 11 },
    cliponaxis: false,
    marker: {
      color,
      line: { color: "#111419", width: 1.5 },
    },
    hovertemplate: `<b>%{x}</b><br>${name}: %{y:.1f}%<extra></extra>`,
  };
}

function activeRound() {
  return state.rounds.get(state.selectedRoundSlug) || null;
}

function scoringExclusions() {
  return state.resultsPayload?.known_limitations || [];
}

function scoreScopeLabel(score) {
  const prefix = state.scoreView === "raw" ? "raw" : "adjusted";
  return `${prefix} ${score.objective.question_count}-question`;
}

function renderCharts() {
  if (!elements.accuracyChart || !elements.repeatabilityChart) return;
  if (!window.Plotly) {
    [elements.accuracyChart, elements.repeatabilityChart].forEach((element) => {
      element.classList.add("plotly-chart--error");
      element.textContent = "Plotly could not be loaded.";
    });
    return;
  }

  const models = [...state.models.values()];
  const names = models.map((model) => model.display_name);
  const scores = models.map((model) => model[state.scoreView].objective);
  const scope = scoreScopeLabel({ objective: scores[0] });
  const source = state.resultsPayload?.response_source?.source_file || "result archives";
  const plottedValues = scores.flatMap((score) => [
    score.run_1_accuracy,
    score.run_2_accuracy,
    score.answer_consistency,
    score.correct_and_consistent_rate,
  ]).map(chartPercent);
  const minimum = Math.min(...plottedValues);
  const lowerBound = Math.max(0, Math.floor((minimum - 5) / 5) * 5);
  const tickValues = [];
  for (let value = lowerBound; value <= 100; value += 5) tickValues.push(value);
  elements.chartSource.textContent = `${scope} view · zoomed ${lowerBound}–100% scale · calculated from ${source}`;

  const layout = {
    autosize: true,
    height: 332,
    margin: { l: 52, r: 14, t: 46, b: 76 },
    barmode: "group",
    barcornerradius: 6,
    bargap: 0.25,
    bargroupgap: 0.08,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#1d2127",
    font: {
      family: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
      color: "#d9dee6",
      size: 11,
    },
    legend: {
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: 1.18,
      yanchor: "top",
      font: { size: 11 },
    },
    xaxis: {
      categoryorder: "array",
      categoryarray: names,
      tickangle: -16,
      tickfont: { size: 10 },
      automargin: true,
      fixedrange: true,
      showgrid: false,
      zeroline: false,
    },
    yaxis: {
      title: { text: "Score (%) · zoomed", font: { size: 11 } },
      range: [lowerBound, 102.5],
      tickvals: tickValues,
      ticksuffix: "%",
      automargin: true,
      fixedrange: true,
      gridcolor: "#46515f",
      gridwidth: 1,
      zeroline: false,
    },
    hoverlabel: { bgcolor: "#171a1f", font: { color: "#ffffff" } },
  };
  const config = {
    responsive: true,
    displayModeBar: false,
    displaylogo: false,
  };

  window.Plotly.react(
    elements.accuracyChart,
    [
      chartTrace("Pass 1", scores.map((score) => chartPercent(score.run_1_accuracy)), "#00d5ff", names),
      chartTrace("Pass 2", scores.map((score) => chartPercent(score.run_2_accuracy)), "#ff4fd8", names),
    ],
    layout,
    config
  );
  window.Plotly.react(
    elements.repeatabilityChart,
    [
      chartTrace("Answer consistency", scores.map((score) => chartPercent(score.answer_consistency)), "#7ee63f", names),
      chartTrace("Correct twice", scores.map((score) => chartPercent(score.correct_and_consistent_rate)), "#ff9f1c", names),
    ],
    layout,
    config
  );

  elements.accuracyChart.setAttribute(
    "aria-label",
    `Grouped bar chart using a zoomed ${lowerBound} to 100 percent scale to compare pass 1 and pass 2 accuracy by model in the ${scope} view.`
  );
  elements.repeatabilityChart.setAttribute(
    "aria-label",
    `Grouped bar chart using a zoomed ${lowerBound} to 100 percent scale to compare answer consistency and correct-twice rate by model in the ${scope} view.`
  );
}

function selectedModel() {
  return state.models.get(state.selectedModelSlug) || null;
}

function selectedScore(model = selectedModel()) {
  return model ? model[state.scoreView] : null;
}

function setScoreView(view) {
  state.scoreView = view;
  const raw = view === "raw";
  elements.rawView.classList.toggle("is-active", raw);
  elements.rawView.setAttribute("aria-pressed", String(raw));
  elements.adjustedView.classList.toggle("is-active", !raw);
  elements.adjustedView.setAttribute("aria-pressed", String(!raw));
  populateComparison();
  populateEvaluation();
}

function populateRoundPresentation() {
  const round = activeRound();
  const firstModel = state.models.values().next().value;
  if (!round || !firstModel) return;
  const rawCount = firstModel.raw.objective.question_count;
  const adjustedCount = firstModel.adjusted.objective.question_count;
  const exclusions = scoringExclusions();
  const qids = exclusions.map((item) => item.qid);
  elements.heroSubtitle.textContent = round.slug === "combined"
    ? `${round.subtitle}. The individual rounds remain available for auditability.`
    : `${round.subtitle}. Review this sample independently or return to the combined reporting view.`;
  elements.resultsTitle.textContent = `${round.display_name} · four-model closed-book comparison`;
  elements.resultsDescription.textContent = `Two independent passes over the same ${rawCount} frozen objective questions. The adjusted reporting view applies identical exclusions to every model.`;
  elements.rawView.textContent = `Raw · ${rawCount}`;
  elements.adjustedView.textContent = `Adjusted · ${adjustedCount}`;
  elements.qualityTitle.textContent = `Reporting exclusions · ${qids.map((qid) => `qID ${qid}`).join(", ")}`;
  elements.qualityDescription.textContent = `${exclusions.map((item) => item.title).join("; ")}. The adjusted view excludes ${exclusions.length === 1 ? "this item" : "these items"} uniformly; raw results remain available for auditability.`;
}

function selectModel(slug) {
  if (!state.models.has(slug)) return;
  state.selectedModelSlug = slug;
  populateComparison();
  populateEvaluation();
}

function metricCell(value) {
  const cell = document.createElement("td");
  cell.textContent = percentage(value);
  return cell;
}

function populateComparison() {
  if (!state.models.size) return;
  elements.modelComparison.replaceChildren();
  state.models.forEach((model) => {
    const score = selectedScore(model).objective;
    const row = document.createElement("tr");
    row.classList.toggle("is-selected", model.slug === state.selectedModelSlug);

    const name = document.createElement("th");
    name.scope = "row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-button";
    button.setAttribute("aria-pressed", String(model.slug === state.selectedModelSlug));
    const label = document.createElement("strong");
    label.textContent = model.display_name;
    const method = document.createElement("span");
    method.textContent = "Closed-book run";
    button.append(label, method);
    button.addEventListener("click", () => selectModel(model.slug));
    name.append(button);

    row.append(
      name,
      metricCell(score.run_1_accuracy),
      metricCell(score.run_2_accuracy),
      metricCell(score.mean_accuracy),
      metricCell(score.answer_consistency),
      metricCell(score.correct_and_consistent_rate)
    );
    elements.modelComparison.append(row);
  });
}

function populateEvaluation() {
  const model = selectedModel();
  if (!model) return;
  const summary = model.summary;
  const score = selectedScore(model);
  const objective = score.objective;
  const isManual = model.collection_method === "manual";

  elements.run1Accuracy.textContent = percentage(objective.run_1_accuracy);
  elements.run1Detail.textContent = `${objective.run_1_correct} of ${objective.question_count} correct`;
  elements.run2Accuracy.textContent = percentage(objective.run_2_accuracy);
  elements.run2Detail.textContent = `${objective.run_2_correct} of ${objective.question_count} correct`;
  elements.consistency.textContent = percentage(objective.answer_consistency);
  elements.consistencyDetail.textContent = `${objective.consistent_answers} of ${objective.question_count} answers matched`;
  elements.stableCorrect.textContent = percentage(objective.correct_and_consistent_rate);
  elements.stableCorrectDetail.textContent = `${objective.correct_and_consistent} correct in both passes`;

  elements.evaluationTitle.textContent = `${model.display_name} · ${scoreScopeLabel(score)} results`;
  elements.evaluationDescription.textContent = isManual
    ? "Two consumer-web passes manually transcribed from the supplied screenshot."
    : "Two independently generated passes with model output and closed-book controls captured by the local runner.";
  const excludedQids = scoringExclusions().map((item) => item.qid);
  elements.evaluationScope.textContent = state.scoreView === "raw"
    ? `qID 1-${model.raw.objective.question_count}`
    : `qID${excludedQids.length === 1 ? "" : "s"} ${excludedQids.join(" + ")} excluded`;
  elements.evaluationModel.textContent = model.display_name;

  elements.datasetResults.replaceChildren();
  Object.entries(score.by_dataset).forEach(([dataset, metrics]) => {
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    const nameStrong = document.createElement("strong");
    nameStrong.textContent = dataset;
    const count = document.createElement("span");
    count.textContent = `${metrics.question_count} questions`;
    name.append(nameStrong, count);
    const values = [
      metrics.run_1_accuracy,
      metrics.run_2_accuracy,
      metrics.answer_consistency,
      metrics.correct_and_consistent_rate,
    ];
    row.append(name);
    values.forEach((value) => {
      const cell = document.createElement("td");
      const label = document.createElement("strong");
      label.textContent = percentage(value);
      const track = document.createElement("span");
      track.className = "mini-track";
      const fill = document.createElement("i");
      fill.style.width = percentage(value);
      track.append(fill);
      cell.append(label, track);
      row.append(cell);
    });
    elements.datasetResults.append(row);
  });

  const healthbench = summary.healthbench;
  if (healthbench) {
    elements.coverageIcon.textContent = healthbench.case_count;
    elements.coverageLabel.textContent = "HealthBench cases";
    elements.coverageTitle.textContent = "Answered twice";
    elements.coverageDescription.textContent = "Open-ended responses are saved, but objective accuracy and text consistency were not assigned.";
  } else {
    elements.coverageIcon.textContent = objective.question_count;
    elements.coverageLabel.textContent = "Objective scope";
    elements.coverageTitle.textContent = `${objective.question_count} questions × 2 passes`;
    elements.coverageDescription.textContent = "This model was evaluated on the objective subset only; HealthBench was not run.";
  }
  elements.evaluation.hidden = false;
  renderCharts();
}

function populateSummary() {
  const { case_counts: counts, cases } = state.benchmark;
  elements.totalCount.textContent = cases.length;
  elements.medqaCount.textContent = counts.MedQA ?? 0;
  elements.medmcqaCount.textContent = counts.MedMCQA ?? 0;
  elements.healthbenchCount.textContent = counts.HealthBench ?? 0;
  const rule = state.benchmark.evaluation_rules?.[0];
  if (rule) {
    elements.ruleTitle.textContent = rule.title;
    elements.ruleRequirement.textContent = rule.requirement;
    elements.ruleEnforcement.textContent = rule.enforcement;
    elements.ruleProhibited.replaceChildren();
    rule.prohibited.forEach((item) => {
      const entry = document.createElement("li");
      entry.textContent = item;
      elements.ruleProhibited.append(entry);
    });
    elements.ruleCard.hidden = false;
  } else {
    elements.ruleCard.hidden = true;
  }
}

elements.rawView.addEventListener("click", () => setScoreView("raw"));
elements.adjustedView.addEventListener("click", () => setScoreView("adjusted"));

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
      detail.textContent = "80 questions · reporting default";
      button.append(detail);
    }
    button.addEventListener("click", () => loadRound(round.slug));
    elements.roundSelector.append(button);
  });
  updateRoundButtons();
}

async function loadRound(roundSlug) {
  if (!state.rounds.has(roundSlug)) return;
  state.selectedRoundSlug = roundSlug;
  updateRoundButtons();
  elements.evaluation.hidden = true;
  try {
    const query = encodeURIComponent(roundSlug);
    const [benchmark, resultsPayload] = await Promise.all([
      fetchJson(`/api/benchmark?round=${query}`, "Benchmark"),
      fetchJson(`/api/results?round=${query}`, "Results"),
    ]);
    if (state.selectedRoundSlug !== roundSlug) return;
    state.benchmark = benchmark;
    state.resultsPayload = resultsPayload;
    state.models = new Map(resultsPayload.models.map((model) => [model.slug, model]));
    state.selectedModelSlug = resultsPayload.default_model_slug;
    state.scoreView = resultsPayload.default_score_view;
    const model = selectedModel();
    if (!model) throw new Error("Default model results are unavailable");
    document.title = `${resultsPayload.round.display_name} · Medical AI Benchmark`;
    populateSummary();
    populateRoundPresentation();
    setScoreView(state.scoreView);
  } catch (error) {
    elements.heroSubtitle.textContent = `Could not load benchmark: ${error.message}`;
  }
}

fetchJson("/api/rounds", "Rounds")
  .then((roundsPayload) => {
    state.rounds = new Map(roundsPayload.rounds.map((round) => [round.slug, round]));
    state.selectedRoundSlug = roundsPayload.default_round_slug;
    populateRoundSelector();
    return loadRound(roundsPayload.default_round_slug);
  })
  .catch((error) => {
    elements.heroSubtitle.textContent = `Could not load benchmark: ${error.message}`;
  });
