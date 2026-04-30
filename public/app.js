const reviewText = document.querySelector("#reviewText");
const predictButton = document.querySelector("#predictButton");
const sampleButton = document.querySelector("#sampleButton");
const clearButton = document.querySelector("#clearButton");
const charCount = document.querySelector("#charCount");
const scoreValue = document.querySelector("#scoreValue");
const confidenceValue = document.querySelector("#confidenceValue");
const meterFill = document.querySelector("#meterFill");
const predictionBadge = document.querySelector("#predictionBadge");
const disclaimer = document.querySelector("#disclaimer");
const indicatorList = document.querySelector("#indicatorList");
const modelStatus = document.querySelector("#modelStatus");

const sampleReviews = [
  "This product is absolutely amazing! It exceeded all my expectations and I highly recommend it to everyone. The quality is perfect, the design is flawless, and the value is unbeatable!",
  "I bought this charger last month. It works, but the cable feels thinner than my old one and it gets warm after about twenty minutes. I would use it as a backup, not my main charger.",
];

function updateCount() {
  const length = reviewText.value.length;
  charCount.textContent = `${length} character${length === 1 ? "" : "s"}`;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.textContent = isLoading ? "Analyzing..." : "Analyze";
}

function renderIndicators(indicators) {
  const rows = [
    ["Word count", indicators.word_count],
    ["Sentence variation", indicators.sentence_length_variation],
    ["Repetition", formatPercent(indicators.repetition_ratio)],
    ["First person", formatPercent(indicators.first_person_ratio)],
    ["Type-token ratio", formatPercent(indicators.type_token_ratio)],
    ["Exclamation ratio", indicators.exclamation_ratio],
    ["Hype language", formatPercent(indicators.hype_word_ratio)],
  ];

  indicatorList.innerHTML = rows
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderResult(result) {
  const score = result.deception_score;
  scoreValue.textContent = formatPercent(score);
  confidenceValue.textContent = `${formatPercent(result.confidence)} confidence`;
  meterFill.style.width = formatPercent(score);
  predictionBadge.textContent = `${result.prediction} · confidence ${formatPercent(result.confidence)}`;
  predictionBadge.className = `badge ${result.label === 1 ? "computer" : "authentic"}`;
  disclaimer.textContent = result.disclaimer;
  renderIndicators(result.key_indicators);
}

function resetResult() {
  scoreValue.textContent = "--";
  confidenceValue.textContent = "-- confidence";
  meterFill.style.width = "0%";
  predictionBadge.textContent = "Waiting for text";
  predictionBadge.className = "badge neutral";
  disclaimer.textContent =
    "Results are probabilistic and should be treated as decision support, not absolute lie detection.";
  indicatorList.innerHTML = `
    <div><dt>Word count</dt><dd>--</dd></div>
    <div><dt>Sentence variation</dt><dd>--</dd></div>
    <div><dt>Repetition</dt><dd>--</dd></div>
    <div><dt>First person</dt><dd>--</dd></div>
  `;
}

async function analyze() {
  const text = reviewText.value.trim();
  if (text.length < 20) {
    predictionBadge.textContent = "Please enter at least 20 characters";
    predictionBadge.className = "badge neutral";
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Prediction failed.");
    renderResult(payload);
  } catch (error) {
    predictionBadge.textContent = error.message;
    predictionBadge.className = "badge neutral";
  } finally {
    setLoading(false);
  }
}

reviewText.addEventListener("input", updateCount);
predictButton.addEventListener("click", analyze);
sampleButton.addEventListener("click", () => {
  const index = reviewText.value.includes("absolutely amazing") ? 1 : 0;
  reviewText.value = sampleReviews[index];
  updateCount();
  analyze();
});
clearButton.addEventListener("click", () => {
  reviewText.value = "";
  updateCount();
  resetResult();
  reviewText.focus();
});

async function loadModelStatus() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    modelStatus.textContent = `${payload.model} · ${payload.trained_rows.toLocaleString()} training rows`;
  } catch {
    modelStatus.textContent = "Model status unavailable";
  }
}

updateCount();
loadModelStatus();
