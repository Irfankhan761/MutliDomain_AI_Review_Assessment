
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let currentJobId = null;
let pollTimer = null;
let jobStartedAt = null;
window.currentReportText = "";

function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 4500);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pretty(value) {
  if (value === null || value === undefined || value === "" || String(value).toLowerCase() === "nan") return "Not available";
  return String(value).replaceAll("_", " ");
}

function titleCase(value) {
  const text = pretty(value).toLowerCase();
  if (text === "not available") return text;
  return text.replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.slice(1));
}

function safeNumber(value, fallback = "-") {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return n % 1 === 0 ? String(n) : n.toFixed(2);
}

function percent(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : "Not available";
}

function cleanMarkdownText(text) {
  return String(text || "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`/g, "")
    .replace(/^>\s?/gm, "")
    .replace(/#{1,6}\s?/g, "")
    .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
    .trim();
}

function isMarkdownDivider(line) {
  const t = line.trim();
  return !t || /^[-_*]{3,}$/.test(t) || /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(t);
}

function plainReportText(text) {
  return cleanMarkdownText(text)
    .split(/\r?\n/)
    .map((line) => cleanMarkdownText(line))
    .filter((line) => line && !isMarkdownDivider(line))
    .join("\n");
}

function sentenceFromReport(reportText, fallback) {
  const clean = plainReportText(reportText);
  const lines = clean.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const good = lines.find(l => l.length > 60 && !l.includes("|") && !/^\d+\./.test(l));
  return good || fallback;
}

function formatIssueList(issueText) {
  const text = String(issueText || "").trim();
  if (!text || text === "none" || text === "no_issue" || text === "not_available") return "No major issue detected";
  return text.replaceAll(";", ", ").replaceAll("_", " ");
}

function getReviewText(row) {
  return row.review_text || row.content || row.clean_review || row.evidence_phrase || row.rag_evidence_text || "No review text available.";
}

function renderReportHtml(reportText) {
  const raw = String(reportText || "").split(/\r?\n/);
  const sections = [];
  let current = { title: "Final Trust/Risk Summary", body: [] };

  raw.forEach((originalLine) => {
    const line = originalLine.trim();
    if (isMarkdownDivider(line)) return;
    const cleaned = cleanMarkdownText(line);
    if (!cleaned) return;

    const heading = cleaned.match(/^(\d+)\.\s+(.+)/);
    if (heading && cleaned.length < 90) {
      if (current.body.length) sections.push(current);
      current = { title: cleaned, body: [] };
      return;
    }

    current.body.push(cleaned);
  });

  if (current.body.length) sections.push(current);
  if (!sections.length) return `<div class="empty-state dark">No final summary available.</div>`;

  return sections.map((section) => {
    const bodyHtml = section.body.map((line) => {
      if (line.includes("|")) {
        const cells = line.split("|").map(x => cleanMarkdownText(x)).filter(Boolean);
        if (!cells.length) return "";
        return `<div class="report-chip-row">${cells.map(c => `<span>${escapeHtml(c)}</span>`).join("")}</div>`;
      }
      if (/^-\s+/.test(line)) return `<li>${escapeHtml(line.replace(/^-\s+/, ""))}</li>`;
      const keyValue = line.match(/^([^:]{3,45}):\s*(.+)$/);
      if (keyValue) {
        return `<p class="report-keyline"><strong>${escapeHtml(keyValue[1])}:</strong> ${escapeHtml(keyValue[2])}</p>`;
      }
      return `<p>${escapeHtml(line)}</p>`;
    }).join("");

    const wrapped = bodyHtml.includes("<li>") ? bodyHtml.replace(/(<li>.*<\/li>)/gs, `<ul class="report-list">$1</ul>`) : bodyHtml;
    return `<article class="report-section-card"><h4>${escapeHtml(section.title)}</h4>${wrapped}</article>`;
  }).join("");
}

function fieldByName(name) {
  const input = document.querySelector(`[name='${name}']`);
  return input ? input.closest(".field") : null;
}

function showField(name, show) {
  const field = fieldByName(name);
  if (!field) return;
  field.classList.toggle("hidden", !show);
}


function configureDomainOptions(mode) {
  const select = document.querySelector("[name='domain']");
  if (!select) return;

  const current = select.value;
  const options = mode === "google_maps_url"
    ? [
        ["auto", "Auto detect"],
        ["hotel", "Hotel"],
        ["restaurant", "Restaurant"]
      ]
    : [
        ["auto", "Auto detect"],
        ["mobile_app", "Mobile App"],
        ["hotel", "Hotel"],
        ["ecommerce", "E-commerce"],
        ["restaurant", "Restaurant"]
      ];

  select.innerHTML = options
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");

  select.value = options.some(([value]) => value === current) ? current : "auto";
}

function configureSortOptions(mode) {
  const select = document.querySelector("[name='sort_order']");
  if (!select) return;

  const current = select.value;
  const options = mode === "google_maps_url"
    ? [
        ["most_relevant", "Most Relevant"],
        ["newest", "Newest"],
        ["highest_rating", "Highest Rating"],
        ["lowest_rating", "Lowest Rating"]
      ]
    : [
        ["newest", "Newest"],
        ["most_relevant", "Most Relevant"]
      ];

  select.innerHTML = options
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");

  const allowed = options.some(([value]) => value === current);
  select.value = allowed ? current : options[0][0];
}

function setModeRequired(mode) {
  const names = ["csv_file", "review_text", "google_url", "app_id", "google_maps_url"];
  names.forEach((name) => {
    const input = document.querySelector(`[name='${name}']`);
    if (input) input.required = false;
  });

  const requiredByMode = {
    single: "review_text",
    google_url: "google_url",
    app_id: "app_id",
    google_maps_url: "google_maps_url"
  };
  const target = document.querySelector(`[name='${requiredByMode[mode] || ""}']`);
  if (target) target.required = true;
}

function setMode(mode) {
  $("#modeInput").value = mode;

  $$(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });

  $$(".mode-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.panel !== mode);
  });

  const domain = document.querySelector("[name='domain']");
  const rating = document.querySelector("[name='rating']");
  const entityName = document.querySelector("[name='entity_name']");
  const sampleSize = document.querySelector("[name='sample_size']");
  const maxScraperReviews = document.querySelector("[name='max_scraper_reviews']");

  setModeRequired(mode);
  configureDomainOptions(mode);
  configureSortOptions(mode);

  if (mode === "csv") {
    showField("domain", true);
    showField("sample_size", true);

    showField("rating", false);
    showField("entity_name", false);
    showField("max_scraper_reviews", false);
    showField("sort_order", false);

    if (domain) domain.value = "auto";
  }

  if (mode === "single") {
    showField("domain", true);
    showField("rating", true);
    showField("entity_name", true);

    showField("sample_size", false);
    showField("max_scraper_reviews", false);
    showField("sort_order", false);

    if (domain && domain.value === "auto") domain.value = "mobile_app";
    if (rating && !rating.value) rating.value = "3";
  }

  if (mode === "google_url" || mode === "app_id") {
    showField("max_scraper_reviews", true);
    showField("sort_order", true);

    showField("domain", false);
    showField("rating", false);
    showField("entity_name", false);
    showField("sample_size", false);

    if (domain) domain.value = "mobile_app";
    if (sampleSize) sampleSize.value = "0";
    if (maxScraperReviews && !maxScraperReviews.value) maxScraperReviews.value = "200";
  }

  if (mode === "google_maps_url") {
    showField("domain", true);
    showField("max_scraper_reviews", true);
    showField("sort_order", true);

    showField("rating", false);
    showField("entity_name", false);
    showField("sample_size", false);

    if (domain) domain.value = "auto";
    if (sampleSize) sampleSize.value = "0";
    if (maxScraperReviews && !maxScraperReviews.value) maxScraperReviews.value = "100";
  }
}

function setStatus(status, progress = 0) {
  const badge = $("#statusBadge");
  badge.textContent = titleCase(status || "waiting");
  badge.className = "status-badge";
  if (status === "completed") badge.classList.add("completed");
  if (status === "failed") badge.classList.add("failed");
  $("#progressBar").style.width = `${Math.max(0, Math.min(100, progress || 0))}%`;
}

function renderTrace(trace = []) {
  const box = $("#traceList");
  if (!trace.length) {
    box.innerHTML = `<div class="empty-state">No trace yet.</div>`;
    return;
  }

  const friendly = trace.map((step) => {
    const title = pretty(step.title);
    if (title.toLowerCase().includes("input")) return { ...step, message: "The orchestrator identified the input type and selected the correct workflow." };
    if (title.toLowerCase().includes("scraper")) return { ...step, message: "Public reviews were collected and prepared for analysis." };
    if (title.toLowerCase().includes("specialised")) {
      const ragUsed = step.output?.use_rag === true;
      const msg = ragUsed
        ? "Sentiment, rating prediction, discrepancy, issue mining, RAG evidence retrieval and risk scoring agents completed."
        : "Sentiment, rating prediction, discrepancy, issue mining and risk scoring agents completed. RAG retrieval was skipped for this run.";
      return { ...step, message: msg };
    }
    return step;
  });

  box.innerHTML = friendly.map((step, i) => {
    const output = step.output || {};
    const chips = Object.entries(output)
      .filter(([_, v]) => v !== null && v !== undefined && String(v).length < 110)
      .slice(0, 5)
      .map(([k, v]) => `<span class="trace-chip"><b>${titleCase(k)}:</b> ${escapeHtml(pretty(v))}</span>`)
      .join("");
    return `
      <div class="trace-item">
        <div class="trace-num">${i + 1}</div>
        <div>
          <h4>${escapeHtml(titleCase(step.title))}</h4>
          <p>${escapeHtml(pretty(step.message))}</p>
          ${chips ? `<div class="trace-output">${chips}</div>` : ""}
        </div>
      </div>
    `;
  }).join("");
}

function renderBars(container, rows = []) {
  const el = $(container);
  if (!rows.length) {
    el.innerHTML = `<div class="empty-state">No data available</div>`;
    return;
  }
  const filtered = rows.filter(r => String(r.label || "").toLowerCase() !== "nan").slice(0, 8);
  const max = Math.max(...filtered.map((r) => Number(r.count || 0)), 1);
  el.innerHTML = filtered.map((r) => {
    const label = titleCase(r.label);
    const count = Number(r.count || 0);
    const width = Math.max(3, (count / max) * 100);
    return `
      <div class="bar-row">
        <span>${escapeHtml(label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        <b>${count.toLocaleString()}</b>
      </div>
    `;
  }).join("");
}

function renderHeadline(headline = {}, summary = {}) {
  const entity = pretty(headline.entity_name || "Selected entity");
  const type = titleCase(headline.entity_type || "entity");
  const score = safeNumber(headline.average_trust_score || summary.average_trust);
  const riskRaw = String(headline.overall_risk_level || "not_available");
  const total = pretty(headline.total_reviews || summary.total_reviews || "-");
  const issues = formatIssueList(headline.top_issues);
  const rec = headline.entity_recommendation || recommendationFromRisk(riskRaw);

  $("#headlineCard").innerHTML = `
    <div class="headline-main">
      <div>
        <h4>${escapeHtml(entity)} — Trust Score ${escapeHtml(score)}/100</h4>
        <p><b>${escapeHtml(type)}</b>. Analysed reviews: <b>${escapeHtml(total)}</b>. Main detected issues: <b>${escapeHtml(issues)}</b>.</p>
        <p>${escapeHtml(pretty(rec))}</p>
      </div>
      <span class="risk-pill ${escapeHtml(riskRaw)}">${escapeHtml(titleCase(riskRaw))}</span>
    </div>
  `;
}

function recommendationFromRisk(risk) {
  if (risk === "high_risk") return "High caution is recommended because strong risk signals were detected.";
  if (risk === "medium_risk") return "Use with caution and review the detected issues before trusting this entity.";
  if (risk === "low_risk") return "Generally reliable, but review the listed issues if any are present.";
  return "Review the generated evidence and risk factors before making a trust decision.";
}

function renderOverallSummary(results = {}, useRag = false) {
  const summary = results.summary || {};
  const headline = results.headline || {};

  const entity = pretty(headline.entity_name || "Selected entity");
  const score = safeNumber(headline.average_trust_score || summary.average_trust);
  const risk = titleCase(headline.overall_risk_level || "not_available");
  const total = pretty(headline.total_reviews || summary.total_reviews || "-");
  const highRisk = pretty(summary.high_risk || 0);
  const issues = formatIssueList(headline.top_issues);
  const recommendation = headline.entity_recommendation || recommendationFromRisk(headline.overall_risk_level);

  $("#overallAiSummary").innerHTML = `
    <p>
      <b>${escapeHtml(entity)}</b> was analysed using the modular review trust pipeline.
      The system reviewed <b>${escapeHtml(total)}</b> review records and calculated an
      average trust score of <b>${escapeHtml(score)}/100</b>.
      The overall risk level is <b>${escapeHtml(risk)}</b>.
      Main detected issues are: <b>${escapeHtml(issues)}</b>.
    </p>
    <p>${escapeHtml(pretty(recommendation))}</p>

    <div class="clean-facts">
      <span><b>Entity</b>${escapeHtml(entity)}</span>
      <span><b>Risk</b>${escapeHtml(risk)}</span>
      <span><b>Reviews</b>${escapeHtml(total)}</span>
      <span><b>High-risk reviews</b>${escapeHtml(highRisk)}</span>
    </div>
  `;

  // const ragReason = useRag
  //   ? `<li><b>RAG evidence:</b> MiniLM embeddings and FAISS retrieve semantically similar supporting reviews.</li>`
  //   : `<li><b>RAG evidence:</b> disabled by the user for this run, so no semantic retrieval was used.</li>`;

  // $("#scoreReasoning").innerHTML = `
  //   <p>The final trust score is calculated from multiple agent outputs, not from one model only.</p>
  //   <ul class="reason-list">
  //     <li><b>Rating signal:</b> low star ratings reduce trust.</li>
  //     <li><b>Transformer sentiment:</b> negative review language increases risk.</li>
  //     <li><b>Rating prediction:</b> the text-based predicted rating is compared with the actual rating.</li>
  //     <li><b>Discrepancy check:</b> mismatch between rating and review text adds penalty.</li>
  //     <li><b>Issue mining:</b> domain-specific issues such as crash, login, payment, fake product or cleanliness are detected.</li>
  //     ${ragReason}
  //     <li><b>Risk scoring:</b> all active signals are converted into trust score, risk level and recommendation.</li>
  //   </ul>
  // `;
}

function renderIssueExamples(rows = [], useRag = false) {
  const box = $("#issueExampleCards");
  if (!rows.length) {
    box.innerHTML = `<div class="empty-state">No issue examples found in this run.</div>`;
    return;
  }

  box.innerHTML = rows.slice(0, 4).map((r, index) => {
    const entity = pretty(r.entity_name || r.entity_id || "Review entity");
    const issue = titleCase(r.primary_issue || "no_issue");
    const risk = String(r.risk_level || "");
    const reviewText = getReviewText(r);
    const explanation = r.evidence_based_explanation || r.explanation_text || r.risk_interpretation || "The explanation agent combined rating, sentiment, discrepancy and issue severity to produce this decision.";
    const evidence = useRag
      ? (r.rag_evidence_text || r.evidence_phrase || reviewText)
      : (r.evidence_phrase || reviewText);
    const evidenceLabel = useRag ? "RAG supporting evidence" : "Review evidence";
    return `
      <article class="result-card issue-card">
        <div class="card-top">
          <div>
            <p class="example-label">Example ${index + 1}</p>
            <h5>${escapeHtml(entity)}</h5>
          </div>
          <span class="chip">Trust ${escapeHtml(safeNumber(r.trust_score))}/100</span>
        </div>
        <div class="chips">
          <span class="chip ${escapeHtml(risk)}">${escapeHtml(titleCase(risk || "not_available"))}</span>
          <span class="chip ${escapeHtml(r.predicted_sentiment || "")}">${escapeHtml(titleCase(r.predicted_sentiment))}</span>
          <span class="chip">${escapeHtml(issue)}</span>
        </div>
        <div class="card-grid">
          <div class="card-mini"><span>Actual Rating</span><strong>${escapeHtml(pretty(r.rating))}</strong></div>
          <div class="card-mini"><span>Predicted Rating</span><strong>${escapeHtml(pretty(r.predicted_star_rating))}</strong></div>
          
          <div class="card-mini"><span>Discrepancy</span><strong>${escapeHtml(titleCase(r.discrepancy_level || r.discrepancy_status))}</strong></div>
          <div class="card-mini"><span>Severity</span><strong>${escapeHtml(titleCase(r.issue_severity_level))}</strong></div>
        </div>
        <div class="review-text"><b>Review text:</b> ${escapeHtml(reviewText)}</div>
        <div class="explanation"><b>Assessment rationale:</b> ${escapeHtml(explanation)}</div>
        <div class="evidence-box"><b>${escapeHtml(evidenceLabel)}:</b> ${escapeHtml(evidence)}</div>
      </article>
    `;
  }).join("");
}

function renderEntityCards(rows = []) {
  const box = $("#entityCards");
  if (!rows.length) {
    box.innerHTML = `<div class="empty-state">No entity-level summary found.</div>`;
    return;
  }
  box.innerHTML = rows.slice(0, 3).map((r) => `
    <article class="result-card entity-card">
      <div class="card-top">
        <h5>${escapeHtml(pretty(r.entity_name || r.entity_id || "Entity"))}</h5>
        <span class="chip">Avg Trust ${escapeHtml(safeNumber(r.average_trust_score))}/100</span>
      </div>
      <div class="chips">
        <span class="chip ${escapeHtml(r.overall_risk_level || "")}">${escapeHtml(titleCase(r.overall_risk_level))}</span>
        <span class="chip">${escapeHtml(titleCase(r.domain))}</span>
        <span class="chip">${escapeHtml(pretty(r.total_reviews))} reviews</span>
      </div>
      <div class="card-grid">
        <div class="card-mini"><span>Average Rating</span><strong>${escapeHtml(safeNumber(r.average_rating))}</strong></div>
        <div class="card-mini"><span>High Risk %</span><strong>${escapeHtml(percent(r.high_risk_percentage))}</strong></div>
        <div class="card-mini"><span>Mismatch %</span><strong>${escapeHtml(percent(r.mismatch_percentage))}</strong></div>
        <div class="card-mini"><span>Top Issues</span><strong>${escapeHtml(formatIssueList(r.top_issues))}</strong></div>
      </div>
      <div class="explanation">${escapeHtml(pretty(r.entity_recommendation || recommendationFromRisk(r.overall_risk_level)))}</div>
    </article>
  `).join("");
}

function renderDownloads(job, reportMeta = {}) {
  const finalLabel = reportMeta.groq_generated ? "Groq Final Summary" : "Local Final Summary";
  const files = [
    ["final_report", finalLabel],
    // ["review_results", "Review Results CSV"],
    // ["entity_summary", "Entity Summary CSV"],
    // ["orchestrator_state", "Orchestrator State"],
    // ["prepared_dataset", "Prepared Dataset"]
  ];

  const artifacts = job.input_artifacts || {};
  if (artifacts.raw_scraper_reviews) files.push(["raw_scraper_reviews", "Raw Scraped Reviews"]);
  if (artifacts.scraper_metadata) files.push(["scraper_metadata", "Scraper Metadata"]);
  if (artifacts.scraper_prepared_dataset) files.push(["scraper_prepared_dataset", "Scraper Common-Schema CSV"]);

  $("#downloadButtons").innerHTML = files
    .map(([key, label]) => `<a href="/api/download/${job.job_id}/${key}" target="_blank">${label}</a>`)
    .join("");
}

function renderResults(job) {
  const results = job.results || {};
  const summary = results.summary || {};
  const reportMeta = results.report_meta || {};
  const runtime = job.runtime_options || {};
  const useRag = runtime.use_rag === true || reportMeta.use_rag === true;
  const groqGenerated = reportMeta.groq_generated === true;

  $("#results-section").classList.remove("hidden");
  $("#report-section").classList.remove("hidden");

  if ($("#reportTitle")) {
    $("#reportTitle").textContent = reportMeta.report_title || (groqGenerated ? "Final Groq Trust/Risk Summary" : "Local Explainable Trust/Risk Summary");
  }
  if ($("#reportEyebrow")) {
    $("#reportEyebrow").textContent = reportMeta.report_eyebrow || (groqGenerated ? "LLM Finalisation" : "Local Finalisation");
  }
  if ($("#evidenceSectionTitle")) {
    $("#evidenceSectionTitle").textContent = useRag ? "Key Review Examples and RAG Evidence" : "Key Review Examples";
  }
  if ($("#evidenceSectionSubtitle")) {
    $("#evidenceSectionSubtitle").textContent = useRag
      ? "Important examples with retrieved semantic evidence"
      : "Important examples from the analysed reviews; RAG was disabled";
  }

  $("#metricReviews").textContent = Number(summary.total_reviews || 0).toLocaleString();
  $("#metricEntities").textContent = Number(summary.total_entities || 0).toLocaleString();
  $("#metricTrust").textContent = safeNumber(summary.average_trust);
  $("#metricHighRisk").textContent = Number(summary.high_risk || 0).toLocaleString();

  renderHeadline(results.headline || {}, summary);
  renderOverallSummary(results, useRag);
  renderBars("#riskBars", results.distributions?.risk || []);
  renderBars("#domainBars", results.distributions?.domain || []);
  renderBars("#issueBars", results.distributions?.issue || []);
  renderIssueExamples(results.issue_examples || results.review_rows || [], useRag);
  renderEntityCards(results.entity_rows || []);

  window.currentReportText = results.final_report || "";
  $("#finalReport").innerHTML = renderReportHtml(window.currentReportText);
  renderDownloads(job, reportMeta);
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}?_=${Date.now()}`, { cache: "no-store" });
    const job = await res.json();
    if (!res.ok) throw new Error(job.error || "Could not read job status.");

    setStatus(job.status, job.progress || 0);
    renderTrace(job.trace || []);

    const lastTrace = (job.trace || []).at(-1);
    const elapsed = jobStartedAt ? Math.max(0, Math.round((Date.now() - jobStartedAt) / 1000)) : 0;
    if (lastTrace && $("#loaderText")) {
      $("#loaderText").textContent = `${pretty(lastTrace.message)} (${elapsed}s elapsed)`;
    }

    if (job.status === "completed") {
      clearInterval(pollTimer);
      pollTimer = null;
      $("#runBtn").disabled = false;
      $("#runBtn").textContent = "Run Analysis";
      $("#loadingOverlay").classList.add("hidden");
      renderResults(job);
      toast("Analysis completed successfully.");
    }

    if (job.status === "failed") {
      clearInterval(pollTimer);
      pollTimer = null;
      $("#runBtn").disabled = false;
      $("#runBtn").textContent = "Run Analysis";
      $("#loadingOverlay").classList.add("hidden");
      toast(`Run failed: ${job.error || "unknown error"}`);
    }
  } catch (error) {
    clearInterval(pollTimer);
    pollTimer = null;
    $("#runBtn").disabled = false;
    $("#runBtn").textContent = "Run Analysis";
    $("#loadingOverlay").classList.add("hidden");
    setStatus("failed", 100);
    toast(error.message || "Job status connection failed.");
  }
}


async function loadHealth() {
  try {
    const res = await fetch(`/api/health?_=${Date.now()}`, { cache: "no-store" });
    const data = await res.json();
    const models = data.models || {};
    const scrapers = data.scrapers || {};
    let mapsReady = "Maps bridge unavailable";
    if (scrapers.google_maps_bridge && scrapers.google_maps_cdp_ready) {
      mapsReady = "Maps bridge ready; signed-in Chrome connected";
    } else if (scrapers.google_maps_bridge) {
      mapsReady = scrapers.error || "Maps bridge ready; start Google Maps Chrome session";
    } else if (scrapers.error) {
      mapsReady = scrapers.error;
    }
    const groqReady = data.groq_configured && data.groq_client_ready;
    const modelCount = Object.values(models).filter(Boolean).length;
    $("#healthText").textContent = `${data.status}; Groq ${groqReady ? "ready" : "setup required"}; models ${modelCount}/3; ${mapsReady}`;
  } catch {
    $("#healthText").textContent = "health check unavailable";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadHealth();
  setMode($("#modeInput").value || "csv");

  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
  $("#resetBtn").addEventListener("click", () => location.reload());

  $("#copyReportBtn").addEventListener("click", async () => {
    await navigator.clipboard.writeText(plainReportText(window.currentReportText || $("#finalReport").textContent || ""));
    toast("Summary copied.");
  });

  $("#analysisForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    form.set("use_rag", e.currentTarget.querySelector("[name='use_rag']").checked ? "true" : "false");
    form.set("use_groq", e.currentTarget.querySelector("[name='use_groq']").checked ? "true" : "false");

    if (form.get("mode") === "google_maps_url") {
      try {
        const healthRes = await fetch(`/api/google-maps/preflight?_=${Date.now()}`, { cache: "no-store" });
        const status = await healthRes.json();
        if (!healthRes.ok || !status.ready) {
          throw new Error(status.error || "Google Maps collector pre-flight failed.");
        }
      } catch (err) {
        toast(err.message || "Google Maps collector pre-flight check failed.");
        return;
      }
    }

    $("#runBtn").disabled = true;
    $("#runBtn").textContent = "Running...";
    jobStartedAt = Date.now();
    const selectedMode = form.get("mode");
    $("#loaderText").textContent = selectedMode === "google_maps_url"
      ? "Connecting to signed-in Chrome and collecting Google Maps reviews..."
      : "The orchestrator is coordinating specialised agents.";
    $("#loadingOverlay").classList.remove("hidden");
    $("#results-section").classList.add("hidden");
    $("#report-section").classList.add("hidden");
    setStatus("running", 5);
    renderTrace([{ title: "Orchestrator running", message: "Input accepted. Running final orchestrator.", output: {} }]);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to start job");
      currentJobId = data.job_id;
      $("#jobPill").textContent = `Job: ${currentJobId}`;
      $("#jobPill").classList.remove("hidden");
      pollTimer = setInterval(() => pollJob(currentJobId), 2500);
      pollJob(currentJobId);
    } catch (err) {
      $("#runBtn").disabled = false;
      $("#runBtn").textContent = "Run Analysis";
      $("#loadingOverlay").classList.add("hidden");
      setStatus("failed", 100);
      toast(err.message);
    }
  });
});
