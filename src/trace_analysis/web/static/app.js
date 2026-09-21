const state = {
  taxonomy: null,
  searchIndex: [],
  selected: null,
  nodes: new Map(),
  plot: { x: 0, y: 0, scale: 1 },
  pain: {
    loaded: false, loading: null, metadata: null, summary: null,
    cases: [], filtered: [], selected: null, reviews: new Map(),
    reviewApiAvailable: false, canReview: false, reviewSignInUrl: null,
  },
};

const painStatusLabels = { confirmed: "确认痛点", review: "需人工复核", excluded: "排除候选" };

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
}[char]));

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function loadGzipJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  if (typeof DecompressionStream === "undefined") throw new Error("当前浏览器不支持 gzip 展开");
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

async function loadIncrementalPainData() {
  try {
    const catalog = await loadJson("data/pain-report/catalog.json");
    const batches = catalog.batches || [];
    const payloads = await Promise.all(batches.map(async (batch) => {
      const [metadata, summary, cases] = await Promise.all([
        loadJson(`data/pain-report/${batch.metadata}`),
        loadJson(`data/pain-report/${batch.summary}`),
        loadJson(`data/pain-report/${batch.case_index}`),
      ]);
      return { batch, metadata, summary, cases };
    }));
    const cases = payloads.flatMap(({ batch, cases: rows }) => rows.map((item) => ({
      ...item,
      batch_id: item.batch_id || batch.batch_id,
      analysis_run_id: item.analysis_run_id || batch.active_run_id,
      status: item.status || item.pain_judgment,
      status_label: item.status_label || item.pain_judgment,
      model_label: item.model_label || (item.models || []).join(" + "),
      detail_path: `data/pain-report/${batch.cases_pattern.replace("{case_id}", encodeURIComponent(item.case_id))}`,
      transcript_path: batch.transcripts_pattern
        ? `data/pain-report/${batch.transcripts_pattern.replace("{case_id}", encodeURIComponent(item.case_id))}`
        : null,
    })));
    const globalSummary = await loadJson("data/pain-report/summary.json");
    return {
      metadata: {
        batch_id: `${batches.length} 个批次`,
        analysis_run_id: "各批次 Active Run",
        generated_at: catalog.updated_at,
        methodology: "先用完整 User Turn 筛选候选 Episode，再结合 Agent、Tool 和结果证据验证；页面合并所有已发布批次的 Active Run。",
        comparison_caveat: "跨模型分布是观察性数据；未控制任务和环境时不能直接解释为模型能力因果差异。",
      },
      summary: {
        total_trace_count: cases.length,
        verified_episode_count: cases.length,
        episode_status_counts: globalSummary.pain_judgment_counts || {},
        model_summary: Object.entries(globalSummary.model_counts || {}).map(([model, count]) => ({
          model, model_label: model, trace_count: count,
          confirmed_trace_count: cases.filter((item) => item.status === "confirmed" && (item.models || []).includes(model)).length,
          review_trace_count: cases.filter((item) => item.status === "review" && (item.models || []).includes(model)).length,
          excluded_trace_count: cases.filter((item) => item.status === "excluded" && (item.models || []).includes(model)).length,
        })),
      },
      cases,
    };
  } catch (error) {
    if (!String(error.message || error).includes("catalog.json")) throw error;
    const [metadata, summary, cases] = await Promise.all([
      loadJson("data/pain-report/metadata.json"),
      loadJson("data/pain-report/summary.json"),
      loadJson("data/pain-report/case-index.json"),
    ]);
    return {
      metadata, summary,
      cases: cases.map((item) => ({
        ...item,
        batch_id: item.batch_id || metadata.batch_id,
        analysis_run_id: item.analysis_run_id || metadata.analysis_run_id,
        detail_path: `data/pain-report/cases/${encodeURIComponent(item.episode_id)}.json`,
      })),
    };
  }
}

function applyHumanReview(item, review) {
  if (!item.model_status) {
    item.model_status = item.status;
    item.model_status_label = item.status_label || painStatusLabels[item.status] || item.status;
  }
  item.human_review = review || null;
  if (review) {
    item.status = review.decision;
    item.status_label = `${painStatusLabels[review.decision] || review.decision}（人工）`;
    item.attribution_override = review.attribution_override || null;
  } else {
    item.status = item.model_status;
    item.status_label = item.model_status_label;
    item.attribution_override = null;
  }
  return item;
}

async function loadHumanReviews(cases) {
  try {
    const payload = await loadJson("/api/pain-reviews");
    state.pain.reviews = new Map((payload.reviews || []).map((review) => [review.case_id, review]));
    state.pain.reviewApiAvailable = true;
    state.pain.canReview = payload.can_review !== false;
    state.pain.reviewSignInUrl = payload.sign_in_url || null;
  } catch {
    state.pain.reviews = new Map();
    state.pain.reviewApiAvailable = false;
    state.pain.canReview = false;
    state.pain.reviewSignInUrl = null;
  }
  cases.forEach((item) => applyHumanReview(item, state.pain.reviews.get(item.case_id)));
}

function isLeaf(node) { return node.is_leaf || !node.children || node.children.length === 0; }

function treeItem(node, depth = 0) {
  state.nodes.set(node.capability_id, node);
  const li = document.createElement("li");
  li.dataset.capabilityId = node.capability_id;
  const row = document.createElement("div");
  row.className = "tree-row";
  const childList = document.createElement("ul");
  const leaf = isLeaf(node);

  function setCollapsed(collapsed) {
    childList.hidden = collapsed;
    li.classList.toggle("is-collapsed", collapsed && !leaf);
    toggle.setAttribute("aria-expanded", String(!collapsed));
  }

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "tree-toggle";
  toggle.setAttribute("aria-label", leaf ? "叶子能力" : `展开或收起 ${node.name}`);
  toggle.textContent = "";
  if (!leaf) setCollapsed(depth >= 2);
  toggle.disabled = leaf;
  toggle.addEventListener("click", () => {
    setCollapsed(!childList.hidden);
  });

  const name = document.createElement("button");
  name.type = "button";
  name.className = "tree-name";
  name.innerHTML = depth === 0
    ? escapeHtml(node.name)
    : `${escapeHtml(node.name)} <span class="tree-count">(${Number(node.query_count || 0)})</span>`;
  name.addEventListener("click", () => leaf ? selectLeaf(node) : toggle.click());
  row.append(toggle, name);
  li.append(row);
  for (const child of node.children || []) childList.append(treeItem(child, depth + 1));
  if (!leaf) li.append(childList);
  return li;
}

function renderTree(root) {
  state.nodes.clear();
  const list = document.createElement("ul");
  list.append(treeItem(root));
  $("tree").replaceChildren(list);
}

function evidenceLabel(type) {
  return ({ requirement: "用户直接需求", requirement_acceptance: "用户接受/确认", negotiation_proposal: "Agent 提议", fulfillment: "Agent 执行或结果" })[type] || type || "其他证据";
}

function originLabel(value) {
  return ({ direct: "用户直接提出", negotiated: "协商确认", unknown: "来源未标记" })[value] || value;
}

function harnessLabel(value) {
  return ({ claude_code: "Claude Code", unknown: "未知 Harness" })[value] || value;
}

function mappingHtml(item, index) {
  const evidence = (item.evidence || []).map((ev) => `
    <div class="evidence ${escapeHtml(ev.evidence_type)}">
      <div class="evidence-type">${escapeHtml(evidenceLabel(ev.evidence_type))}</div>
      <p class="quote">${escapeHtml(ev.quote)}</p>
      <div class="id-list"><div class="id-row"><span class="id-label">Event ID</span><code class="id-value">${escapeHtml(ev.event_id || "—")}</code></div></div>
    </div>`).join("");
  return `<section class="mapping">
      <h4 class="mapping-title">能力映射 ${index + 1}</h4>
      <div class="case-facts">
        <span class="fact"><span>需求来源</span><strong>${escapeHtml(originLabel(item.requirement_origin))}</strong></span>
        <span class="fact"><span>满足情况</span><strong>${escapeHtml(item.fulfillment || "unknown")}</strong></span>
        <span class="fact"><span>证据约束</span><strong>${escapeHtml(item.entailment_audit || "未审核")}</strong></span>
      </div>
      ${item.normalized_need ? `<p class="normalized-need"><strong>需求识别</strong><br>${escapeHtml(item.normalized_need)}</p>` : ""}
      <div class="evidence-chain"><h4>证据链</h4>${evidence}</div>
      <div class="id-list">
        <div class="id-row"><span class="id-label">Trace ID</span><code class="id-value">${escapeHtml(item.trace_id || "—")}</code></div>
        <div class="id-row"><span class="id-label">Turn ID</span><code class="id-value">${escapeHtml((item.turn_ids || []).join(", ") || "—")}</code></div>
        <div class="id-row"><span class="id-label">Case ID</span><code class="id-value">${escapeHtml(item.case_id || "—")}</code></div>
      </div>
    </section>`;
}

function groupCases(cases) {
  const groups = new Map();
  for (const item of cases) {
    const query = (item.user_query || []).join("\n") || item.normalized_need || "未找到可展示的用户原话";
    if (!groups.has(query)) groups.set(query, []);
    groups.get(query).push(item);
  }
  return [...groups.entries()];
}

function queryGroupHtml([query, mappings], index) {
  return `<details class="query-group" ${index === 0 ? "open" : ""}>
    <summary><span class="query-label">用户 Query ${index + 1} · ${mappings.length} 条能力映射</span></summary>
    <div class="query-box">${escapeHtml(query)}</div>
    ${mappings.map(mappingHtml).join("")}
  </details>`;
}

async function selectLeaf(node) {
  state.selected = node.capability_id;
  document.querySelectorAll(".tree-name").forEach((el) => el.removeAttribute("aria-current"));
  const row = document.querySelector(`[data-capability-id="${CSS.escape(node.capability_id)}"] .tree-name`);
  if (row) row.setAttribute("aria-current", "true");
  $("empty-detail").hidden = true;
  const detail = $("detail-content");
  detail.hidden = false;
  detail.innerHTML = `<p class="muted">正在读取案例…</p>`;
  try {
    const cases = await loadJson(`data/cases/${encodeURIComponent(node.capability_id)}.json`);
    const groups = groupCases(cases);
    detail.innerHTML = `
      <p class="path">${(node.path || []).map(escapeHtml).join(" › ")}</p>
      <h2>${escapeHtml(node.name)}</h2>
      <p class="definition">${escapeHtml(node.definition || "暂无释义")}</p>
      <div class="detail-meta">
        <span><strong>${Number(node.query_count || 0)}</strong> 个用户 Query</span>
        <span><strong>${cases.length}</strong> 个能力映射案例</span>
      </div>
      <div class="case-list"><h3>全部支持案例</h3>${groups.length ? groups.map(queryGroupHtml).join("") : "<p class='muted'>该节点没有可发布案例。</p>"}</div>`;
    if (window.innerWidth <= 860) $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    detail.innerHTML = `<p class="error">案例读取失败：${escapeHtml(error.message)}</p>`;
  }
}

function painBadge(value, label) {
  return `<span class="pain-badge ${escapeHtml(value || "unknown")}">${escapeHtml(label || value || "未知")}</span>`;
}

function uniqueOptions(values) {
  return [...new Map(values.filter((item) => item && item.value).map((item) => [item.value, item])).values()]
    .sort((left, right) => left.label.localeCompare(right.label, "zh-CN"));
}

function fillPainSelect(id, items) {
  const select = $(id);
  const first = select.options[0];
  select.replaceChildren(first, ...uniqueOptions(items).map((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    return option;
  }));
}

function painCaseCard(item) {
  const gaps = (item.capability_gaps || []).map((gap) => `<span>${escapeHtml(gap)}</span>`).join("");
  return `<button type="button" class="pain-case-card" data-pain-episode-id="${escapeHtml(item.case_id || item.episode_id)}">
    <span class="pain-card-top"><span>Episode ${item.display_number}</span>${painBadge(item.status, item.status_label)}</span>
    <strong>${escapeHtml(item.title)}</strong>
    <span class="pain-card-summary">${escapeHtml(item.summary)}</span>
    <span class="pain-card-meta"><span>${escapeHtml(item.model_label)}</span><span>${escapeHtml(item.batch_id || "unknown batch")}</span><span>${escapeHtml(item.outcome_label)}</span></span>
    ${gaps ? `<span class="pain-card-gaps">${gaps}</span>` : ""}
  </button>`;
}

function renderPainList() {
  const list = $("pain-list");
  $("pain-result-count").textContent = `显示 ${state.pain.filtered.length} / ${state.pain.cases.length} 个 Episode`;
  if (!state.pain.filtered.length) {
    list.innerHTML = `<p class="muted">没有符合当前筛选条件的案例。</p>`;
    return;
  }
  list.innerHTML = state.pain.filtered.map(painCaseCard).join("");
  list.querySelectorAll("[data-pain-episode-id]").forEach((button) => {
    button.addEventListener("click", () => selectPainCase(button.dataset.painEpisodeId));
  });
  if (state.pain.selected) {
    const selected = list.querySelector(`[data-pain-episode-id="${CSS.escape(state.pain.selected)}"]`);
    if (selected) selected.setAttribute("aria-current", "true");
  }
}

function applyPainFilters() {
  const query = $("pain-search").value.trim().toLocaleLowerCase();
  const batch = $("pain-batch-filter").value;
  const status = $("pain-status-filter").value;
  const model = $("pain-model-filter").value;
  const harness = $("pain-harness-filter").value;
  const attribution = $("pain-attribution-filter").value;
  const outcome = $("pain-outcome-filter").value;
  const gap = $("pain-gap-filter").value;
  state.pain.filtered = state.pain.cases.filter((item) => {
    const haystack = [item.batch_id, item.trace_id, item.episode_id, item.title, item.summary, item.model_label,
      ...(item.harnesses || []), ...(item.harnesses || []).map(harnessLabel), ...(item.capability_gaps || [])]
      .join("\n").toLocaleLowerCase();
    return (!query || haystack.includes(query))
      && (!batch || item.batch_id === batch)
      && (!status || item.status === status)
      && (!model || (item.models || []).includes(model))
      && (!harness || (item.harnesses || []).includes(harness))
      && (!attribution || item.attribution === attribution)
      && (!outcome || item.outcome === outcome)
      && (!gap || (item.capability_gaps || []).includes(gap));
  });
  renderPainSummary(state.pain.filtered);
  renderPainList();
}

function renderPainSummary(cases) {
  const traceIds = new Set(cases.map((item) => item.trace_id).filter(Boolean));
  $("pain-total-count").textContent = traceIds.size;
  $("pain-verified-count").textContent = cases.length;
  $("pain-confirmed-count").textContent = cases.filter((item) => item.status === "confirmed").length;
  $("pain-review-count").textContent = cases.filter((item) => item.status === "review").length;

  const byModel = new Map();
  for (const item of cases) {
    for (const model of item.models || ["unknown"]) {
      if (!byModel.has(model)) byModel.set(model, new Map());
      const traces = byModel.get(model);
      if (!traces.has(item.trace_id)) traces.set(item.trace_id, new Set());
      traces.get(item.trace_id).add(item.status);
    }
  }
  const rows = [...byModel.entries()].map(([model, traces]) => {
    const statuses = [...traces.values()];
    return {
      model,
      traceCount: traces.size,
      confirmed: statuses.filter((values) => values.has("confirmed")).length,
      review: statuses.filter((values) => !values.has("confirmed") && values.has("review")).length,
      excluded: statuses.filter((values) => !values.has("confirmed") && !values.has("review") && values.has("excluded")).length,
    };
  }).sort((left, right) => right.traceCount - left.traceCount || left.model.localeCompare(right.model));
  $("pain-model-table").innerHTML = rows.map((item) => `<tr>
    <td>${escapeHtml(item.model)}</td><td>${item.traceCount}</td><td>${item.confirmed}</td><td>${item.review}</td><td>${item.excluded}</td>
  </tr>`).join("") || `<tr><td colspan="5" class="muted">没有模型统计。</td></tr>`;
}

function painEvidenceHtml(items) {
  if (!items || !items.length) return `<p class="muted">没有可发布的关键证据。</p>`;
  return items.map((item) => `<article class="pain-evidence">
    <p class="pain-evidence-kind">${escapeHtml(item.kind)}</p>
    <blockquote>${escapeHtml(item.quote)}</blockquote>
    <div class="id-list">
      <div class="id-row"><span class="id-label">Turn ID</span><code class="id-value">${escapeHtml(item.turn_id || "—")}</code></div>
      <div class="id-row"><span class="id-label">Event ID</span><code class="id-value">${escapeHtml(item.event_id || "—")}</code></div>
    </div>
  </article>`).join("");
}

function painRequirementsHtml(requirements) {
  if (!requirements || !requirements.length) return `<p class="muted">未提取到结构化用户要求。</p>`;
  return requirements.map((item) => `<article class="pain-requirement">
    <p>${escapeHtml(item.text)}</p>
    <div class="case-facts">
      <span class="fact"><span>来源</span><strong>${escapeHtml(item.origin_label)}</strong></span>
      <span class="fact"><span>完成情况</span><strong>${escapeHtml(item.status_label)}</strong></span>
    </div>
  </article>`).join("");
}

function painGapsHtml(gaps) {
  if (!gaps || !gaps.length) return `<p class="muted">未确认能力缺口。</p>`;
  return gaps.map((gap) => `<article class="pain-gap-detail">
    <h4>${escapeHtml(gap.capability)}</h4>
    <p>${escapeHtml(gap.reason || "—")}</p>
    ${painEvidenceHtml(gap.evidence)}
  </article>`).join("");
}

function painTranscriptEventHtml(event) {
  const labels = { assistant_message: "Assistant 回复", tool_call: "Tool Call", tool_result: "Tool Result", error: "Error" };
  const label = labels[event.type] || event.role || event.type;
  return `<article class="transcript-event transcript-${escapeHtml(event.role)}"><div class="transcript-event-meta"><strong>${escapeHtml(label)}</strong><code>${escapeHtml(event.event_id || "—")}</code></div><pre>${escapeHtml(event.content || "—")}</pre>${event.truncated ? `<p class="muted">公开视图已截断；原事件共 ${Number(event.original_char_count || 0).toLocaleString()} 字符。</p>` : ""}</article>`;
}

function painTranscriptHtml(transcript) {
  const turns = transcript.turns || [];
  if (!turns.length) return `<p class="muted">该 Episode 没有可发布的真实 User Turn。</p>`;
  return `<p class="muted">${escapeHtml(transcript.publication_note || "")}</p>${turns.map((turn, index) => `<details class="transcript-turn"><summary><span>User Turn ${index + 1}</span><span class="transcript-user-preview">${escapeHtml(turn.user?.content || "—")}</span></summary><div class="transcript-user"><div class="transcript-event-meta"><strong>User</strong><code>${escapeHtml(turn.turn_id || "—")}</code></div><pre>${escapeHtml(turn.user?.content || "—")}</pre></div><div class="transcript-agent-events">${(turn.agent_events || []).length ? turn.agent_events.map(painTranscriptEventHtml).join("") : `<p class="muted">该 User Turn 后没有可观察的 Agent 响应。</p>`}</div></details>`).join("")}`;
}

function humanReviewHtml(item) {
  if (!state.pain.reviewApiAvailable) {
    return `<section class="pain-detail-section human-review-panel">
      <h3>人工复核</h3>
      <p class="muted">当前是只读预览。请使用本地审核服务打开页面后提交审核。</p>
    </section>`;
  }
  if (!state.pain.canReview) {
    const action = state.pain.reviewSignInUrl
      ? `<a class="secondary-button review-sign-in" href="${escapeHtml(state.pain.reviewSignInUrl)}">登录后审核</a>`
      : `<p class="muted">你可以查看人工结论，但当前账号没有提交审核的权限。</p>`;
    return `<section class="pain-detail-section human-review-panel">
      <h3>人工复核</h3>${action}
    </section>`;
  }
  const review = item.human_review;
  const decision = review?.decision || item.status || "review";
  return `<section class="pain-detail-section human-review-panel">
    <h3>人工复核</h3>
    <div class="review-comparison">
      <span><small>模型判断</small><strong>${escapeHtml(item.model_status_label || painStatusLabels[item.model_status] || item.model_status)}</strong></span>
      <span><small>当前采用</small><strong>${escapeHtml(item.status_label)}</strong></span>
    </div>
    <form id="pain-review-form">
      <label><span>人工结论</span><select id="pain-review-decision" required>
        <option value="confirmed"${decision === "confirmed" ? " selected" : ""}>确认痛点</option>
        <option value="excluded"${decision === "excluded" ? " selected" : ""}>排除候选</option>
        <option value="review"${decision === "review" ? " selected" : ""}>仍需复核</option>
      </select></label>
      <label><span>审核理由</span><textarea id="pain-review-note" rows="4" required placeholder="写明采纳或推翻模型判断的证据与理由">${escapeHtml(review?.note || "")}</textarea></label>
      <div class="review-actions">
        <button id="pain-review-submit" type="submit">${review ? "更新审核" : "保存审核"}</button>
        <span id="pain-review-message" class="muted" aria-live="polite">${review ? `版本 ${Number(review.revision)} · ${escapeHtml(review.created_at)}` : "保存后会自动同步到独立 JSONL"}</span>
      </div>
    </form>
  </section>`;
}

async function submitHumanReview(event, selected) {
  event.preventDefault();
  const button = $("pain-review-submit");
  const message = $("pain-review-message");
  const note = $("pain-review-note").value.trim();
  if (!note) {
    message.textContent = "请填写审核理由。";
    return;
  }
  button.disabled = true;
  message.textContent = "正在保存并同步…";
  try {
    const response = await fetch("/api/pain-reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_id: selected.case_id,
        decision: $("pain-review-decision").value,
        note,
        attribution_override: selected.human_review?.attribution_override || null,
        expected_revision: selected.human_review?.revision || 0,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
    state.pain.reviews.set(selected.case_id, payload.review);
    applyHumanReview(selected, payload.review);
    applyPainFilters();
    await selectPainCase(selected.case_id || selected.episode_id);
  } catch (error) {
    button.disabled = false;
    message.textContent = `保存失败：${error.message}`;
  }
}

async function selectPainCase(episodeId) {
  state.pain.selected = episodeId;
  document.querySelectorAll("[data-pain-episode-id]").forEach((button) => {
    if (button.dataset.painEpisodeId === episodeId) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  });
  $("pain-empty-detail").hidden = true;
  const detail = $("pain-detail-content");
  detail.hidden = false;
  detail.innerHTML = `<p class="muted">正在读取 Episode 证据…</p>`;
  try {
    const selected = state.pain.cases.find((item) => (item.case_id || item.episode_id) === episodeId);
    const item = await loadJson(selected?.detail_path || `data/pain-report/cases/${encodeURIComponent(episodeId)}.json`);
    applyHumanReview(item, selected?.human_review || null);
    detail.innerHTML = `
      <div class="pain-detail-heading">
        <div><p class="eyebrow">EPISODE ${item.display_number}</p><h2>${escapeHtml(item.title)}</h2></div>
        ${painBadge(item.status, item.status_label)}
      </div>
      <p class="pain-lead">${escapeHtml(item.summary)}</p>
      <div class="pain-facts">
        <span><small>被分析模型</small><strong>${escapeHtml(item.model_label)}</strong></span>
        <span><small>Harness</small><strong>${escapeHtml((item.harnesses || []).map(harnessLabel).join(", "))}</strong></span>
        <span><small>任务结果</small><strong>${escapeHtml(item.outcome_label)}</strong></span>
        <span><small>归因类型</small><strong>${escapeHtml(item.attribution_label)}</strong></span>
        <span><small>初始 Query</small><strong>${escapeHtml(item.initial_query_clarity_label)}</strong></span>
        <span><small>分析价值</small><strong>${escapeHtml(item.analysis_value_label)}</strong></span>
      </div>
      <section class="pain-detail-section"><h3>用户目标</h3><p>${escapeHtml(item.goal || "无法完整还原")}</p><p class="muted">${escapeHtml(item.goal_reason || "")}</p></section>
      <section class="pain-detail-section"><h3>初始 Query 判断</h3><p>${escapeHtml(item.initial_query_clarity_reason || "—")}</p></section>
      <section class="pain-detail-section"><h3>用户要求及完成情况</h3>${painRequirementsHtml(item.requirements)}</section>
      <section class="pain-detail-section"><h3>Agent 行为验证</h3>
        <h4>观察到的失败</h4><p>${escapeHtml(item.agent_failure || "未观察到可确认的 Agent 失败。")}</p>
        <h4>归因理由</h4><p>${escapeHtml(item.agent_related_reason || "现有证据不足。")}</p>
      </section>
      <section class="pain-detail-section"><h3>能力缺口</h3>${painGapsHtml(item.capability_gap_details)}</section>
      <section class="pain-detail-section"><h3>Episode 原始交互</h3><div id="pain-transcript"><button class="secondary-button" id="load-pain-transcript" type="button">加载 User Turn 与 Agent 响应</button></div></section>
      <section class="pain-detail-section benchmark-note"><h3>Benchmark 构造备注</h3><p>${escapeHtml(item.benchmark_note)}</p></section>
      ${humanReviewHtml(item)}
      <section class="pain-detail-section"><h3>溯源标识</h3><div class="id-list">
        <div class="id-row"><span class="id-label">Trace ID</span><code class="id-value">${escapeHtml(item.trace_id)}</code></div>
        <div class="id-row"><span class="id-label">Batch ID</span><code class="id-value">${escapeHtml(item.batch_id || "—")}</code></div>
        <div class="id-row"><span class="id-label">Episode ID</span><code class="id-value">${escapeHtml(item.episode_id)}</code></div>
        <div class="id-row"><span class="id-label">Query Turn</span><code class="id-value">${escapeHtml(item.query_turn_id || "—")}</code></div>
        <div class="id-row"><span class="id-label">Start Turn</span><code class="id-value">${escapeHtml(item.episode_boundary?.start_turn_id || "—")}</code></div>
        <div class="id-row"><span class="id-label">End Turn</span><code class="id-value">${escapeHtml(item.episode_boundary?.end_turn_id || "—")}</code></div>
      </div></section>
      <p class="privacy-note">${escapeHtml(item.privacy_note)}</p>`;
    const reviewForm = $("pain-review-form");
    if (reviewForm && selected) reviewForm.addEventListener("submit", (event) => submitHumanReview(event, selected));
    const transcriptButton = $("load-pain-transcript");
    if (transcriptButton) transcriptButton.addEventListener("click", async () => {
      const container = $("pain-transcript"); container.innerHTML = `<p class="muted">正在加载 Episode 交互…</p>`;
      try { const transcriptPath = selected?.transcript_path || item.transcript_path; if (!transcriptPath) throw new Error("该历史发布版本没有 transcript_path"); container.innerHTML = painTranscriptHtml(await loadGzipJson(transcriptPath)); }
      catch (error) { container.innerHTML = `<p class="error">Episode 交互读取失败：${escapeHtml(error.message)}</p>`; }
    });
    if (window.innerWidth <= 980) $("pain-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    detail.innerHTML = `<p class="error">Episode 读取失败：${escapeHtml(error.message)}</p>`;
  }
}

async function initializePain() {
  if (state.pain.loaded) return;
  if (state.pain.loading) return state.pain.loading;
  $("pain-list").innerHTML = `<p class="muted">正在读取痛点分析数据…</p>`;
  state.pain.loading = (async () => {
    try {
      const { metadata, summary, cases } = await loadIncrementalPainData();
      state.pain.metadata = metadata;
      state.pain.summary = summary;
      state.pain.cases = cases;
      await loadHumanReviews(cases);
      state.pain.filtered = cases;
      state.pain.loaded = true;
      $("pain-methodology").textContent = metadata.methodology;
      $("pain-version-line").textContent = `Batch：${metadata.batch_id} · Run：${metadata.analysis_run_id} · 生成：${metadata.generated_at}`;
      $("pain-caveat").textContent = metadata.comparison_caveat;
      renderPainSummary(cases);
      fillPainSelect("pain-batch-filter", cases.map((item) => ({ value: item.batch_id, label: item.batch_id })));
      fillPainSelect("pain-status-filter", cases.map((item) => ({ value: item.status, label: item.status_label })));
      fillPainSelect("pain-model-filter", cases.flatMap((item) => (item.models || []).map((model) => ({ value: model, label: item.model_label }))));
      fillPainSelect("pain-harness-filter", cases.flatMap((item) => (item.harnesses || ["unknown"]).map((harness) => ({ value: harness, label: harnessLabel(harness) }))));
      fillPainSelect("pain-attribution-filter", cases.map((item) => ({ value: item.attribution, label: item.attribution_label })));
      fillPainSelect("pain-outcome-filter", cases.map((item) => ({ value: item.outcome, label: item.outcome_label })));
      fillPainSelect("pain-gap-filter", cases.flatMap((item) => (item.capability_gaps || []).map((gap) => ({ value: gap, label: gap }))));
      renderPainList();
      const firstConfirmed = cases.find((item) => item.status === "confirmed") || cases[0];
      if (firstConfirmed) selectPainCase(firstConfirmed.case_id || firstConfirmed.episode_id);
    } catch (error) {
      $("pain-list").innerHTML = `<p class="error">痛点分析数据读取失败：${escapeHtml(error.message)}</p>`;
      $("pain-version-line").textContent = "痛点分析快照不可用";
    } finally {
      state.pain.loading = null;
    }
  })();
  return state.pain.loading;
}

function switchView(view) {
  document.querySelectorAll("[data-view]").forEach((section) => { section.hidden = section.dataset.view !== view; });
  document.querySelectorAll("[data-view-button]").forEach((button) => {
    if (button.dataset.viewButton === view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (view === "pain") initializePain();
}

function plotColor(hue, depth, maxDepth) {
  const lightness = 72 - (depth / Math.max(maxDepth, 1)) * 34;
  return `hsl(${hue} 52% ${lightness}%)`;
}

function maxTreeDepth(node, depth = 0) {
  return Math.max(depth, ...(node.children || []).map((child) => maxTreeDepth(child, depth + 1)));
}

let plotZoom = null;
let plotSvgSelection = null;
let plotSimulation = null;

function resetPlot() {
  if (plotZoom && plotSvgSelection) plotSvgSelection.transition().duration(350).call(plotZoom.transform, d3.zoomIdentity);
  if (plotSimulation) plotSimulation.alpha(.55).restart();
}

function renderCapabilityPlot(root) {
  if (!window.d3) {
    $("capability-plot").innerHTML = `<p class="error">能力圆图组件加载失败，请检查网络连接后刷新页面。</p>`;
    return;
  }
  const svg = $("plot-svg");
  const viewport = $("plot-viewport");
  const tooltip = $("plot-tooltip");
  const top = root.children || [];
  const hues = [215, 24, 278, 148, 342, 52, 188, 315];
  const maxDepth = maxTreeDepth(root);
  viewport.replaceChildren();
  const topHue = new Map(top.map((node, index) => [node.capability_id, hues[index % hues.length]]));
  const nodes = [];
  const links = [];
  function collect(node, parent = null, depth = 0, topId = null) {
    const resolvedTop = depth === 1 ? node.capability_id : topId;
    const entry = {
      id: node.capability_id,
      data: node,
      depth,
      topId: resolvedTop,
      radius: Math.min(13, 4 + Math.sqrt(Number(node.query_count || 0)) * .85),
    };
    nodes.push(entry);
    if (parent) links.push({ source: parent.id, target: entry.id });
    for (const child of node.children || []) collect(child, entry, depth + 1, resolvedTop);
  }
  collect(root);
  const topCenters = new Map(top.map((node, index) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / Math.max(top.length, 1);
    return [node.capability_id, { x: 500 + Math.cos(angle) * 185, y: 310 + Math.sin(angle) * 185 }];
  }));
  nodes.forEach((node, index) => {
    const center = topCenters.get(node.topId) || { x: 500, y: 310 };
    const angle = index * 2.399963229728653;
    node.x = center.x + Math.cos(angle) * (12 + node.depth * 8);
    node.y = center.y + Math.sin(angle) * (12 + node.depth * 8);
  });
  const d3Viewport = d3.select(viewport);
  const linkSelection = d3Viewport.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "plot-link");
  const circles = d3Viewport.append("g").selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", (datum) => `plot-point${isLeaf(datum.data) ? " is-leaf" : ""}`)
    .attr("r", (datum) => datum.radius)
    .attr("fill", (datum) => datum.depth === 0 ? "var(--muted)" : plotColor(topHue.get(datum.topId) || 210, datum.depth, maxDepth))
    .attr("fill-opacity", (datum) => isLeaf(datum.data) ? .94 : .72)
    .attr("tabindex", 0)
    .attr("aria-label", (datum) => `${datum.data.name}，${Number(datum.data.query_count || 0)} 个用户 Query`);

  function showTooltip(event, datum) {
      const box = $("capability-plot").getBoundingClientRect();
      tooltip.innerHTML = `<strong>${escapeHtml(datum.data.name)}</strong>${escapeHtml(datum.data.definition || "暂无释义")}<br><span class="tooltip-path">${escapeHtml((datum.data.path || []).join(" › "))}<br>${Number(datum.data.query_count || 0)} 个用户 Query</span>`;
      tooltip.hidden = false;
      const clientX = event.clientX || box.left + box.width / 2;
      const clientY = event.clientY || box.top + box.height / 2;
      tooltip.style.left = `${Math.min(box.width - 312, Math.max(12, clientX - box.left + 12))}px`;
      tooltip.style.top = `${Math.max(12, clientY - box.top + 12)}px`;
  }

  circles
    .on("pointerenter pointermove", showTooltip)
    .on("pointerleave blur", () => { tooltip.hidden = true; })
    .on("focus", showTooltip)
    .on("click", (event, datum) => {
      event.stopPropagation();
      if (isLeaf(datum.data) && !event.defaultPrevented) {
        switchView("forest");
        selectLeaf(datum.data);
      }
    });

  const labels = d3Viewport.append("g").selectAll("text")
    .data(nodes.filter((datum) => datum.depth === 1))
    .join("text")
    .attr("class", "plot-label")
    .attr("dy", ".35em")
    .text((datum) => datum.data.name);

  plotSvgSelection = d3.select(svg);
  plotZoom = d3.zoom()
    .scaleExtent([.7, 8])
    .on("start", () => $("capability-plot").classList.add("is-dragging"))
    .on("zoom", (event) => d3Viewport.attr("transform", event.transform))
    .on("end", () => $("capability-plot").classList.remove("is-dragging"));
  plotSvgSelection.call(plotZoom).on("dblclick.zoom", null);

  const drag = d3.drag()
    .on("start", (event, datum) => {
      if (!event.active) plotSimulation.alphaTarget(.28).restart();
      datum.fx = datum.x;
      datum.fy = datum.y;
    })
    .on("drag", (event, datum) => {
      datum.fx = event.x;
      datum.fy = event.y;
    })
    .on("end", (event, datum) => {
      if (!event.active) plotSimulation.alphaTarget(0);
      datum.fx = null;
      datum.fy = null;
    });
  circles.call(drag);

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  plotSimulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((datum) => datum.id).distance((link) => 30 + Math.max(link.source.depth || 0, link.target.depth || 0) * 7).strength(.72))
    .force("charge", d3.forceManyBody().strength((datum) => datum.depth === 0 ? -280 : -42))
    .force("collision", d3.forceCollide().radius((datum) => datum.radius + 3).iterations(2))
    .force("x", d3.forceX((datum) => (topCenters.get(datum.topId) || { x: 500 }).x).strength((datum) => datum.depth <= 1 ? .12 : .035))
    .force("y", d3.forceY((datum) => (topCenters.get(datum.topId) || { y: 310 }).y).strength((datum) => datum.depth <= 1 ? .12 : .035))
    .force("center", d3.forceCenter(500, 310).strength(.025))
    .alphaDecay(reducedMotion ? .12 : .035)
    .velocityDecay(.42)
    .on("tick", () => {
      linkSelection
        .attr("x1", (link) => link.source.x).attr("y1", (link) => link.source.y)
        .attr("x2", (link) => link.target.x).attr("y2", (link) => link.target.y);
      circles.attr("cx", (datum) => datum.x).attr("cy", (datum) => datum.y);
      labels.attr("x", (datum) => datum.x).attr("y", (datum) => datum.y - datum.radius - 9);
    });

  $("plot-legend").innerHTML = top.map((node, index) => `<span><i style="background:${plotColor(hues[index % hues.length], 1, maxDepth)}"></i>${escapeHtml(node.name)}</span>`).join("");
  plotSvgSelection.call(plotZoom.transform, d3.zoomIdentity);
}

function applySearch() {
  const query = $("search-input").value.trim().toLocaleLowerCase();
  const matched = new Set();
  if (query) {
    for (const item of state.searchIndex) {
      const haystack = [item.name, ...(item.path || []), ...(item.queries || [])].join("\n").toLocaleLowerCase();
      if (haystack.includes(query)) matched.add(item.capability_id);
    }
  }
  let visibleLeaves = 0;
  function update(li) {
    const id = li.dataset.capabilityId;
    const node = state.nodes.get(id);
    const children = [...li.querySelectorAll(":scope > ul > li")];
    const childVisible = children.map(update).some(Boolean);
    const ownVisible = !query || (isLeaf(node) && matched.has(id));
    const visible = ownVisible || childVisible;
    li.classList.toggle("hidden", !visible);
    if (query && childVisible) {
      const list = li.querySelector(":scope > ul");
      const toggle = li.querySelector(":scope > .tree-row > .tree-toggle");
      if (list) list.hidden = false;
      li.classList.remove("is-collapsed");
      if (toggle && !toggle.disabled) toggle.setAttribute("aria-expanded", "true");
    }
    if (visible && isLeaf(node)) visibleLeaves += 1;
    return visible;
  }
  const rootLi = $("tree").querySelector(":scope > ul > li");
  if (rootLi) update(rootLi);
  $("search-status").textContent = query ? `找到 ${visibleLeaves} 个相关叶子能力` : "";
}

function collapseAll() {
  document.querySelectorAll(".tree li ul").forEach((list) => {
    list.hidden = true;
    list.parentElement.classList.add("is-collapsed");
  });
  const rootList = $("tree").querySelector(":scope > ul > li > ul");
  if (rootList) { rootList.hidden = false; rootList.parentElement.classList.remove("is-collapsed"); }
  document.querySelectorAll(".tree-toggle:not(:disabled)").forEach((button) => button.setAttribute("aria-expanded", "false"));
  const rootToggle = $("tree").querySelector(":scope > ul > li > .tree-row > .tree-toggle");
  if (rootToggle) rootToggle.setAttribute("aria-expanded", "true");
}

async function initialize() {
  try {
    const [metadata, taxonomy, searchIndex] = await Promise.all([
      loadJson("data/metadata.json"), loadJson("data/taxonomy.json"), loadJson("data/search-index.json")
    ]);
    document.title = metadata.site_title;
    $("site-title").textContent = metadata.site_title;
    $("version-line").textContent = `版本：${metadata.taxonomy_version} · 状态：${metadata.taxonomy_status || "未标记"} · 分析生成：${metadata.analysis_generated_at || "未知"}`;
    $("node-count").textContent = metadata.node_count;
    $("leaf-count").textContent = metadata.leaf_count;
    $("query-count").textContent = metadata.query_count;
    $("case-count").textContent = metadata.case_count;
    state.taxonomy = taxonomy;
    state.searchIndex = searchIndex;
    renderTree(taxonomy.root);
    renderCapabilityPlot(taxonomy.root);
  } catch (error) {
    $("tree").innerHTML = `<p class="error">网站数据读取失败：${escapeHtml(error.message)}。请通过 HTTP 服务打开，不要直接双击 index.html。</p>`;
    $("version-line").textContent = "数据不可用";
  }
}

$("search-input").addEventListener("input", applySearch);
$("collapse-button").addEventListener("click", collapseAll);
$("reset-plot-button").addEventListener("click", resetPlot);
$("refresh-button").addEventListener("click", () => window.location.reload());
document.querySelectorAll("[data-view-button]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.viewButton)));
[$("pain-search"), $("pain-batch-filter"), $("pain-status-filter"), $("pain-model-filter"), $("pain-harness-filter"), $("pain-attribution-filter"), $("pain-outcome-filter"), $("pain-gap-filter")]
  .forEach((control) => control.addEventListener(control.tagName === "INPUT" ? "input" : "change", applyPainFilters));
initialize();
