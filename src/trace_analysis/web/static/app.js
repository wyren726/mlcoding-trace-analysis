const state = { taxonomy: null, searchIndex: [], selected: null, nodes: new Map(), plot: { x: 0, y: 0, scale: 1 } };

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
}[char]));

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
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

function switchView(view) {
  document.querySelectorAll("[data-view]").forEach((section) => { section.hidden = section.dataset.view !== view; });
  document.querySelectorAll("[data-view-button]").forEach((button) => {
    if (button.dataset.viewButton === view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
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
initialize();
