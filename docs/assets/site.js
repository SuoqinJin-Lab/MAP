const projectTree = [
  { kind: "root", label: "MAP/" },
  { kind: "file", label: "requirements-map.txt", depth: 1, href: "#install" },
  { kind: "file", label: "requirements-map-baselines.txt", depth: 1, href: "#install" },
  { kind: "file", label: "toy.sh", depth: 1, href: "#toy-workflow" },
  { kind: "dir", label: "map/preprocess/", depth: 1, href: "#bind-handler", entity: "handler" },
  { kind: "dir", label: "handlers/", depth: 2, href: "#bind-handler", entity: "handler" },
  { kind: "file", label: "builtin.py", depth: 2, href: "#bind-handler", entity: "handler" },
  { kind: "dir", label: "storage/", depth: 1, href: "#environment" },
  { kind: "dir", label: "raw_datasets/", depth: 2, href: "#data-statistics", entity: "raw" },
  { kind: "file", label: "Tahoe-100M/", depth: 3, href: "#data-statistics", entity: "raw" },
  { kind: "dir", label: "frozen_models/", depth: 2, href: "#assets" },
  { kind: "file", label: "state/SE-600M", depth: 3, href: "#state-cache", entity: "se600m" },
  { kind: "file", label: "state/ESM2 genes", depth: 3, href: "#knowledge-cache", entity: "geneCache" },
  { kind: "file", label: "mapkg/checkpoint + vocab", depth: 3, href: "#knowledge-cache", entity: "mapkg" },
  { kind: "dir", label: "projects/", depth: 2, href: "#preprocess", entity: "workspace" },
  { kind: "dir", label: "<project_name>/", depth: 3, href: "#fetch-populations", entity: "workspace" },
  { kind: "file", label: "preprocess.json", depth: 4, href: "#fetch-populations" },
  { kind: "file", label: "contract.json", depth: 4, href: "#create-project" },
  { kind: "file", label: "condition_filter.json", depth: 4, href: "#filter-conditions" },
  { kind: "dir", label: "materialize/", depth: 4, href: "#materialize-cell-metadata" },
  { kind: "file", label: "cell metadata mmap", depth: 5, href: "#materialize-cell-metadata", entity: "cellMetadata" },
  { kind: "file", label: "STATE input mmap", depth: 5, href: "#materialize-state-inputs", entity: "stateInputs" },
  { kind: "file", label: "HVG expression mmap", depth: 5, href: "#materialize-hvg-expression", entity: "hvgExpression" },
  { kind: "dir", label: "splits/", depth: 4, href: "#sampling-split", entity: "splits" },
  { kind: "file", label: "<split_id>/split.json", depth: 5, href: "#sampling-split", entity: "splits" },
  { kind: "dir", label: "<split_id>/materialize/drug_moa/", depth: 5, href: "#baseline-preparation", entity: "methodMaterial" },
  { kind: "dir", label: "<split_id>/methods/<method>/", depth: 5, href: "#validate", entity: "evaluations" },
  { kind: "file", label: "MAP token caches", depth: 5, href: "#knowledge-cache", entity: "geneCache" },
  { kind: "file", label: "STATE embeddings", depth: 5, href: "#state-cache", entity: "cellCache" },
  { kind: "dir", label: "materialize/<artifact>/", depth: 5, href: "#baseline-preparation", entity: "methodMaterial" },
  { kind: "dir", label: "<split_id>/methods/<method>/runs/<run_name>/", depth: 5, href: "#train-run", entity: "runs" },
  { kind: "file", label: "last.pt + checkpoints/ + run_config.json", depth: 5, href: "#train-run", entity: "runs" },
  { kind: "dir", label: "<split_id>/methods/<method>/runs/<run_name>/evaluations/<evaluation_name>/", depth: 5, href: "#evaluate-run", entity: "evaluations" },
  { kind: "file", label: "metrics + predictions + report", depth: 5, href: "#analysis", entity: "evaluations" },
  { kind: "dir", label: "toy_runs/", depth: 2, href: "#toy-workflow" },
];

const diagramTargets = {
  raw: "#data-statistics",
  handler: "#bind-handler",
  condition: "#fetch-populations",
  control: "#fetch-populations",
  conditionCell: "#fetch-populations",
  cellMetadata: "#materialize-cell-metadata",
  stateInputs: "#materialize-state-inputs",
  hvgExpression: "#materialize-hvg-expression",
  hvgTarget: "#materialize-hvg-expression",
  splits: "#sampling-split",
  mapkg: "#knowledge-cache",
  se600m: "#state-cache",
  drugCache: "#knowledge-cache",
  geneCache: "#knowledge-cache",
  controlCache: "#state-cache",
  cellCache: "#state-cache",
  methodMaterial: "#baseline-preparation",
  runs: "#train-run",
  evaluations: "#evaluate-run",
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function renderSidebar() {
  const aside = document.querySelector("aside");
  if (!aside) return;
  const tree = projectTree.map(item => {
    if (item.kind === "root") return `<div class="tree-root">${item.label}</div>`;
    const icon = item.kind === "dir" ? "⌄" : "·";
    const entity = item.entity ? ` data-flash="${item.entity}"` : "";
    return `<a href="${item.href}"${entity} class="tree-item tree-${item.kind}" style="--depth:${item.depth}"><i>${icon}</i><span>${escapeHtml(item.label)}</span></a>`;
  }).join("");
  aside.innerHTML = `<a class="side-brand" href="#top"><span>MAP</span><strong>Experiment Guide</strong></a><div class="side-caption">STORAGE & OUTPUTS</div><nav class="project-tree">${tree}</nav>`;
}

function flashEntity(name) {
  const entity = document.querySelector(`[data-entity="${name}"]`);
  if (!entity) return;
  entity.classList.remove("entity-flash");
  window.requestAnimationFrame(() => entity.classList.add("entity-flash"));
}

function enableNavigation() {
  document.querySelectorAll("[data-flash]").forEach(link => {
    link.addEventListener("click", () => flashEntity(link.dataset.flash));
  });
  document.querySelectorAll(".diagram-node[data-entity]").forEach(node => {
    const target = diagramTargets[node.dataset.entity];
    if (!target) return;
    node.classList.add("diagram-link");
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "link");
    const open = () => { window.location.hash = target; };
    node.addEventListener("click", open);
    node.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") open();
    });
  });
}

function enableCopy() {
  document.querySelectorAll(".copy").forEach(button => button.addEventListener("click", async () => {
    const code = button.closest(".code-block").querySelector("code").innerText;
    try { await navigator.clipboard.writeText(code); } catch (_) {}
    button.textContent = "已复制";
    window.setTimeout(() => { button.textContent = "复制"; }, 900);
  }));
}

document.addEventListener("DOMContentLoaded", () => {
  renderSidebar();
  enableNavigation();
  enableCopy();
});
