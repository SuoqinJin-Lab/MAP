const projectTree = [
  { kind: "root", label: "MAP/" },
  { kind: "file", label: "requirements-map.txt", depth: 1, href: "#install" },
  { kind: "file", label: "requirements-map-baselines.txt", depth: 1, href: "#install" },
  { kind: "file", label: "toy.sh", depth: 1, href: "#toy-workflow" },
  { kind: "dir", label: "storage/", depth: 1, href: "#environment" },
  { kind: "dir", label: "raw_datasets/", depth: 2, href: "#data-statistics", entity: "raw" },
  { kind: "file", label: "Tahoe-100M/", depth: 3, href: "#data-statistics", entity: "raw" },
  { kind: "file", label: "data_summary.json", depth: 4, href: "#data-statistics", entity: "raw" },
  { kind: "dir", label: "frozen_models/", depth: 2, href: "#assets", entity: "frozen" },
  { kind: "file", label: "state/SE-600M", depth: 3, href: "#state-cache", entity: "frozen" },
  { kind: "file", label: "state/ESM2 genes", depth: 3, href: "#knowledge-gene", entity: "esm2" },
  { kind: "file", label: "mapkg/checkpoint + vocab", depth: 3, href: "#knowledge-drug", entity: "mapkg" },
  { kind: "dir", label: "projects/", depth: 2, href: "#preprocess", entity: "workspace" },
  { kind: "dir", label: "<project_name>/", depth: 3, href: "#fetch-cell-line", entity: "workspace" },
  { kind: "file", label: "preprocess.json", depth: 4, href: "#fetch-cell-line" },
  { kind: "file", label: "selection.json", depth: 4, href: "#fetch-cell-line" },
  { kind: "file", label: "contract.json", depth: 4, href: "#materialize" },
  { kind: "file", label: "workflow.json", depth: 4, href: "#create-workflow" },
  { kind: "dir", label: "materialized/", depth: 4, href: "#materialize" },
  { kind: "file", label: "HVG + cell mmap", depth: 5, href: "#materialize", entity: "hvg" },
  { kind: "file", label: "splits/", depth: 5, href: "#split", entity: "splits" },
  { kind: "file", label: "MAP token caches", depth: 5, href: "#knowledge-gene", entity: "geneCache" },
  { kind: "file", label: "STATE embeddings", depth: 5, href: "#state-cache", entity: "cellCache" },
  { kind: "dir", label: "baselines/", depth: 5, href: "#baseline-preparation" },
  { kind: "dir", label: "runs/<run_name>/", depth: 4, href: "#train-run", entity: "runs" },
  { kind: "file", label: "best.pt + run_config.json", depth: 5, href: "#train-run", entity: "runs" },
  { kind: "dir", label: "evaluations/<run_name>/", depth: 4, href: "#evaluate-run", entity: "evaluations" },
  { kind: "file", label: "metrics + predictions + report", depth: 5, href: "#analysis", entity: "evaluations" },
  { kind: "dir", label: "toy_runs/", depth: 2, href: "#toy-workflow" },
];

const diagramTargets = {
  raw: "#data-statistics",
  condition: "#fetch-cell-line",
  splits: "#split",
  control: "#materialize",
  conditionCell: "#materialize",
  esm2: "#knowledge-gene",
  mapkg: "#knowledge-drug",
  hvg: "#hvg-selection",
  frozen: "#state-cache",
  drugCache: "#knowledge-drug",
  geneCache: "#knowledge-gene",
  cellCache: "#state-cache",
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
