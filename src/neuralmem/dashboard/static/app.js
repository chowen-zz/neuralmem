/* NeuralMem Dashboard — Frontend Application */

(function () {
  "use strict";

  // ---- State ----
  let currentPage = "overview";
  let memoriesOffset = 0;
  const memoriesLimit = 25;
  let graphSimulation = null;

  // ---- Navigation ----
  function initNavigation() {
    var items = document.querySelectorAll(".nav-item");
    items.forEach(function (item) {
      item.addEventListener("click", function () {
        var page = item.getAttribute("data-page");
        if (page === currentPage) return;
        switchPage(page);
      });
    });
  }

  function switchPage(page) {
    document.querySelectorAll(".nav-item").forEach(function (el) {
      el.classList.toggle("active", el.getAttribute("data-page") === page);
    });
    document.querySelectorAll(".page").forEach(function (el) {
      el.classList.toggle("active", el.id === "page-" + page);
    });
    currentPage = page;

    if (page === "overview") loadOverview();
    if (page === "memories") loadMemories();
    if (page === "graph") loadGraph();
    if (page === "settings") loadSettings();
  }

  // ---- API Helpers ----
  function api(url, options) {
    return fetch(url, options).then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    });
  }

  // ---- Overview ----
  function loadOverview() {
    loadHealth();
    loadMemoryStats();
    loadGraphStats();
    loadMetrics();
  }

  function loadHealth() {
    api("/api/health")
      .then(function (data) {
        var statusEl = document.getElementById("health-status");
        var detailsEl = document.getElementById("health-details");
        var dot = document.getElementById("status-dot");
        var statusText = document.getElementById("status-text");

        statusEl.textContent = data.status.toUpperCase();
        dot.className = "status-dot " + data.status;
        statusText.textContent = data.status.toUpperCase();

        var lines = [];
        Object.keys(data.checks || {}).forEach(function (key) {
          lines.push(key + ": " + data.checks[key]);
        });
        detailsEl.innerHTML = lines.join("<br>");
      })
      .catch(function () {
        document.getElementById("health-status").textContent = "ERROR";
      });
  }

  function loadMemoryStats() {
    api("/api/memories?limit=1&offset=0")
      .then(function (data) {
        document.getElementById("memory-count").textContent = data.total;
        document.getElementById("memory-details").textContent =
          "Showing page of " + data.total + " total memories";
      })
      .catch(function () {
        document.getElementById("memory-count").textContent = "ERR";
      });
  }

  function loadGraphStats() {
    api("/api/graph/stats")
      .then(function (data) {
        document.getElementById("graph-nodes").textContent =
          (data.node_count || 0) + " nodes";
        document.getElementById("graph-edges").textContent =
          (data.edge_count || 0) + " edges";
      })
      .catch(function () {
        document.getElementById("graph-nodes").textContent = "ERR";
      });
  }

  function loadMetrics() {
    api("/api/metrics")
      .then(function (data) {
        var counters = data.counters || {};
        var keys = Object.keys(counters);
        document.getElementById("metrics-counters").textContent =
          keys.length + " counters";

        var lines = [];
        keys.forEach(function (k) {
          lines.push(k + ": " + counters[k]);
        });
        var histograms = data.histograms || {};
        Object.keys(histograms).forEach(function (k) {
          var h = histograms[k];
          lines.push(k + ": n=" + h.count + ", mean=" + (h.mean * 1000).toFixed(1) + "ms");
        });
        document.getElementById("metrics-details").innerHTML =
          lines.slice(0, 8).join("<br>");
      })
      .catch(function () {
        document.getElementById("metrics-counters").textContent = "ERR";
      });
  }

  // ---- Memories ----
  function loadMemories() {
    var search = document.getElementById("memory-search").value;
    var typeFilter = document.getElementById("memory-type-filter").value;

    var url = "/api/memories?limit=" + memoriesLimit + "&offset=" + memoriesOffset;
    api(url)
      .then(function (data) {
        var memories = data.memories || [];

        // Client-side filtering
        if (search) {
          var q = search.toLowerCase();
          memories = memories.filter(function (m) {
            return m.content.toLowerCase().indexOf(q) !== -1;
          });
        }
        if (typeFilter) {
          memories = memories.filter(function (m) {
            return m.memory_type === typeFilter;
          });
        }

        renderMemoriesTable(memories);
        renderPagination(data.total);
      })
      .catch(function () {
        document.getElementById("memories-tbody").innerHTML =
          '<tr><td colspan="7">Failed to load memories</td></tr>';
      });
  }

  function renderMemoriesTable(memories) {
    var tbody = document.getElementById("memories-tbody");
    if (!memories.length) {
      tbody.innerHTML =
        '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">No memories found</td></tr>';
      return;
    }

    tbody.innerHTML = memories
      .map(function (m) {
        var tags = (m.tags || [])
          .map(function (t) {
            return '<span class="tag">' + escapeHtml(t) + "</span>";
          })
          .join("");
        var created = m.created_at ? new Date(m.created_at).toLocaleDateString() : "--";
        return (
          "<tr>" +
          '<td title="' + escapeHtml(m.id) + '">' + escapeHtml(m.id.slice(0, 12)) + "...</td>" +
          '<td class="content-cell" title="' + escapeHtml(m.content) + '">' +
          escapeHtml(m.content.slice(0, 100)) +
          "</td>" +
          "<td>" + escapeHtml(m.memory_type) + "</td>" +
          "<td>" + m.importance.toFixed(2) + "</td>" +
          "<td>" + (tags || "--") + "</td>" +
          "<td>" + created + "</td>" +
          "<td>" + m.access_count + "</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  function renderPagination(total) {
    var container = document.getElementById("memories-pagination");
    var pages = Math.ceil(total / memoriesLimit);
    var current = Math.floor(memoriesOffset / memoriesLimit);

    if (pages <= 1) {
      container.innerHTML = "";
      return;
    }

    var html = "";
    for (var i = 0; i < pages && i < 20; i++) {
      html +=
        '<button class="page-btn' +
        (i === current ? " active" : "") +
        '" data-page="' +
        i +
        '">' +
        (i + 1) +
        "</button>";
    }
    container.innerHTML = html;

    container.querySelectorAll(".page-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        memoriesOffset = parseInt(btn.getAttribute("data-page")) * memoriesLimit;
        loadMemories();
      });
    });
  }

  function initMemorySearch() {
    document.getElementById("btn-search-memory").addEventListener("click", function () {
      memoriesOffset = 0;
      loadMemories();
    });
    document.getElementById("memory-search").addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        memoriesOffset = 0;
        loadMemories();
      }
    });
  }

  // ---- Knowledge Graph ----
  function loadGraph() {
    // We need nodes and edges — use /api/memories to build a simple graph
    // In a real implementation this would have a dedicated /api/graph/data endpoint
    var svg = d3.select("#graph-svg");
    svg.selectAll("*").remove();

    var container = document.getElementById("graph-container");
    var width = container.clientWidth;
    var height = container.clientHeight;

    svg.attr("viewBox", "0 0 " + width + " " + height);

    // Fetch memories and graph stats
    Promise.all([
      api("/api/memories?limit=100&offset=0"),
      api("/api/graph/stats"),
    ])
      .then(function (results) {
        var memories = results[0].memories || [];
        var stats = results[1];

        document.getElementById("graph-info").textContent =
          stats.node_count + " nodes, " + stats.edge_count + " edges";

        // Build a simple graph from memories
        var nodes = [];
        var links = [];
        var nodeMap = {};

        memories.forEach(function (m, i) {
          var id = "mem_" + i;
          nodeMap[id] = true;
          nodes.push({
            id: id,
            label: m.content.slice(0, 30),
            type: m.memory_type,
            radius: 6 + m.importance * 10,
          });

          // Link to related nodes by shared tags
          memories.forEach(function (m2, j) {
            if (j <= i) return;
            var shared = (m.tags || []).filter(function (t) {
              return (m2.tags || []).indexOf(t) !== -1;
            });
            if (shared.length > 0) {
              links.push({
                source: id,
                target: "mem_" + j,
                label: shared[0],
              });
            }
          });
        });

        renderGraph(svg, nodes, links, width, height);
      })
      .catch(function () {
        document.getElementById("graph-info").textContent = "Failed to load graph";
      });
  }

  function renderGraph(svg, nodes, links, width, height) {
    var colorMap = {
      fact: "#58a6ff",
      preference: "#bc8cff",
      episodic: "#3fb950",
      semantic: "#d29922",
      procedural: "#f85149",
      working: "#8b949e",
    };

    if (graphSimulation) graphSimulation.stop();

    graphSimulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id(function (d) { return d.id; })
          .distance(80)
      )
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(20));

    var link = svg
      .append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "graph-link")
      .attr("stroke-width", 1);

    var linkLabel = svg
      .append("g")
      .selectAll("text")
      .data(links)
      .join("text")
      .attr("class", "graph-link-label")
      .text(function (d) { return d.label || ""; });

    var node = svg
      .append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("class", "graph-node")
      .call(
        d3
          .drag()
          .on("start", dragStarted)
          .on("drag", dragged)
          .on("end", dragEnded)
      );

    node
      .append("circle")
      .attr("r", function (d) { return d.radius; })
      .attr("fill", function (d) { return colorMap[d.type] || "#8b949e"; });

    node
      .append("text")
      .attr("dx", 12)
      .attr("dy", 4)
      .text(function (d) { return d.label; });

    graphSimulation.on("tick", function () {
      link
        .attr("x1", function (d) { return d.source.x; })
        .attr("y1", function (d) { return d.source.y; })
        .attr("x2", function (d) { return d.target.x; })
        .attr("y2", function (d) { return d.target.y; });

      linkLabel
        .attr("x", function (d) { return (d.source.x + d.target.x) / 2; })
        .attr("y", function (d) { return (d.source.y + d.target.y) / 2; });

      node.attr("transform", function (d) {
        return "translate(" + d.x + "," + d.y + ")";
      });
    });

    function dragStarted(event, d) {
      if (!event.active) graphSimulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragEnded(event, d) {
      if (!event.active) graphSimulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }

  function initGraph() {
    document.getElementById("btn-refresh-graph").addEventListener("click", function () {
      loadGraph();
    });
  }

  // ---- Settings ----
  function loadSettings() {
    api("/api/config")
      .then(function (data) {
        var grid = document.getElementById("settings-grid");
        var html = "";
        Object.keys(data).forEach(function (key) {
          html +=
            '<div class="settings-item">' +
            '<div class="key">' + escapeHtml(key) + "</div>" +
            '<div class="value">' + escapeHtml(String(data[key])) + "</div>" +
            "</div>";
        });
        grid.innerHTML = html;
      })
      .catch(function () {
        document.getElementById("settings-grid").innerHTML =
          "<p>Failed to load configuration</p>";
      });
  }

  // ---- Utilities ----
  function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  // ---- Init ----
  function init() {
    initNavigation();
    initMemorySearch();
    initGraph();
    loadOverview();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
