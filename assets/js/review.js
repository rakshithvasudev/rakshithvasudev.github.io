/* Private review mode: ?review=on to enable, ?review=off to disable.
   Notes live in localStorage only; nothing is ever sent anywhere. */
(function () {
  "use strict";
  var qs = new URLSearchParams(window.location.search);
  if (qs.get("review") === "on") localStorage.setItem("rv-review", "1");
  if (qs.get("review") === "off") localStorage.removeItem("rv-review");
  if (localStorage.getItem("rv-review") !== "1") return;

  var content = document.querySelector(".post-content");
  if (!content) return;

  var KEY = "rv-notes:" + window.location.pathname;
  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); }
    catch (e) { return []; }
  }
  function save(notes) { localStorage.setItem(KEY, JSON.stringify(notes)); }

  /* badge */
  var badge = document.createElement("div");
  badge.className = "rv-badge";
  document.body.appendChild(badge);
  function render() {
    var n = load().length;
    badge.innerHTML =
      "<strong>review</strong> " + n + " note" + (n === 1 ? "" : "s") +
      ' <button data-rv="list">list</button>' +
      ' <button data-rv="copy">copy</button>' +
      ' <button data-rv="download">download</button>' +
      ' <button data-rv="clear">clear</button>';
  }
  render();

  function exportPayload() {
    return JSON.stringify({ page: window.location.pathname, exported: new Date().toISOString(), notes: load() }, null, 2);
  }

  badge.addEventListener("click", function (e) {
    var act = e.target.getAttribute && e.target.getAttribute("data-rv");
    if (!act) return;
    if (act === "copy") {
      navigator.clipboard.writeText(exportPayload()).then(function () {
        e.target.textContent = "copied"; setTimeout(function () { e.target.textContent = "copy"; }, 1200);
      });
    } else if (act === "download") {
      var blob = new Blob([exportPayload()], { type: "application/json" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "review" + window.location.pathname.replace(/\//g, "-").replace(/\.html$/, "") + ".json";
      a.click(); URL.revokeObjectURL(a.href);
    } else if (act === "clear") {
      if (confirm("Delete all notes on this page?")) { save([]); render(); }
    } else if (act === "list") {
      showList();
    }
  });

  function showList() {
    var old = document.querySelector(".rv-list");
    if (old) { old.remove(); return; }
    var panel = document.createElement("div");
    panel.className = "rv-list";
    var notes = load();
    if (!notes.length) panel.innerHTML = "<p>No notes yet. Select text to add one.</p>";
    notes.forEach(function (n, i) {
      var item = document.createElement("div");
      item.className = "rv-item";
      var q = document.createElement("blockquote"); q.textContent = n.quote.slice(0, 140);
      var c = document.createElement("p"); c.textContent = n.comment;
      var del = document.createElement("button"); del.textContent = "delete";
      del.addEventListener("click", function () {
        var all = load(); all.splice(i, 1); save(all); render(); panel.remove(); showList();
      });
      item.appendChild(q); item.appendChild(c); item.appendChild(del);
      panel.appendChild(item);
    });
    document.body.appendChild(panel);
  }

  /* selection -> note chip */
  var chip = document.createElement("button");
  chip.className = "rv-chip";
  chip.textContent = "+ note";
  chip.style.display = "none";
  document.body.appendChild(chip);
  var pending = null;

  document.addEventListener("mouseup", function () {
    setTimeout(function () {
      var sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.rangeCount) { chip.style.display = "none"; return; }
      var range = sel.getRangeAt(0);
      if (!content.contains(range.commonAncestorContainer)) { chip.style.display = "none"; return; }
      var text = sel.toString().trim();
      if (text.length < 3) { chip.style.display = "none"; return; }
      var full = content.innerText;
      var idx = full.indexOf(text);
      pending = {
        quote: text,
        prefix: idx > 0 ? full.slice(Math.max(0, idx - 60), idx) : "",
        suffix: idx >= 0 ? full.slice(idx + text.length, idx + text.length + 60) : ""
      };
      var r = range.getBoundingClientRect();
      chip.style.top = (window.scrollY + r.bottom + 6) + "px";
      chip.style.left = (window.scrollX + Math.min(r.right, window.innerWidth - 90)) + "px";
      chip.style.display = "block";
    }, 10);
  });

  chip.addEventListener("mousedown", function (e) {
    e.preventDefault(); e.stopPropagation();
    if (!pending) return;
    var comment = prompt('Note on: "' + pending.quote.slice(0, 80) + '"');
    chip.style.display = "none";
    if (!comment) return;
    var notes = load();
    notes.push({ quote: pending.quote, prefix: pending.prefix, suffix: pending.suffix, comment: comment, ts: new Date().toISOString() });
    save(notes); render();
    pending = null;
  });
})();
