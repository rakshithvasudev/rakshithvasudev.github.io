(function () {
  var content = document.getElementById("post-content");
  if (!content) return;

  var headings = content.querySelectorAll("h2, h3");

  // Hover anchor links on every section heading.
  headings.forEach(function (h) {
    if (!h.id) return;
    var a = document.createElement("a");
    a.className = "heading-anchor";
    a.href = "#" + h.id;
    a.textContent = "#";
    a.setAttribute("aria-label", "Link to this section");
    h.appendChild(a);
  });

  // Build a table of contents when the post has enough structure to need one.
  var h2s = content.querySelectorAll("h2");
  if (h2s.length < 3) return;

  var toc = document.getElementById("toc");
  var list = document.getElementById("toc-list");
  var ul = document.createElement("ul");

  headings.forEach(function (h) {
    if (!h.id) return;
    var li = document.createElement("li");
    li.className = "toc-" + h.tagName.toLowerCase();
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/#$/, "");
    li.appendChild(a);
    ul.appendChild(li);
  });

  list.appendChild(ul);
  toc.hidden = false;
})();
