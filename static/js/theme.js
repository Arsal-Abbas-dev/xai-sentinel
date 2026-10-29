/* Theme toggle + mobile nav. Keeps the existing localStorage "theme" key and html.dark class. */
(function () {
  function syncIcon() {
    var icon = document.getElementById('theme-icon');
    if (icon) icon.textContent = document.documentElement.classList.contains('dark') ? '🌙' : '☀️';
  }
  if (typeof window.toggleTheme !== 'function') {
    window.toggleTheme = function () {
      var html = document.documentElement;
      var dark = html.classList.toggle('dark');
      try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch (e) {}
      syncIcon();
      document.dispatchEvent(new CustomEvent('themechange'));
    };
  }
  window.toggleNav = function (open) {
    var side = document.getElementById('app-sidebar');
    var back = document.getElementById('nav-backdrop');
    var btn = document.getElementById('nav-open');
    if (!side) return;
    side.classList.toggle('open', open);
    if (back) back.classList.toggle('show', open);
    if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  };
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') window.toggleNav(false); });
  document.addEventListener('DOMContentLoaded', syncIcon);
})();
