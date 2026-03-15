/**
 * mobile-level.js
 * Sets <html> class to level-entry | level-basic | level-flagship from viewport width.
 * On mobile device: level from actual viewport width.
 * On desktop browser: level from available width (capped at 428px) so resize updates level.
 */
(function () {
  var MOBILE_WIDTH_THRESHOLD = 768;
  var BROWSER_CAP = 428;

  function getWidth() {
    if (typeof window.visualViewport !== "undefined" && window.visualViewport.width > 0) {
      return window.visualViewport.width;
    }
    return window.innerWidth || document.documentElement.clientWidth || 0;
  }

  function isMobileEnv(width) {
    return width <= MOBILE_WIDTH_THRESHOLD ||
      (navigator.maxTouchPoints > 0 && width <= MOBILE_WIDTH_THRESHOLD);
  }

  function levelFromWidth(width) {
    if (width <= 375) return "level-entry";
    if (width <= 412) return "level-basic";
    return "level-flagship";
  }

  function updateLevel() {
    var width = getWidth();
    var mobile = isMobileEnv(width);
    var effectiveWidth = mobile ? width : Math.min(width, BROWSER_CAP);
    var level = levelFromWidth(effectiveWidth);

    var root = document.documentElement;
    root.classList.remove("level-entry", "level-basic", "level-flagship");
    root.classList.add(level);
    root.classList.remove("env-mobile", "env-browser");
    root.classList.add(mobile ? "env-mobile" : "env-browser");
  }

  updateLevel();

  window.addEventListener("resize", updateLevel);
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", updateLevel);
  }
})();
