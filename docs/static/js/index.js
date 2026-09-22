document.addEventListener("DOMContentLoaded", () => {
  const burger = document.querySelector(".navbar-burger");
  const menu = document.getElementById("project-navbar");

  if (burger && menu) {
    burger.addEventListener("click", () => {
      const expanded = burger.getAttribute("aria-expanded") === "true";
      burger.setAttribute("aria-expanded", String(!expanded));
      burger.classList.toggle("is-active");
      menu.classList.toggle("is-active");
    });

    menu.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        burger.setAttribute("aria-expanded", "false");
        burger.classList.remove("is-active");
        menu.classList.remove("is-active");
      });
    });
  }

  const progressBar = document.getElementById("scroll-progress-bar");
  const updateProgress = () => {
    if (!progressBar) return;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const progress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
    progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  };

  updateProgress();
  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);
});
