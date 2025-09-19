document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => {
      if (cb.checked) {
        fetch("/save_progress", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user: cb.dataset.user,
            topic: cb.dataset.topic,
            step: cb.value
          })
        });
      }
    });
  });
});
