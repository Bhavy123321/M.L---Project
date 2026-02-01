document.addEventListener("DOMContentLoaded", () => {
  // Smooth scroll to error if present
  const alert = document.querySelector(".alert-danger");
  if (alert) alert.scrollIntoView({ behavior: "smooth", block: "start" });
});
