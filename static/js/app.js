// Small UX improvements
document.addEventListener("DOMContentLoaded", () => {
  // Auto-scroll to form if error exists
  const alert = document.querySelector(".alert-danger");
  const formCard = document.getElementById("formCard");
  if (alert && formCard) {
    formCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
