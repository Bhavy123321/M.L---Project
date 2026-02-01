document.addEventListener("DOMContentLoaded", () => {
  const alert = document.querySelector(".alert-danger");
  const formCard = document.getElementById("formCard");
  if (alert && formCard) {
    formCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
