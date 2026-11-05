/**
 * Fuerza tooltips / títulos / validación nativa en español.
 */
(function () {
  const validationMessages = {
    valueMissing: "Completa este campo.",
    valueMissingFile: "Selecciona un archivo.",
    typeMismatch: "El formato no es válido.",
    patternMismatch: "El valor no tiene el formato esperado.",
    tooShort: "El texto es demasiado corto.",
    tooLong: "El texto es demasiado largo.",
    rangeUnderflow: "El valor es demasiado bajo.",
    rangeOverflow: "El valor es demasiado alto.",
    stepMismatch: "El valor no es válido.",
    badInput: "Valor no válido.",
  };

  function messageFor(el) {
    const v = el.validity;
    if (!v || v.valid) return "";
    if (v.valueMissing) {
      return el.type === "file"
        ? validationMessages.valueMissingFile
        : validationMessages.valueMissing;
    }
    if (v.typeMismatch) return validationMessages.typeMismatch;
    if (v.patternMismatch) return validationMessages.patternMismatch;
    if (v.tooShort) return validationMessages.tooShort;
    if (v.tooLong) return validationMessages.tooLong;
    if (v.rangeUnderflow) return validationMessages.rangeUnderflow;
    if (v.rangeOverflow) return validationMessages.rangeOverflow;
    if (v.stepMismatch) return validationMessages.stepMismatch;
    if (v.badInput) return validationMessages.badInput;
    return "Revisa este campo.";
  }

  function wireValidation(el) {
    el.addEventListener("invalid", function () {
      el.setCustomValidity(messageFor(el));
    });
    el.addEventListener("input", function () {
      el.setCustomValidity("");
    });
    el.addEventListener("change", function () {
      el.setCustomValidity("");
    });
  }

  document.querySelectorAll("input, select, textarea").forEach(wireValidation);

  document.querySelectorAll('input[type="file"]').forEach(function (el) {
    if (!el.getAttribute("title")) {
      el.setAttribute("title", "Selecciona un archivo");
    }
  });

  // Selectores de archivo “custom”: botón + etiqueta en español
  document.querySelectorAll("[data-es-file-input]").forEach(function (wrap) {
    const input = wrap.querySelector('input[type="file"]');
    const button = wrap.querySelector("[data-es-file-button]");
    const label = wrap.querySelector("[data-es-file-label]");
    if (!input || !button || !label) return;

    const emptyText =
      label.getAttribute("data-empty-text") || "Ningún archivo seleccionado";

    function syncLabel() {
      if (input.files && input.files.length > 0) {
        label.textContent =
          input.files.length === 1
            ? input.files[0].name
            : input.files.length + " archivos seleccionados";
        label.title = label.textContent;
      } else {
        label.textContent = emptyText;
        label.title = emptyText;
      }
    }

    button.addEventListener("click", function () {
      input.click();
    });
    input.addEventListener("change", syncLabel);
    syncLabel();
  });

  if (window.bootstrap && bootstrap.Tooltip) {
    document
      .querySelectorAll('[data-bs-toggle="tooltip"], [data-bs-title], [title]')
      .forEach(function (el) {
        if (el.closest(".navbar-brand, .nav-link, .dropdown-item")) return;
        if (el.tagName === "OPTION") return;
        // Solo tooltips Bootstrap explícitos; title nativo ya está en español.
        if (el.getAttribute("data-bs-toggle") === "tooltip") {
          bootstrap.Tooltip.getOrCreateInstance(el);
        }
      });
  }
})();
