/**
 * Nav wiring + lead-time selector for the dashboard-web frontend.
 * Only "Weight map" is functional as of step 3 -- the other 3 nav items
 * are real, clickable, and show an explicit "not yet built" state rather
 * than a broken/no-op click (frontendplan.md §2.7's interaction model:
 * switching views never reloads the page).
 */

const VIEWS = {
  "weight-map": {
    title: "Weight map",
    needsLead: true,
    render: (container, lead) => WeightMapView.render(container, lead),
  },
  "skill-trends": { title: "Skill trends", needsLead: false, render: null },
  "blended-map": { title: "Blended map", needsLead: true, render: null },
  "extreme-probability": {
    title: "Extreme-probability map",
    needsLead: true,
    render: null,
  },
};

let currentViewId = "weight-map";
let currentLead = null;

function renderNotBuilt(container, title) {
  container.innerHTML = "";
  const message = document.createElement("p");
  message.className = "not-built";
  message.textContent =
    `${title} is not built yet in this frontend -- it exists in the ` +
    "Streamlit app (dashboard/app.py) and will be added to this static " +
    "frontend in a later step of workspace/frontend-prompts/.";
  container.appendChild(message);
}

async function renderCurrentView() {
  const container = document.getElementById("app-main");
  const view = VIEWS[currentViewId];

  if (!view.render) {
    renderNotBuilt(container, view.title);
    return;
  }

  if (view.needsLead) {
    if (currentLead === null) {
      try {
        const { leads } = await fetchLeads();
        currentLead = leads[0];
      } catch (err) {
        container.innerHTML = `<p class="error-state">Failed to load lead times: ${err.message}</p>`;
        return;
      }
    }
    renderLeadSelector(container, currentLead);
  }

  const viewContainer = document.createElement("div");
  container.appendChild(viewContainer);
  await view.render(viewContainer, currentLead);
}

function renderLeadSelector(container, selectedLead) {
  container.innerHTML = "";
  const controls = document.createElement("div");
  controls.className = "view-controls";

  const label = document.createElement("label");
  label.setAttribute("for", "lead-select");
  label.textContent = "Lead time (hours)";
  controls.appendChild(label);

  const select = document.createElement("select");
  select.id = "lead-select";
  fetchLeads()
    .then(({ leads }) => {
      select.innerHTML = "";
      leads.forEach((lead) => {
        const option = document.createElement("option");
        option.value = lead;
        option.textContent = lead;
        if (lead === selectedLead) option.selected = true;
        select.appendChild(option);
      });
    })
    .catch(() => {
      const option = document.createElement("option");
      option.value = selectedLead;
      option.textContent = selectedLead;
      select.appendChild(option);
    });

  select.addEventListener("change", async () => {
    currentLead = Number(select.value);
    await renderCurrentView();
  });
  controls.appendChild(select);

  container.appendChild(controls);
}

function wireNav() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", async () => {
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("is-active"));
      button.classList.add("is-active");
      currentViewId = button.dataset.view;
      await renderCurrentView();
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  wireNav();
  renderCurrentView();
});
