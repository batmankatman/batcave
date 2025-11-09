const projects = [
  {
    title: 'Project One',
    description: 'Activity Time Tracking Analysis - 8 weeks of comprehensive personal time tracking with minute-level precision',
    url: 'project-one.html',
    tags: ['Python', 'Data Analysis']
  },
  {
    title: 'Project Two',
    description: 'Previous Time Tracking Dashboard - Earlier version of the time tracking application',
    url: 'project-two.html',
    tags: ['Python', 'Streamlit']
  },
  {
    title: 'Project Three',
    description: 'Personal Time Tracker Dashboard v2 - Enhanced interactive Streamlit app with dynamic visualizations and insights',
    url: 'https://personal-time-tracker-v2-xs5yszmumrwumhxpktuv8r.streamlit.app',
    tags: ['Python', 'Streamlit', 'Dashboard', 'Interactive']
  }
];

function renderProjects() {
  const grid = document.getElementById('projects-grid');
  if (!grid) return; // safe when projects section isn't present
  grid.innerHTML = projects.map(p => `
    <a class="project-card" href="${p.url}" target="_blank" rel="noopener">
      <h4>${p.title}</h4>
      <p>${p.description}</p>
      <div class="tags">${p.tags.map(t=>`<span class="tag">${t}</span>`).join('')}</div>
    </a>
  `).join('');
}

function setYear() {
  const el = document.getElementById('year');
  if (el) el.textContent = new Date().getFullYear();
}

window.addEventListener('DOMContentLoaded', () => {
  renderProjects();
  setYear();
});
