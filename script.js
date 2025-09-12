const projects = [
  {
    title: 'Project One',
    description: 'Activity Time Tracking Analysis - 8 weeks of comprehensive personal time tracking with minute-level precision',
    url: 'project-one.html',
    tags: ['Python', 'Data Analysis']
  },
  {
    title: 'Project Two',
    description: 'TBD',
    url: '#',
    tags: ['JavaScript']
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
