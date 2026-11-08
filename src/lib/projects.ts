// The visitor's own projects: every video they ask Vidgin to make is kept
// here, in their browser only (localStorage). Shared by the generator card
// (which adds a project) and the top-bar menu (which lists them).

export interface Project {
  prompt: string;
  duration: number;
  createdAt: number;
}

const KEY = 'vidgen-projects';
const EVENT = 'vidgen-projects-changed';

export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Project[]) : [];
  } catch {
    return [];
  }
}

export function addProject(prompt: string, duration: number): void {
  const list = loadProjects();
  list.unshift({ prompt, duration, createdAt: Date.now() });
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, 50)));
  } catch {
    // Storage full or blocked — the list just won't persist.
  }
  window.dispatchEvent(new Event(EVENT));
}

export function subscribeProjects(onChange: () => void): () => void {
  const handler = () => onChange();
  window.addEventListener(EVENT, handler);
  window.addEventListener('storage', handler);
  return () => {
    window.removeEventListener(EVENT, handler);
    window.removeEventListener('storage', handler);
  };
}

// "Use this project again": the menu asks, the generator card listens and
// loads the project's prompt + duration back into the box.
const REUSE_EVENT = 'vidgen-project-reuse';

export function requestReuseProject(project: Project): void {
  window.dispatchEvent(new CustomEvent<Project>(REUSE_EVENT, { detail: project }));
}

export function subscribeProjectReuse(onReuse: (project: Project) => void): () => void {
  const handler = (e: Event) => onReuse((e as CustomEvent<Project>).detail);
  window.addEventListener(REUSE_EVENT, handler);
  return () => window.removeEventListener(REUSE_EVENT, handler);
}
