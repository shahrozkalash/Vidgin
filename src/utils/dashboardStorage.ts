/**
 * Persistence layer for Vidgen Dashboard and Projects
 * Safely persists and loads projects, clips, jobs, and usage to localStorage with JSON export/import.
 */

import { DashboardStorageState, Project, Clip, UsageRecord } from '../types/dashboard';
import { processMonthlyUsage } from './dashboardAggregation';

const STORAGE_KEY = 'vidgen_dashboard_data_v1';

export const DEFAULT_INITIAL_PROJECTS: Project[] = [
  {
    id: 'proj-nordic-wild',
    name: 'Nordic Wilderness Series',
    targetSeconds: 30,
    createdAt: Date.now() - 1000 * 60 * 60 * 24 * 3, // 3 days ago
    updatedAt: Date.now() - 1000 * 60 * 25,
    coverClipId: 'clip-nordic-1',
    clips: [
      {
        id: 'clip-nordic-1',
        projectId: 'proj-nordic-wild',
        prompt: 'A cinematic red fox emerging through misty pine forest at sunrise, soft volumetric sunlight',
        seed: 42109,
        durationSec: 5.0,
        status: 'done',
        progress: 1.0,
        createdAt: Date.now() - 1000 * 60 * 60 * 24 * 2,
        continuity: { seam: 98.2, identity: 96.5, colorDrift: 1.4 },
        favorite: true,
      },
      {
        id: 'clip-nordic-2',
        projectId: 'proj-nordic-wild',
        parentId: 'clip-nordic-1',
        prompt: 'Red fox turns head toward distant river, soft morning light filtering through birch trees',
        seed: 42109,
        durationSec: 5.0,
        status: 'done',
        progress: 1.0,
        createdAt: Date.now() - 1000 * 60 * 60 * 20,
        continuity: { seam: 96.4, identity: 94.8, colorDrift: 1.9 },
        favorite: true,
      },
      {
        id: 'clip-nordic-3',
        projectId: 'proj-nordic-wild',
        parentId: 'clip-nordic-2',
        prompt: 'Aerial drone camera pulls back showing misty river gorge and morning light sweeping over conifers',
        seed: 78234,
        durationSec: 5.0,
        status: 'done',
        progress: 1.0,
        createdAt: Date.now() - 1000 * 60 * 60 * 2,
        continuity: { seam: 93.1, identity: 91.2, colorDrift: 2.8 },
      },
    ],
  },
  {
    id: 'proj-cyber-city',
    name: 'Cyberpunk Transit 2088',
    targetSeconds: 20,
    createdAt: Date.now() - 1000 * 60 * 60 * 48,
    updatedAt: Date.now() - 1000 * 60 * 60 * 5,
    coverClipId: 'clip-cyber-1',
    clips: [
      {
        id: 'clip-cyber-1',
        projectId: 'proj-cyber-city',
        prompt: 'A futuristic cyber rover navigating wet neon alleys in rain, steady tracking camera shot',
        seed: 10452,
        durationSec: 5.0,
        status: 'done',
        progress: 1.0,
        createdAt: Date.now() - 1000 * 60 * 60 * 40,
        continuity: { seam: 97.0, identity: 95.0, colorDrift: 1.8 },
      },
      {
        id: 'clip-cyber-2',
        projectId: 'proj-cyber-city',
        parentId: 'clip-cyber-1',
        prompt: 'Hover craft speeds overhead through neon skyscrapers, water splashing against camera lens',
        seed: 10452,
        durationSec: 5.0,
        status: 'done',
        progress: 1.0,
        createdAt: Date.now() - 1000 * 60 * 60 * 12,
        continuity: { seam: 94.8, identity: 93.1, colorDrift: 2.2 },
      },
    ],
  },
  {
    id: 'proj-ocean-aerial',
    name: 'Turquoise Coral Coastline',
    targetSeconds: 15,
    createdAt: Date.now() - 1000 * 60 * 60 * 12,
    updatedAt: Date.now() - 1000 * 60 * 45,
    coverClipId: 'clip-ocean-1',
    clips: [
      {
        id: 'clip-ocean-1',
        projectId: 'proj-ocean-aerial',
        prompt: 'A vintage sailboat cutting through turquoise ocean waves, aerial cinematic pan at sunset',
        seed: 88402,
        durationSec: 5.0,
        status: 'done',
        progress: 1.0,
        createdAt: Date.now() - 1000 * 60 * 60 * 10,
        continuity: { seam: 99.1, identity: 97.4, colorDrift: 0.9 },
        favorite: true,
      },
    ],
  },
];

/**
 * Creates default dashboard storage state
 */
export function createDefaultStorageState(): DashboardStorageState {
  const { usage } = processMonthlyUsage(null);
  return {
    version: 1,
    projects: DEFAULT_INITIAL_PROJECTS,
    activeProjectId: DEFAULT_INITIAL_PROJECTS[0].id,
    jobs: [],
    usage,
    settings: {
      concurrency: 1,
      simulateFailures: false,
      defaultTargetSeconds: 30,
      defaultContextFrames: 24,
    },
  };
}

/**
 * Saves dashboard state safely to localStorage
 */
export function saveToLocalStorage(state: DashboardStorageState): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const serialized = JSON.stringify(state);
    localStorage.setItem(STORAGE_KEY, serialized);
    return true;
  } catch (err) {
    console.warn('Failed to save Vidgen dashboard state to localStorage:', err);
    return false;
  }
}

/**
 * Loads dashboard state from localStorage with fallback to default
 */
export function loadFromLocalStorage(): DashboardStorageState {
  if (typeof window === 'undefined') {
    return createDefaultStorageState();
  }

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      const initial = createDefaultStorageState();
      saveToLocalStorage(initial);
      return initial;
    }

    const parsed = JSON.parse(raw) as DashboardStorageState;
    if (!parsed || !Array.isArray(parsed.projects)) {
      return createDefaultStorageState();
    }

    // Verify monthly rollover
    const { usage } = processMonthlyUsage(parsed.usage);
    parsed.usage = usage;

    return parsed;
  } catch (err) {
    console.warn('Corrupt localStorage state encountered, resetting to default:', err);
    return createDefaultStorageState();
  }
}

/**
 * Exports a single project to JSON string
 */
export function exportProjectToJson(project: Project): string {
  const exportPayload = {
    app: 'vidgen',
    schemaVersion: 1,
    exportedAt: new Date().toISOString(),
    project,
  };
  return JSON.stringify(exportPayload, null, 2);
}

/**
 * Imports and validates a project from JSON string
 */
export function importProjectFromJson(jsonString: string): Project {
  let parsed: any;
  try {
    parsed = JSON.parse(jsonString);
  } catch {
    throw new Error('Invalid JSON format: Could not parse input text.');
  }

  const rawProject = parsed.project || parsed;

  if (!rawProject || typeof rawProject !== 'object') {
    throw new Error('Invalid project structure: Root must be a Project object.');
  }

  if (!rawProject.name || typeof rawProject.name !== 'string') {
    throw new Error('Invalid project: Missing or invalid "name" attribute.');
  }

  if (!Array.isArray(rawProject.clips)) {
    rawProject.clips = [];
  }

  const newProject: Project = {
    id: `proj-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    name: `${rawProject.name} (Imported)`,
    targetSeconds: Number(rawProject.targetSeconds) || 30,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    coverClipId: rawProject.coverClipId,
    clips: rawProject.clips.map((c: any, index: number): Clip => ({
      id: c.id || `clip-imported-${Date.now()}-${index}`,
      projectId: rawProject.id || 'imported',
      parentId: c.parentId,
      prompt: String(c.prompt || 'Imported scene clip'),
      seed: Number(c.seed) || 42,
      durationSec: Number(c.durationSec) || 5.0,
      status: c.status === 'done' ? 'done' : 'queued',
      progress: c.status === 'done' ? 1.0 : 0,
      createdAt: Number(c.createdAt) || Date.now(),
      continuity: c.continuity,
      favorite: Boolean(c.favorite),
      thumbnailUrl: c.thumbnailUrl,
    })),
  };

  return newProject;
}
