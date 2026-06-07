// Build-time data access layer.
//
// All JSON in `src/data/` is produced by the Python exporter
// (ExportService.export_astro). These loaders run only during `astro build`
// (in Node), so reading the full dataset here is fine — the browser never sees
// these files. Pages slice the data into static pages; only one page's worth of
// markup ships to any given HTML file.

import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

// `astro build` runs from the project root (both via the Python exporter's
// subprocess cwd and a manual `npm run build`), so resolve the generated bundle
// relative to cwd. import.meta.url is unreliable here — Vite bundles this module
// into an SSR chunk whose URL no longer points at src/lib.
const DATA_DIR = join(process.cwd(), 'src', 'data');

export const PAGE_SIZE = 60;

function readJson<T>(name: string, fallback: T): T {
  const path = join(DATA_DIR, name);
  if (!existsSync(path)) return fallback;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as T;
  } catch {
    return fallback;
  }
}

// ---- Types (mirror the Python bundle writers) ----

export interface FaceRecord {
  face_id: number;
  person_id: number | null;
  name: string;
  bbox: { left: number; top: number; width: number; height: number };
}

export interface PairVariant {
  bw: string;
  color: string;
}

export interface Photo {
  id: string;
  thumb: string;
  thumbW: number;
  thumbH: number;
  medium: string;
  mediumW: number;
  mediumH: number;
  original: string;
  width: number;
  height: number;
  fileName: string;
  folder: string;
  date: string;
  faces: FaceRecord[];
  personIds: number[];
  persons: string[];
  pair: PairVariant | null;
  hasGps: boolean;
}

export interface PersonRelationship {
  relatedPersonId: number;
  relatedName: string;
  label: string;
}

export interface Person {
  id: number;
  name: string;
  isAutoNamed: boolean;
  thumb: string | null;
  thumbW: number;
  thumbH: number;
  imageCount: number;
  imageIds: string[];
  faceThumbs: string[];
  gender: string;
  nickname: string;
  marriedName: string;
  familyCode: string;
  externalFamilyCode: string;
  birthDate: string;
  birthPlace: string;
  deathDate: string;
  deathPlace: string;
  birthYear: string;
  deathYear: string;
  notes: string;
  groups: string[];
  relationships: PersonRelationship[];
}

export interface Manifest {
  generatedAt: string;
  pageSize: number;
  personCount: number;
  photoCount: number;
  hasMap: boolean;
  hasSlideshow: boolean;
  hasCollages: boolean;
  title: string;
}

// ---- Loaders ----

export function loadManifest(): Manifest {
  return readJson<Manifest>('manifest.json', {
    generatedAt: '',
    pageSize: PAGE_SIZE,
    personCount: 0,
    photoCount: 0,
    hasMap: false,
    hasSlideshow: false,
    hasCollages: false,
    title: 'Face Gallery',
  });
}

export function loadPersons(): Person[] {
  return readJson<Person[]>('persons.json', []);
}

export function loadPhotos(): Photo[] {
  return readJson<Photo[]>('photos.json', []);
}

// ---- Pagination ----

export interface PageInfo<T> {
  items: T[];
  page: number;
  pageCount: number;
  total: number;
}

export function paginate<T>(items: T[], page: number, size = PAGE_SIZE): PageInfo<T> {
  const pageCount = Math.max(1, Math.ceil(items.length / size));
  const safe = Math.min(Math.max(1, page), pageCount);
  const start = (safe - 1) * size;
  return {
    items: items.slice(start, start + size),
    page: safe,
    pageCount,
    total: items.length,
  };
}

// getStaticPaths helper: one path per page (1-based).
export function pagePaths<T>(items: T[], size = PAGE_SIZE) {
  const pageCount = Math.max(1, Math.ceil(items.length / size));
  return Array.from({ length: pageCount }, (_, i) => {
    const page = i + 1;
    return {
      params: { page: String(page) },
      props: { page: paginate(items, page, size) },
    };
  });
}
