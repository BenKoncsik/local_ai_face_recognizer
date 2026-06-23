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

export interface BBoxPct {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface PointPct {
  left: number;
  top: number;
}

// A tagged object's appearance in one photo. Mirrors the Python exporter's
// _object_record_for_export. Coordinates use the same image-relative percentage
// space as faces. `bbox`/`point`/`confidence` are optional — a record may carry
// only a name (e.g. legacy data or geometry-less occurrences).
export interface ObjectRecord {
  object_id: number;
  name: string;
  confidence: number | null;
  note: string;
  bbox: BBoxPct | null;
  point: PointPct | null;
}

export interface PairVariant {
  bw: string;
  color: string;
}

export interface CompareMember {
  label: string;
  src: string;
  isBw: boolean;
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
  // Optional: bundles produced before object-tagging export won't have this key.
  objects?: ObjectRecord[];
  personIds: number[];
  persons: string[];
  pair: PairVariant | null;
  // Full comparison group (B&W original + colorized variants). Older bundles
  // without this key fall back to the simple two-image `pair` slider.
  compare?: CompareMember[] | null;
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

// A person linked to an object, with a localized role label.
export interface ObjectPersonRef {
  id: number;
  name: string;
  role: string;
  roleLabel: string;
  note: string;
}

// A tagged-object collection record (mirrors Person for the objects pages).
export interface ObjectSummary {
  id: number;
  name: string;
  description: string;
  notes: string;
  thumb: string | null;
  thumbW: number;
  thumbH: number;
  imageCount: number;
  imageIds: string[];
  occurrenceCount: number;
  noteCount: number;
  persons: ObjectPersonRef[];
}

// A node in the interactive family-tree graph. Edges are person-id lists; the
// client lays out generation bands, recolours on selection and filters lineages.
export interface FamilyTreePerson {
  id: number;
  name: string;
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
  thumb: string | null;
  imageCount: number;
  /** Photos this person appears in (thumbnails), for the detail dialog. */
  photos: { id: string; thumb: string }[];
  generation: number;
  isMember: boolean;
  parents: number[];
  children: number[];
  spouses: number[];
  spouseMarriageDates: Record<string, { date: string; place: string }>;
}

export interface FamilyTree {
  persons: FamilyTreePerson[];
}

export interface Manifest {
  generatedAt: string;
  appVersion: string;
  pageSize: number;
  personCount: number;
  photoCount: number;
  objectCount: number;
  hasMap: boolean;
  hasSlideshow: boolean;
  hasObjects: boolean;
  hasCollages: boolean;
  hasFamilyTree: boolean;
  title: string;
  hasAuth: boolean;
  authToken: string;
}

// ---- Loaders ----

export function loadManifest(): Manifest {
  return readJson<Manifest>('manifest.json', {
    generatedAt: '',
    appVersion: '',
    pageSize: PAGE_SIZE,
    personCount: 0,
    photoCount: 0,
    objectCount: 0,
    hasMap: false,
    hasSlideshow: false,
    hasObjects: false,
    hasCollages: false,
    hasFamilyTree: false,
    title: 'Face Gallery',
    hasAuth: false,
    authToken: '',
  });
}

export function loadFamilyTree(): FamilyTree {
  return readJson<FamilyTree>('family-tree.json', { persons: [] });
}

export function loadPersons(): Person[] {
  return readJson<Person[]>('persons.json', []);
}

export function loadObjects(): ObjectSummary[] {
  return readJson<ObjectSummary[]>('objects.json', []);
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
