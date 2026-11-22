import { Directory, File, Paths } from "expo-file-system";

export function appDir(name: string): Directory {
  const dir = new Directory(Paths.document, name);
  if (!dir.exists) dir.create({ intermediates: true });
  return dir;
}

export function readJson<T>(name: string, fallback: T): T {
  try {
    const f = new File(Paths.document, name);
    return f.exists ? { ...fallback, ...(JSON.parse(f.textSync()) as T) } : fallback;
  } catch {
    return fallback;
  }
}

export function readJsonArray<T>(name: string): T[] {
  try {
    const f = new File(Paths.document, name);
    return f.exists ? (JSON.parse(f.textSync()) as T[]) : [];
  } catch {
    return [];
  }
}

export function writeJson(name: string, value: unknown): void {
  const f = new File(Paths.document, name);
  if (!f.exists) f.create();
  f.write(JSON.stringify(value));
}
