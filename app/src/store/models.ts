// Model files on the device: download from a URL or import from the Files app.

import * as DocumentPicker from "expo-document-picker";
import { File, Paths } from "expo-file-system";

import { appDir } from "./files";

export type ModelFile = { filename: string; url: string; bytes: number };
export type ModelSpec = {
  id: string;
  name: string;
  description: string;
  model: ModelFile;
  mmproj: ModelFile | null;
};

const HF = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/";

export const CATALOG: ModelSpec[] = [
  {
    id: "candid-4b",
    name: "Candid 4B",
    description: "Our trained model. Import the two .gguf files from the training run.",
    model: { filename: "candid-4b-Q4_K_M.gguf", url: "", bytes: 2_741_241_728 },
    mmproj: { filename: "candid-4b-mmproj-f16.gguf", url: "", bytes: 672_422_944 },
  },
  {
    id: "qwen3.5-4b",
    name: "Qwen3.5 4B (base)",
    description: "The untrained base model, to try the app before Candid is available.",
    model: { filename: "Qwen3.5-4B-Q4_K_M.gguf", url: HF + "Qwen3.5-4B-Q4_K_M.gguf", bytes: 2_740_000_000 },
    mmproj: { filename: "mmproj-F16.gguf", url: HF + "mmproj-F16.gguf", bytes: 672_000_000 },
  },
];

const modelsDir = () => appDir("models");

export function modelFile(spec: ModelSpec, which: "model" | "mmproj"): File | null {
  const f = which === "model" ? spec.model : spec.mmproj;
  return f ? new File(modelsDir(), spec.id + "--" + f.filename) : null;
}

export function isInstalled(spec: ModelSpec): boolean {
  return Boolean(modelFile(spec, "model")?.exists) && (!spec.mmproj || Boolean(modelFile(spec, "mmproj")?.exists));
}

export function spaceNeeded(spec: ModelSpec): number {
  return spec.model.bytes + (spec.mmproj?.bytes ?? 0);
}

export const freeSpace = (): number => Paths.availableDiskSpace;

export async function download(
  spec: ModelSpec,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  const parts = [spec.model, spec.mmproj].filter((p): p is ModelFile => Boolean(p));
  if (parts.some((p) => !p.url)) throw new Error("This model has no download link yet. Import its files instead.");
  const total = parts.reduce((a, p) => a + p.bytes, 0);
  let done = 0;
  for (const [i, part] of parts.entries()) {
    const dest = modelFile(spec, i === 0 ? "model" : "mmproj")!;
    if (dest.exists) {
      done += part.bytes;
      continue;
    }
    const tmp = new File(modelsDir(), dest.name + ".part");
    if (tmp.exists) tmp.delete();
    await File.downloadFileAsync(part.url, tmp, {
      idempotent: true,
      signal,
      onProgress: ({ bytesWritten }) => onProgress(Math.min(1, (done + bytesWritten) / total)),
    });
    tmp.moveSync(dest);
    done += part.bytes;
  }
  onProgress(1);
}

/** Pick .gguf files from the Files app; the one with "mmproj" in its name is the vision part. */
export async function importFiles(spec: ModelSpec): Promise<string[]> {
  const res = await DocumentPicker.getDocumentAsync({ multiple: true, copyToCacheDirectory: true, type: "*/*" });
  if (res.canceled) return [];
  const imported: string[] = [];
  for (const asset of res.assets) {
    if (!asset.name.toLowerCase().endsWith(".gguf")) continue;
    const which = /mmproj/i.test(asset.name) ? "mmproj" : "model";
    const dest = modelFile(spec, which);
    if (!dest) continue;
    if (dest.exists) dest.delete();
    new File(asset.uri).moveSync(dest);
    imported.push(`${which}: ${asset.name}`);
  }
  return imported;
}

export function remove(spec: ModelSpec): void {
  for (const which of ["model", "mmproj"] as const) {
    const f = modelFile(spec, which);
    if (f?.exists) f.delete();
  }
}
