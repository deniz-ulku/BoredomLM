import { useRef, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import {
  CATALOG, download, freeSpace, importFiles, isInstalled, remove, spaceNeeded, type ModelSpec,
} from "../store/models";
import { getHfToken } from "../store/secrets";
import { S } from "./strings";
import { useTheme, type Theme } from "./theme";

type Props = { activeId: string | null; loading: number | null; onUse: (spec: ModelSpec) => void };

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

export function ModelsScreen({ activeId, loading, onUse }: Props) {
  const t = useTheme();
  const s = styles(t);
  const [, refresh] = useState(0);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [message, setMessage] = useState<Record<string, string>>({});
  const aborts = useRef<Record<string, AbortController>>({});

  const say = (id: string, text: string) => setMessage((m) => ({ ...m, [id]: text }));

  async function onDownload(spec: ModelSpec) {
    const ctrl = new AbortController();
    aborts.current[spec.id] = ctrl;
    say(spec.id, "");
    try {
      await download(spec, (f) => setProgress((p) => ({ ...p, [spec.id]: f })), ctrl.signal, await getHfToken());
    } catch (e) {
      say(spec.id, (e as Error).message);
    } finally {
      delete aborts.current[spec.id];
      setProgress((p) => {
        const { [spec.id]: _, ...rest } = p;
        return rest;
      });
      refresh((n) => n + 1);
    }
  }

  async function onImport(spec: ModelSpec) {
    try {
      const done = await importFiles(spec);
      if (done.length) say(spec.id, `${S.imported}: ${done.join(", ")}`);
    } catch (e) {
      say(spec.id, (e as Error).message);
    }
    refresh((n) => n + 1);
  }

  return (
    <ScrollView style={s.fill} contentContainerStyle={s.page}>
      {CATALOG.map((spec) => {
        const installed = isInstalled(spec);
        const p = progress[spec.id];
        const active = activeId === spec.id;
        return (
          <View key={spec.id} style={s.card}>
            <View style={s.row}>
              <Text style={s.name}>{spec.name}</Text>
              <Text style={[s.badge, active && s.badgeActive]}>
                {active ? S.active : installed ? S.installed : S.notInstalled}
              </Text>
            </View>
            <Text style={s.muted}>{spec.description}</Text>
            <Text style={s.muted}>
              {S.needSpace} {gb(spaceNeeded(spec))} · {gb(freeSpace())} {S.free}
            </Text>
            {p !== undefined && (
              <View style={s.bar}><View style={[s.barFill, { width: `${Math.round(p * 100)}%` }]} /></View>
            )}
            {active && loading !== null && (
              <Text style={s.muted}>{S.loading} {Math.round(loading)}%</Text>
            )}
            {message[spec.id] ? <Text style={s.note}>{message[spec.id]}</Text> : null}
            {!spec.model.url && !installed ? <Text style={s.note}>{S.noLink}</Text> : null}
            <View style={s.actions}>
              {installed && !active && <Btn t={t} label={S.use} primary onPress={() => onUse(spec)} />}
              {!installed && p === undefined && spec.model.url !== "" && (
                <Btn t={t} label={S.download} primary onPress={() => onDownload(spec)} />
              )}
              {p !== undefined && <Btn t={t} label={S.cancel} onPress={() => aborts.current[spec.id]?.abort()} />}
              {!installed && p === undefined && <Btn t={t} label={S.importFiles} onPress={() => onImport(spec)} />}
              {installed && !active && (
                <Btn t={t} label={S.delete} onPress={() => (remove(spec), refresh((n) => n + 1))} />
              )}
            </View>
          </View>
        );
      })}
    </ScrollView>
  );
}

function Btn({ t, label, onPress, primary }: { t: Theme; label: string; onPress: () => void; primary?: boolean }) {
  const s = styles(t);
  return (
    <Pressable style={[s.btn, primary && s.btnPrimary]} onPress={onPress} accessibilityRole="button">
      <Text style={primary ? s.btnPrimaryText : s.btnText}>{label}</Text>
    </Pressable>
  );
}

const styles = (t: Theme) => StyleSheet.create({
  fill: { flex: 1, backgroundColor: t.bg },
  page: { padding: 16, gap: 12 },
  card: { backgroundColor: t.surface, borderRadius: 12, borderWidth: 1, borderColor: t.line, padding: 14, gap: 8 },
  row: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 },
  name: { color: t.text, fontSize: 17, fontWeight: "600", flexShrink: 1 },
  badge: { color: t.muted, fontSize: 12 },
  badgeActive: { color: t.accent, fontWeight: "600" },
  muted: { color: t.muted, fontSize: 14 },
  note: { color: t.text, fontSize: 13 },
  bar: { height: 6, borderRadius: 3, backgroundColor: t.line, overflow: "hidden" },
  barFill: { height: 6, backgroundColor: t.accent },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 4 },
  btn: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 9, borderWidth: 1, borderColor: t.line },
  btnText: { color: t.text },
  btnPrimary: { backgroundColor: t.accent, borderColor: t.accent },
  btnPrimaryText: { color: t.accentText, fontWeight: "600" },
});
