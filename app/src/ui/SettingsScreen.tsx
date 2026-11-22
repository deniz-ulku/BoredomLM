import { useEffect, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Switch, Text, TextInput, View } from "react-native";

import { clearPairs, loadPairs } from "../store/learning";
import { getHfToken, setHfToken } from "../store/secrets";
import type { Settings } from "../store/settings";
import { S } from "./strings";
import { useTheme, type Theme } from "./theme";

type Props = { settings: Settings; onChange: (s: Settings) => void; onClearChat: () => void };

export function SettingsScreen({ settings, onChange, onClearChat }: Props) {
  const t = useTheme();
  const s = styles(t);
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => onChange({ ...settings, [k]: v });
  const [learned, setLearned] = useState(() => loadPairs().length);
  const [token, setToken] = useState("");
  useEffect(() => {
    getHfToken().then((v) => setToken(v ?? "")).catch(() => undefined);
  }, []);

  return (
    <ScrollView style={s.fill} contentContainerStyle={s.page}>
      <Toggle t={t} label={S.searchWeb} hint={S.searchWebHint} value={settings.searchEnabled}
        onChange={(v) => set("searchEnabled", v)} />
      {settings.searchEnabled && (
        <View style={s.item}>
          <Text style={s.label}>{S.searxng}</Text>
          <TextInput
            style={s.input}
            value={settings.searxngUrl}
            onChangeText={(v) => set("searxngUrl", v.trim())}
            placeholder="https://searx.example.org"
            placeholderTextColor={t.muted}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
        </View>
      )}
      <Toggle t={t} label={S.imitate} hint={S.imitateHint} value={settings.imitateMe}
        onChange={(v) => set("imitateMe", v)} />
      <Toggle t={t} label={S.thinking} hint={S.thinkingHint} value={settings.thinking}
        onChange={(v) => set("thinking", v)} />
      <Toggle t={t} label={S.learn} hint={`${S.learnHint} ${learned} ${S.learnCount}.`} value={settings.learnFromMe}
        onChange={(v) => set("learnFromMe", v)} />
      {learned > 0 && (
        <Pressable style={s.danger} onPress={() => (clearPairs(), setLearned(0))} accessibilityRole="button">
          <Text style={s.dangerText}>{S.clearLearning}</Text>
        </Pressable>
      )}
      <View style={s.item}>
        <Text style={s.label}>{S.hfToken}</Text>
        <TextInput
          style={s.input}
          value={token}
          onChangeText={setToken}
          onEndEditing={() => setHfToken(token).catch(() => undefined)}
          placeholder="hf_..."
          placeholderTextColor={t.muted}
          autoCapitalize="none"
          autoCorrect={false}
          secureTextEntry
        />
      </View>
      <Pressable style={s.danger} onPress={onClearChat} accessibilityRole="button">
        <Text style={s.dangerText}>{S.clearChat}</Text>
      </Pressable>
    </ScrollView>
  );
}

function Toggle(p: { t: Theme; label: string; hint: string; value: boolean; onChange: (v: boolean) => void }) {
  const s = styles(p.t);
  return (
    <View style={[s.item, s.row]}>
      <View style={s.grow}>
        <Text style={s.label}>{p.label}</Text>
        <Text style={s.hint}>{p.hint}</Text>
      </View>
      <Switch value={p.value} onValueChange={p.onChange} trackColor={{ true: p.t.accent, false: p.t.line }} />
    </View>
  );
}

const styles = (t: Theme) => StyleSheet.create({
  fill: { flex: 1, backgroundColor: t.bg },
  page: { padding: 16, gap: 10 },
  item: { backgroundColor: t.surface, borderRadius: 12, borderWidth: 1, borderColor: t.line, padding: 14, gap: 6 },
  row: { flexDirection: "row", alignItems: "center", gap: 12 },
  grow: { flex: 1, gap: 3 },
  label: { color: t.text, fontSize: 16, fontWeight: "500" },
  hint: { color: t.muted, fontSize: 13 },
  input: { color: t.text, borderWidth: 1, borderColor: t.line, borderRadius: 9, paddingHorizontal: 10, paddingVertical: 8 },
  danger: { padding: 14, alignItems: "center" },
  dangerText: { color: t.danger, fontSize: 15 },
});
