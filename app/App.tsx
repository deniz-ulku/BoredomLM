import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useRef, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { loadEngine, type Engine } from "./src/llm/engine";
import { saveConversation } from "./src/store/history";
import { CATALOG, isInstalled, modelFile, type ModelSpec } from "./src/store/models";
import { loadSettings, saveSettings, type Settings } from "./src/store/settings";
import { ChatScreen } from "./src/ui/ChatScreen";
import { ModelsScreen } from "./src/ui/ModelsScreen";
import { SettingsScreen } from "./src/ui/SettingsScreen";
import { S } from "./src/ui/strings";
import { useTheme } from "./src/ui/theme";

type Tab = "chat" | "models" | "settings";

export default function App() {
  const t = useTheme();
  const [tab, setTab] = useState<Tab>("chat");
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [engine, setEngine] = useState<Engine | null>(null);
  const [loading, setLoading] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [chatKey, setChatKey] = useState(0);
  const engineRef = useRef<Engine | null>(null);

  const updateSettings = useCallback((s: Settings) => {
    setSettings(s);
    saveSettings(s);
  }, []);

  // (Re)load the model when the active model or the thinking setting changes.
  useEffect(() => {
    const spec = CATALOG.find((m) => m.id === settings.activeModelId);
    if (!spec || !isInstalled(spec)) return;
    let cancelled = false;
    (async () => {
      setLoading(0);
      setError("");
      await engineRef.current?.release();
      engineRef.current = null;
      setEngine(null);
      try {
        const e = await loadEngine(modelFile(spec, "model")!.uri, modelFile(spec, "mmproj")?.uri ?? null,
          { thinking: settings.thinking }, (p) => setLoading(p));
        if (cancelled) return void (await e.release());
        engineRef.current = e;
        setEngine(e);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [settings.activeModelId, settings.thinking]);

  const use = (spec: ModelSpec) => {
    updateSettings({ ...settings, activeModelId: spec.id });
    setTab("chat");
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={[styles.fill, { backgroundColor: t.bg }]} edges={["top", "bottom"]}>
        <StatusBar style="auto" />
        <View style={[styles.header, { borderBottomColor: t.line }]}>
          <Text style={[styles.brand, { color: t.text }]}>Candid</Text>
          {loading !== null && <Text style={{ color: t.muted }}>{S.loading} {Math.round(loading)}%</Text>}
        </View>
        {error ? <Text style={[styles.error, { color: t.danger }]}>{error}</Text> : null}
        <View style={styles.fill}>
          {tab === "chat" && (
            <ChatScreen key={chatKey} engine={engine} settings={settings} onOpenModels={() => setTab("models")} />
          )}
          {tab === "models" && <ModelsScreen activeId={settings.activeModelId} loading={loading} onUse={use} />}
          {tab === "settings" && (
            <SettingsScreen
              settings={settings}
              onChange={updateSettings}
              onClearChat={() => {
                saveConversation([]);
                setChatKey((k) => k + 1);
                setTab("chat");
              }}
            />
          )}
        </View>
        <View style={[styles.tabs, { borderTopColor: t.line, backgroundColor: t.surface }]}>
          {(["chat", "models", "settings"] as const).map((k) => (
            <Pressable key={k} style={styles.tab} onPress={() => setTab(k)} accessibilityRole="tab"
              accessibilityState={{ selected: tab === k }}>
              <Text style={{ color: tab === k ? t.accent : t.muted, fontWeight: tab === k ? "600" : "400" }}>
                {S[k]}
              </Text>
            </Pressable>
          ))}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  header: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 16,
    paddingVertical: 10, borderBottomWidth: 1 },
  brand: { fontSize: 18, fontWeight: "700", letterSpacing: 0.3 },
  error: { paddingHorizontal: 16, paddingVertical: 8 },
  tabs: { flexDirection: "row", borderTopWidth: 1 },
  tab: { flex: 1, alignItems: "center", paddingVertical: 12 },
});
