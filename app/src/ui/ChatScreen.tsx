import * as ImagePicker from "expo-image-picker";
import { File } from "expo-file-system";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator, FlatList, Image, KeyboardAvoidingView, Linking, Platform, Pressable, StyleSheet, Text,
  TextInput, View,
} from "react-native";

import { toChatMessage, type Engine } from "../llm/engine";
import { buildSystemPrompt } from "../llm/prompt";
import { runChat, type ChatMessage } from "../llm/toolLoop";
import { citedSources, guessLang } from "../search/citations";
import { runTool, TOOLS, type ToolContext } from "../search/tools";
import { buildStyleProfile } from "../style/imitate";
import { appDir } from "../store/files";
import { loadConversation, newId, saveConversation, type UiMessage } from "../store/history";
import type { Settings } from "../store/settings";
import { S } from "./strings";
import { useTheme, type Theme } from "./theme";

const HISTORY_TURNS = 12; // past messages sent to the model
const IMAGE_TURNS = 2; // only the latest photos are re-sent; older ones cost too much context

type Props = { engine: Engine | null; settings: Settings; onOpenModels: () => void };

export function ChatScreen({ engine, settings, onOpenModels }: Props) {
  const t = useTheme();
  const s = styles(t);
  const [messages, setMessages] = useState<UiMessage[]>(() => loadConversation());
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState("");
  const [status, setStatus] = useState("");
  const list = useRef<FlatList<UiMessage>>(null);

  useEffect(() => saveConversation(messages), [messages]);

  const addPhotos = useCallback(async (camera: boolean) => {
    if (camera) {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) return;
    }
    const opts: ImagePicker.ImagePickerOptions = { mediaTypes: ["images"], quality: 0.85, allowsMultipleSelection: !camera };
    const res = camera ? await ImagePicker.launchCameraAsync(opts) : await ImagePicker.launchImageLibraryAsync(opts);
    if (res.canceled) return;
    const dir = appDir("images");
    const kept = res.assets.map((a) => {
      const dest = new File(dir, `${newId()}.jpg`);
      new File(a.uri).copySync(dest);
      return dest.uri;
    });
    setImages((prev) => [...prev, ...kept]);
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!engine || busy || (!text && images.length === 0)) return;
    const userMsg: UiMessage = { id: newId(), role: "user", text, images, createdAt: Date.now() };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setImages([]);
    setBusy(true);
    setLive("");
    setStatus("");

    const style = settings.imitateMe
      ? buildStyleProfile(history.filter((m) => m.role === "user").map((m) => m.text))
      : null;
    const recent = history.slice(-HISTORY_TURNS);
    const withImages = new Set(recent.filter((m) => m.images.length).slice(-IMAGE_TURNS).map((m) => m.id));
    const chat: ChatMessage[] = [
      { role: "system", content: buildSystemPrompt({ now: new Date(), searchEnabled: settings.searchEnabled,
        online: true, styleProfile: style }) },
      ...recent.map((m) => toChatMessage(m.role, m.text || S.photo,
        engine.vision && withImages.has(m.id) ? m.images : [])),
    ];
    const ctx: ToolContext = { searxngUrl: settings.searxngUrl, lang: guessLang(text), sources: [] };

    let answer = "";
    try {
      const out = await runChat({
        complete: engine.complete,
        messages: chat,
        tools: settings.searchEnabled ? TOOLS : undefined,
        runTool: (name, args) => runTool(name, args, ctx),
        onEvent: (e) => {
          if (e.kind === "text") setLive(e.text);
          if (e.kind === "tool") {
            setLive("");
            setStatus(e.name === "web_search" ? `${S.searching}: ${safeArg(e.args, "query")}` : `${S.reading}: ${safeArg(e.args, "url")}`);
          }
        },
      });
      answer = out.answer;
    } catch (e) {
      answer = `${S.failed}: ${(e as Error).message}`;
    }
    const cited = citedSources(answer, ctx.sources);
    setMessages((prev) => [...prev, { id: newId(), role: "assistant", text: answer, images: [],
      sources: cited, createdAt: Date.now() }]);
    setBusy(false);
    setLive("");
    setStatus("");
  }, [engine, busy, input, images, messages, settings]);

  if (!engine) {
    return (
      <View style={[s.fill, s.center]}>
        <Text style={s.title}>{S.noModel}</Text>
        <Text style={s.muted}>{S.noModelHint}</Text>
        <Pressable style={s.primary} onPress={onOpenModels}><Text style={s.primaryText}>{S.openModels}</Text></Pressable>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView style={s.fill} behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <FlatList
        ref={list}
        data={messages}
        keyExtractor={(m) => m.id}
        contentContainerStyle={s.list}
        onContentSizeChange={() => list.current?.scrollToEnd({ animated: true })}
        renderItem={({ item }) => <Bubble m={item} t={t} />}
        ListFooterComponent={busy ? (
          <View style={s.assistant}>
            {status ? <Text style={s.status}>{status}</Text> : null}
            {live ? <Text style={s.text}>{live}</Text> : <ActivityIndicator color={t.muted} />}
          </View>
        ) : undefined}
      />
      {images.length > 0 && (
        <View style={s.pending}>
          {images.map((u) => (
            <Pressable key={u} onLongPress={() => setImages((p) => p.filter((x) => x !== u))}>
              <Image source={{ uri: u }} style={s.thumb} />
            </Pressable>
          ))}
        </View>
      )}
      <View style={s.composer}>
        <Pressable style={s.iconBtn} onPress={() => addPhotos(true)} accessibilityLabel={S.camera}>
          <Text style={s.iconText}>{S.camera}</Text>
        </Pressable>
        <Pressable style={s.iconBtn} onPress={() => addPhotos(false)} accessibilityLabel={S.photos}>
          <Text style={s.iconText}>{S.photos}</Text>
        </Pressable>
        <TextInput
          style={s.input}
          value={input}
          onChangeText={setInput}
          placeholder={engine.vision ? S.askPhoto : S.ask}
          placeholderTextColor={t.muted}
          multiline
        />
        {busy ? (
          <Pressable style={s.send} onPress={() => engine.stop()}><Text style={s.primaryText}>{S.stop}</Text></Pressable>
        ) : (
          <Pressable style={s.send} onPress={send}><Text style={s.primaryText}>{S.send}</Text></Pressable>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

function Bubble({ m, t }: { m: UiMessage; t: Theme }) {
  const s = styles(t);
  return (
    <View style={m.role === "user" ? s.user : s.assistant}>
      {m.images.length > 0 && (
        <View style={s.pending}>{m.images.map((u) => <Image key={u} source={{ uri: u }} style={s.thumb} />)}</View>
      )}
      {m.text ? <Text style={s.text} selectable>{m.text}</Text> : null}
      {m.sources?.map((src) => (
        <Text key={src.n} style={s.source} onPress={() => Linking.openURL(src.url)} numberOfLines={1}>
          [{src.n}] {src.title || src.url}
        </Text>
      ))}
    </View>
  );
}

function safeArg(json: string, key: string): string {
  try {
    return String((JSON.parse(json) as Record<string, unknown>)[key] ?? "");
  } catch {
    return "";
  }
}

const styles = (t: Theme) => StyleSheet.create({
  fill: { flex: 1, backgroundColor: t.bg },
  center: { alignItems: "center", justifyContent: "center", gap: 12, padding: 24 },
  title: { color: t.text, fontSize: 20, fontWeight: "600" },
  muted: { color: t.muted, textAlign: "center" },
  list: { padding: 16, gap: 12 },
  user: { alignSelf: "flex-end", maxWidth: "88%", backgroundColor: t.userBubble, borderRadius: 14, padding: 12, gap: 8 },
  assistant: { alignSelf: "stretch", paddingVertical: 4, gap: 6 },
  text: { color: t.text, fontSize: 16, lineHeight: 23 },
  status: { color: t.muted, fontSize: 13 },
  source: { color: t.accent, fontSize: 13 },
  pending: { flexDirection: "row", flexWrap: "wrap", gap: 6, paddingHorizontal: 16, paddingTop: 6 },
  thumb: { width: 64, height: 64, borderRadius: 8, backgroundColor: t.line },
  composer: { flexDirection: "row", alignItems: "flex-end", gap: 8, padding: 10, borderTopWidth: 1,
    borderTopColor: t.line, backgroundColor: t.surface },
  input: { flex: 1, maxHeight: 140, minHeight: 40, color: t.text, fontSize: 16, paddingHorizontal: 12, paddingVertical: 9,
    borderRadius: 12, borderWidth: 1, borderColor: t.line, backgroundColor: t.bg },
  iconBtn: { height: 40, paddingHorizontal: 10, borderRadius: 10, justifyContent: "center", borderWidth: 1, borderColor: t.line },
  iconText: { color: t.text, fontSize: 13 },
  send: { height: 40, paddingHorizontal: 14, borderRadius: 10, justifyContent: "center", backgroundColor: t.accent },
  primary: { paddingHorizontal: 18, paddingVertical: 12, borderRadius: 10, backgroundColor: t.accent },
  primaryText: { color: t.accentText, fontWeight: "600" },
});
