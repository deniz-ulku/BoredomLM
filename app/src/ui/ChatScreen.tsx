import * as ImagePicker from "expo-image-picker";
import { File } from "expo-file-system";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator, FlatList, Image, KeyboardAvoidingView, Linking, Modal, Platform, Pressable, StyleSheet, Text,
  TextInput, View,
} from "react-native";

import { toChatMessage, type Engine } from "../llm/engine";
import { buildSystemPrompt } from "../llm/prompt";
import { deepThink } from "../llm/deepThink";
import { runChat, type ChatMessage, type LoopEvent } from "../llm/toolLoop";
import { makePair, regenerationPairs, type PromptTurn } from "../learn/pairs";
import { citedSources, guessLang } from "../search/citations";
import { runTool, TOOLS, type ToolContext } from "../search/tools";
import { buildStyleProfile } from "../style/imitate";
import { appDir } from "../store/files";
import { loadConversation, newId, saveConversation, type UiMessage } from "../store/history";
import { savePair, savePairs } from "../store/learning";
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
  const [deep, setDeep] = useState(false);
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

  /** Answer the last user message of `history` and append the reply. */
  const respond = useCallback(async (history: UiMessage[], versions?: string[]) => {
    if (!engine) return;
    const last = history[history.length - 1];
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
    const ctx: ToolContext = { searxngUrl: settings.searxngUrl, lang: guessLang(last?.text ?? ""), sources: [] };

    const onEvent = (e: LoopEvent | { kind: "attempt"; index: number; total: number }) => {
      if (e.kind === "text") setLive(e.text);
      if (e.kind === "attempt") {
        setLive("");
        setStatus(`${S.attempt} ${e.index}/${e.total}`);
      }
      if (e.kind === "tool" && e.name !== "calculate") {
        setLive("");
        setStatus(e.name === "web_search" ? `${S.searching}: ${safeArg(e.args, "query")}` : `${S.reading}: ${safeArg(e.args, "url")}`);
      }
    };
    const tools = settings.searchEnabled ? TOOLS : undefined;
    const toolRunner = (name: string, args: string) => runTool(name, args, ctx);

    let answer = "";
    let deepMeta: UiMessage["deep"];
    try {
      if (deep) {
        const res = await deepThink({ complete: engine.complete, messages: chat, attempts: 3, tools,
          runTool: toolRunner, onEvent });
        answer = res.answer;
        deepMeta = { support: res.support, total: res.total, agreed: res.agreed };
      } else {
        answer = (await runChat({ complete: engine.complete, messages: chat, tools, runTool: toolRunner, onEvent }))
          .answer;
      }
    } catch (e) {
      answer = `${S.failed}: ${(e as Error).message}`;
    }
    const cited = citedSources(answer, ctx.sources);
    setMessages([...history, { id: newId(), role: "assistant", text: answer, images: [], sources: cited,
      deep: deepMeta, versions, createdAt: Date.now() }]);
    setBusy(false);
    setLive("");
    setStatus("");
  }, [engine, settings, deep]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!engine || busy || (!text && images.length === 0)) return;
    // Moving on after regenerating means the shown version was kept: learn it over the replaced ones.
    const prev = messages[messages.length - 1];
    if (settings.learnFromMe && prev?.role === "assistant" && prev.versions?.length) {
      savePairs(regenerationPairs(toTurns(messages.slice(0, -1)), prev.versions, prev.text, Date.now(), newId));
    }
    const history = [...messages.map((m) => (m === prev ? { ...m, versions: undefined } : m)),
      { id: newId(), role: "user" as const, text, images, createdAt: Date.now() }];
    setMessages(history);
    setInput("");
    setImages([]);
    await respond(history);
  }, [engine, busy, input, images, messages, settings, respond]);

  const regenerate = useCallback(async () => {
    const prev = messages[messages.length - 1];
    if (!engine || busy || prev?.role !== "assistant") return;
    const history = messages.slice(0, -1);
    setMessages(history);
    await respond(history, [...(prev.versions ?? []), prev.text]);
  }, [engine, busy, messages, respond]);

  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const saveEdit = useCallback(() => {
    if (!editing) return;
    const i = messages.findIndex((m) => m.id === editing.id);
    if (i < 0) return setEditing(null);
    const old = messages[i];
    if (settings.learnFromMe) {
      savePair(makePair("edit", toTurns(messages.slice(0, i)), editing.text, old.text, Date.now(), newId()));
    }
    setMessages(messages.map((m) => (m.id === editing.id ? { ...m, text: editing.text.trim(), edited: true } : m)));
    setEditing(null);
  }, [editing, messages, settings.learnFromMe]);

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
        renderItem={({ item, index }) => (
          <Bubble m={item} t={t} onEdit={item.role === "assistant" && !busy
            ? () => setEditing({ id: item.id, text: item.text }) : undefined}
            onRegenerate={!busy && item.role === "assistant" && index === messages.length - 1 ? regenerate : undefined} />
        )}
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
        <Pressable style={[s.iconBtn, deep && s.iconBtnOn]} onPress={() => setDeep((d) => !d)}
          accessibilityRole="switch" accessibilityState={{ checked: deep }} accessibilityLabel={S.deep}>
          <Text style={[s.iconText, deep && s.iconTextOn]}>{S.deep}</Text>
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
      <Modal visible={editing !== null} animationType="slide" onRequestClose={() => setEditing(null)}>
        <View style={[s.fill, s.editor]}>
          <Text style={s.title}>{S.editTitle}</Text>
          <Text style={s.status}>{settings.learnFromMe ? S.editHintLearn : S.editHint}</Text>
          <TextInput style={[s.input, s.editInput]} value={editing?.text ?? ""} multiline autoFocus
            onChangeText={(text) => setEditing((e) => (e ? { ...e, text } : e))} />
          <View style={s.row}>
            <Pressable style={s.iconBtn} onPress={() => setEditing(null)}><Text style={s.iconText}>{S.cancel}</Text></Pressable>
            <Pressable style={s.send} onPress={saveEdit}><Text style={s.primaryText}>{S.save}</Text></Pressable>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

function toTurns(messages: UiMessage[]): PromptTurn[] {
  return messages.map((m) => ({ role: m.role, content: m.text || S.photo }));
}

type BubbleProps = { m: UiMessage; t: Theme; onEdit?: () => void; onRegenerate?: () => void };

function Bubble({ m, t, onEdit, onRegenerate }: BubbleProps) {
  const s = styles(t);
  return (
    <Pressable onLongPress={onEdit} delayLongPress={350} style={m.role === "user" ? s.user : s.assistant}>
      {m.images.length > 0 && (
        <View style={s.pending}>{m.images.map((u) => <Image key={u} source={{ uri: u }} style={s.thumb} />)}</View>
      )}
      {m.text ? <Text style={s.text} selectable>{m.text}</Text> : null}
      {m.deep ? (
        <Text style={[s.status, !m.deep.agreed && s.warn]}>
          {m.deep.agreed ? `${m.deep.support} ${S.agreedOf} ${m.deep.total} ${S.agreedTail}` : S.notAgreed}
        </Text>
      ) : null}
      {m.sources?.map((src) => (
        <Text key={src.n} style={s.source} onPress={() => Linking.openURL(src.url)} numberOfLines={1}>
          [{src.n}] {src.title || src.url}
        </Text>
      ))}
      {m.role === "assistant" && (onRegenerate || m.edited) ? (
        <View style={s.row}>
          {m.edited ? <Text style={s.status}>{S.edited}</Text> : null}
          {onRegenerate ? <Text style={s.action} onPress={onRegenerate}>{S.regenerate}</Text> : null}
        </View>
      ) : null}
    </Pressable>
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
  iconBtnOn: { backgroundColor: t.accent, borderColor: t.accent },
  iconTextOn: { color: t.accentText, fontWeight: "600" },
  warn: { color: t.danger },
  row: { flexDirection: "row", alignItems: "center", gap: 12 },
  action: { color: t.accent, fontSize: 13, paddingVertical: 4 },
  editor: { padding: 16, paddingTop: 56, gap: 12 },
  editInput: { flex: 1, maxHeight: undefined, textAlignVertical: "top" },
  send: { height: 40, paddingHorizontal: 14, borderRadius: 10, justifyContent: "center", backgroundColor: t.accent },
  primary: { paddingHorizontal: 18, paddingVertical: 12, borderRadius: 10, backgroundColor: t.accent },
  primaryText: { color: t.accentText, fontWeight: "600" },
});
