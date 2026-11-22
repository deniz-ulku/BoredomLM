# Candid app

The phone app: one codebase for iPhone and Android (Expo + React Native), running the model on the device with [llama.rn](https://github.com/mybigday/llama.rn) (llama.cpp: Metal on iPhone, OpenCL/CPU on Android).

What it does:
- Chat with photos (camera or gallery, no limit). The vision projector reads them on the phone.
- Web search when online, from the phone itself: optional SearXNG, then DuckDuckGo, then Wikipedia. No API keys, no servers of ours. Answers cite sources as [n].
- Settings: web search on/off, optional SearXNG address, "Write like me" (answers follow your own style, learned on the phone from your messages), thinking mode for hard questions.
- Models: download the base Qwen3.5-4B to try the app, or import the trained Candid `.gguf` files (model + `mmproj`) from the Files app.
- Turkish or English UI, from the phone's language.

## Layout

```
App.tsx               tabs, model loading
src/llm/engine.ts     llama.rn context, multimodal, streaming completion
src/llm/toolLoop.ts   generate -> run tool calls -> repeat (pure, tested)
src/llm/prompt.ts     system prompt
src/search/           providers, HTML parsing, tools, citations (pure, tested)
src/style/imitate.ts  "Write like me" style profile (pure, tested)
src/store/            settings, conversation, model files
src/ui/               screens, theme, strings
```

## Develop

```bash
npm install
npm test            # unit tests for search, tool loop, prompt, style
npm run typecheck
npx expo run:ios    # needs a Mac with Xcode
npx expo run:android  # needs Android Studio / SDK
```

The app uses native code (llama.rn), so it does not run in Expo Go; use a development build (`expo run:*` or EAS Build).

Memory: the 4B model at Q4_K_M plus the vision projector needs about 3.5 GB of RAM. Target devices are iPhone 15 Pro or newer and Android phones with 8 GB of RAM or more.

## Status

Written and checked in a Linux cloud session: unit tests pass, TypeScript compiles, and Metro bundles the JavaScript for both iOS and Android. It has not yet been built into a native app or run on a phone.
