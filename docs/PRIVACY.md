# Candid privacy policy (draft)

Last updated: 2026-11-22. Draft for the App Store and Google Play listings; publish it at a public URL before submitting.

## Short version

Candid runs its AI model on your phone. Your conversations, photos and learning data stay on the device. There is no Candid account, no Candid server and no analytics.

## What stays on your phone

- Conversations, including the photos you attach (copied into the app's own storage).
- Settings, and the Hugging Face token if you enter one (kept in the iOS Keychain / Android Keystore).
- "Learn from me" data: the corrections and kept answers the app learns from, and the personal adapter trained from them at night. They are never uploaded.

Deleting the app deletes all of it. Settings → "Clear conversation" and "Clear learned data" delete parts of it at any time.

## What leaves your phone, and only when you use it

| When | Sent to | What is sent |
|---|---|---|
| Web search is on and the model decides to search | DuckDuckGo (html.duckduckgo.com), or the SearXNG server you set in Settings | The search query the model wrote |
| Same | Wikipedia (xx.wikipedia.org) | The search query |
| The model opens a search result | The website of that result | A normal page request |
| You download a model | Hugging Face (huggingface.co) | A download request (with your token, if you set one) |

These services receive your phone's IP address like any website does, and their own privacy policies apply. Candid does not send them your conversation, only the query or page address. Turn off web search in Settings to keep every answer offline.

## What Candid does not do

- No accounts, ads, trackers, analytics or crash reporting.
- No selling or sharing of data: there is nothing on our side to share.

## Children

Candid is not directed at children under 13 (16 in the EU).

## Contact

[contact address to be added before publishing]
