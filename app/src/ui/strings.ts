// UI text in Turkish and English, picked from the device language.

const en = {
  chat: "Chat", models: "Models", settings: "Settings",
  noModel: "No model loaded", noModelHint: "Candid runs on your phone. Download or import a model first.",
  openModels: "Open models", ask: "Ask anything", askPhoto: "Ask anything, or add a photo",
  send: "Send", stop: "Stop", camera: "Camera", photos: "Photos",
  searching: "Searching", reading: "Reading", failed: "Something went wrong while answering",
  loading: "Loading model", installed: "Installed", notInstalled: "Not installed", active: "In use",
  use: "Use", download: "Download", importFiles: "Import files", delete: "Delete", cancel: "Cancel",
  needSpace: "Needs", free: "free", noLink: "No download link yet: import the .gguf files from the training run.",
  imported: "Imported", searchWeb: "Search the web", searchWebHint: "Searches run from your phone; nothing goes through our servers.",
  searxng: "SearXNG address (optional)", imitate: "Write like me",
  imitateHint: "Answers follow your own writing style, learned from your messages on this phone.",
  thinking: "Think before answering", thinkingHint: "Slower, better on hard problems.",
  clearChat: "Clear conversation", photo: "(photo)",
  deep: "Deep", attempt: "Solving", agreedOf: "of", agreedTail: "solutions agree",
  notAgreed: "The solutions disagree; check this answer.",
};

const tr: typeof en = {
  chat: "Sohbet", models: "Modeller", settings: "Ayarlar",
  noModel: "Model yüklü değil", noModelHint: "Candid telefonunda çalışır. Önce bir model indir ya da içe aktar.",
  openModels: "Modellere git", ask: "Bir şey sor", askPhoto: "Bir şey sor ya da fotoğraf ekle",
  send: "Gönder", stop: "Durdur", camera: "Kamera", photos: "Fotoğraf",
  searching: "Aranıyor", reading: "Okunuyor", failed: "Cevap verirken bir sorun çıktı",
  loading: "Model yükleniyor", installed: "Yüklü", notInstalled: "Yüklü değil", active: "Kullanılıyor",
  use: "Kullan", download: "İndir", importFiles: "Dosyaları içe aktar", delete: "Sil", cancel: "İptal",
  needSpace: "Gereken", free: "boş", noLink: "Henüz indirme linki yok: eğitimden çıkan .gguf dosyalarını içe aktar.",
  imported: "İçe aktarıldı", searchWeb: "Webde ara", searchWebHint: "Aramalar telefonundan yapılır; hiçbir şey bizim sunucularımızdan geçmez.",
  searxng: "SearXNG adresi (isteğe bağlı)", imitate: "Benim gibi yaz",
  imitateHint: "Cevaplar, bu telefondaki mesajlarından öğrenilen kendi yazım tarzını izler.",
  thinking: "Cevaplamadan önce düşün", thinkingHint: "Daha yavaş, zor sorularda daha iyi.",
  clearChat: "Sohbeti temizle", photo: "(fotoğraf)",
  deep: "Derin", attempt: "Çözülüyor", agreedOf: "/", agreedTail: "çözüm aynı sonuca vardı",
  notAgreed: "Çözümler farklı sonuçlara vardı; bu cevabı kontrol et.",
};

export type Strings = typeof en;

export function stringsFor(locale: string): Strings {
  return locale.toLowerCase().startsWith("tr") ? tr : en;
}

export const S: Strings = stringsFor(Intl.DateTimeFormat().resolvedOptions().locale ?? "en");
