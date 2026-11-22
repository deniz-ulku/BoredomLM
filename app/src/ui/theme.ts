import { useColorScheme } from "react-native";

export type Theme = {
  bg: string; surface: string; text: string; muted: string; line: string;
  accent: string; accentText: string; userBubble: string; danger: string;
};

const light: Theme = {
  bg: "#f6f7f8", surface: "#ffffff", text: "#16191d", muted: "#626b75", line: "#dde1e5",
  accent: "#1f5f8b", accentText: "#ffffff", userBubble: "#e3edf4", danger: "#b3261e",
};

const dark: Theme = {
  bg: "#101315", surface: "#181c1f", text: "#e8ebee", muted: "#98a2ac", line: "#2a3035",
  accent: "#7fb6dc", accentText: "#0b1a24", userBubble: "#1d2a33", danger: "#f2b8b5",
};

export const useTheme = (): Theme => (useColorScheme() === "dark" ? dark : light);
