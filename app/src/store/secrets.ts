// Secrets live in the iOS Keychain / Android Keystore, never in the settings file.
import * as SecureStore from "expo-secure-store";

const HF = "hf_token";

export const getHfToken = (): Promise<string | null> => SecureStore.getItemAsync(HF);
export const setHfToken = async (token: string): Promise<void> => {
  if (token.trim()) await SecureStore.setItemAsync(HF, token.trim());
  else await SecureStore.deleteItemAsync(HF);
};
