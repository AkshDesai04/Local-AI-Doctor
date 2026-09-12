export const AUTH_REQUIRED_EVENT = "lad:auth-required";
export const AUTH_CHANGED_EVENT = "lad:auth-changed";
export const AUTH_SESSION_KEY = "lad.auth_token";

let memoryToken: string | null | undefined;

export function getSessionAuthToken(): string | undefined {
  if (memoryToken === undefined) {
    try {
      memoryToken = window.sessionStorage.getItem(AUTH_SESSION_KEY);
    } catch {
      memoryToken = null;
    }
  }
  return memoryToken || undefined;
}

export function hasSessionAuthToken(): boolean {
  return getSessionAuthToken() !== undefined;
}

export function saveSessionAuthToken(value: string): void {
  const token = value.trim();
  if (!token) return;
  memoryToken = token;
  try {
    window.sessionStorage.setItem(AUTH_SESSION_KEY, token);
  } catch {
    // The in-memory copy still supports privacy-restricted browser contexts.
  }
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function clearSessionAuthToken(): void {
  memoryToken = null;
  try {
    window.sessionStorage.removeItem(AUTH_SESSION_KEY);
  } catch {
    // The in-memory copy has already been cleared.
  }
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function notifyAuthenticationRequired(): void {
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
}
