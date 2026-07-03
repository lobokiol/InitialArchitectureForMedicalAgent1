const LS_ACCESS = 'triage_access_token';
const LS_REFRESH = 'triage_refresh_token';

export function getAccessToken(): string | null {
  return localStorage.getItem(LS_ACCESS);
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(LS_REFRESH);
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem(LS_ACCESS, access);
  localStorage.setItem(LS_REFRESH, refresh);
}

export function clearTokens() {
  localStorage.removeItem(LS_ACCESS);
  localStorage.removeItem(LS_REFRESH);
}
