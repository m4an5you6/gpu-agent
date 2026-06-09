declare global {
  interface Window {
    /** Set true by the server only for `gpucloud dashboard --tui` (or GPUCLOUD_DASHBOARD_TUI=1). */
    __GPUCLOUD_DASHBOARD_EMBEDDED_CHAT__?: boolean;
    /** @deprecated Older injected name; treated as on when true. */
    __GPUCLOUD_DASHBOARD_TUI__?: boolean;
  }
}

/** True only when the dashboard was started with embedded TUI Chat (`gpucloud dashboard --tui`). */
export function isDashboardEmbeddedChatEnabled(): boolean {
  if (typeof window === "undefined") return false;
  if (window.__GPUCLOUD_DASHBOARD_EMBEDDED_CHAT__ === true) return true;
  return window.__GPUCLOUD_DASHBOARD_TUI__ === true;
}
