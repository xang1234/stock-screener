(() => {
  const BRIDGE_CHANNEL = "xui-session-bridge";
  const REQUEST_TYPE = "XUI_BRIDGE_REQUEST";
  const RESPONSE_TYPE = "XUI_BRIDGE_RESPONSE";
  const ALLOWED_ACTIONS = new Set(["PING", "CAPTURE_AND_IMPORT"]);
  const ALLOWED_ORIGINS = new Set([
    "http://localhost:80",
    "http://127.0.0.1:80",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://localhost:443",
    "https://127.0.0.1:443",
  ]);

  function normalizeOrigin(value) {
    try {
      const parsed = new URL(value);
      const isHttp = parsed.protocol === "http:" || parsed.protocol === "https:";
      if (!isHttp || !parsed.hostname) return null;
      const hasPort = Boolean(parsed.port);
      const port = hasPort ? parsed.port : parsed.protocol === "https:" ? "443" : "80";
      return `${parsed.protocol}//${parsed.hostname.toLowerCase()}:${port}`;
    } catch (_error) {
      return null;
    }
  }

  function postResponse(requestId, ok, payload, error) {
    window.postMessage(
      {
        channel: BRIDGE_CHANNEL,
        type: RESPONSE_TYPE,
        requestId,
        ok,
        payload: payload || null,
        error: error || null,
      },
      window.location.origin,
    );
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const normalizedEventOrigin = normalizeOrigin(event.origin);
    if (!normalizedEventOrigin || !ALLOWED_ORIGINS.has(normalizedEventOrigin)) return;
    const data = event.data;
    if (!data || data.channel !== BRIDGE_CHANNEL || data.type !== REQUEST_TYPE) return;
    const requestId = data.requestId;
    if (!requestId) return;
    if (!ALLOWED_ACTIONS.has(data.action)) {
      postResponse(requestId, false, null, `Unsupported action: ${String(data.action || "")}`);
      return;
    }

    chrome.runtime.sendMessage(
      {
        type: data.action,
        payload: {
          ...(data.payload || {}),
          app_origin: normalizedEventOrigin,
        },
      },
      (response) => {
        const runtimeError = chrome.runtime.lastError;
        if (runtimeError) {
          postResponse(requestId, false, null, runtimeError.message);
          return;
        }
        if (!response || response.ok !== true) {
          postResponse(requestId, false, null, response?.error || "Bridge request failed.");
          return;
        }
        postResponse(requestId, true, response.payload || {}, null);
      },
    );
  });
})();
