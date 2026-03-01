const EXTENSION_VERSION = "0.1.0";
const LOCAL_IMPORT_HOSTS = new Set(["localhost", "127.0.0.1"]);

function toPromise(fn) {
  return new Promise((resolve, reject) => {
    fn((value) => {
      const runtimeError = chrome.runtime.lastError;
      if (runtimeError) {
        reject(new Error(runtimeError.message));
        return;
      }
      resolve(value);
    });
  });
}

function getCookiesForDomain(url) {
  return toPromise((done) => chrome.cookies.getAll({ url }, done));
}

function normalizeCookie(cookie) {
  return {
    name: cookie.name,
    value: cookie.value,
    domain: cookie.domain,
    path: cookie.path || "/",
    secure: Boolean(cookie.secure),
    httpOnly: Boolean(cookie.httpOnly),
    sameSite: cookie.sameSite || "unspecified",
    expirationDate: typeof cookie.expirationDate === "number" ? cookie.expirationDate : null,
  };
}

function dedupeCookies(cookies) {
  const byKey = new Map();
  cookies.forEach((cookie) => {
    const key = `${cookie.name}|${cookie.domain}|${cookie.path}`;
    byKey.set(key, cookie);
  });
  return Array.from(byKey.values());
}

async function captureXCookies() {
  const [xCookies, twitterCookies] = await Promise.all([
    getCookiesForDomain("https://x.com"),
    getCookiesForDomain("https://twitter.com"),
  ]);
  const normalized = dedupeCookies([...(xCookies || []), ...(twitterCookies || [])].map(normalizeCookie));
  return {
    cookies: normalized,
    browser: navigator.userAgent,
    extension_version: EXTENSION_VERSION,
  };
}

function validateImportUrl(importUrl) {
  if (typeof importUrl !== "string" || !importUrl.trim()) {
    throw new Error("Missing import_url.");
  }
  let parsed;
  try {
    parsed = new URL(importUrl.trim());
  } catch (_error) {
    throw new Error("import_url must be a valid URL.");
  }
  if (!["http:", "https:"].includes(parsed.protocol) || !LOCAL_IMPORT_HOSTS.has(parsed.hostname)) {
    throw new Error("import_url must target localhost or 127.0.0.1.");
  }
  if (parsed.username || parsed.password) {
    throw new Error("import_url cannot include credentials.");
  }
  return parsed.toString();
}

async function captureAndImport(payload) {
  const challengeId = String(payload?.challengeId || "").trim();
  const challengeToken = String(payload?.challengeToken || "").trim();
  const appOrigin = String(payload?.app_origin || "").trim();
  if (!challengeId || !challengeToken || !appOrigin) {
    throw new Error("Missing challenge credentials or app origin.");
  }
  const importUrl = validateImportUrl(payload?.importUrl);
  const captured = await captureXCookies();
  const response = await fetch(importUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-XUI-Bridge-Origin": appOrigin,
    },
    body: JSON.stringify({
      challenge_id: challengeId,
      challenge_token: challengeToken,
      cookies: captured.cookies,
      browser: captured.browser,
      extension_version: captured.extension_version,
    }),
  });
  let responseBody = {};
  try {
    responseBody = await response.json();
  } catch (_error) {
    responseBody = {};
  }
  if (!response.ok) {
    const detail = responseBody?.detail || `Import failed with HTTP ${response.status}`;
    throw new Error(String(detail));
  }
  return {
    imported: true,
    authenticated: Boolean(responseBody?.authenticated),
    status_code: String(responseBody?.status_code || ""),
    message: String(responseBody?.message || ""),
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const type = message?.type;
  if (!type) {
    sendResponse({ ok: false, error: "Missing message type." });
    return false;
  }

  if (type === "PING") {
    sendResponse({
      ok: true,
      payload: { extension_version: EXTENSION_VERSION },
    });
    return false;
  }

  if (type === "CAPTURE_AND_IMPORT") {
    captureAndImport(message?.payload || {})
      .then((payload) => {
        sendResponse({ ok: true, payload });
      })
      .catch((error) => {
        sendResponse({ ok: false, error: error?.message || "Failed to capture/import X session." });
      });
    return true;
  }

  sendResponse({ ok: false, error: `Unsupported bridge action: ${type}` });
  return false;
});
