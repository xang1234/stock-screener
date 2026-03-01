const BRIDGE_CHANNEL = 'xui-session-bridge';
const REQUEST_TYPE = 'XUI_BRIDGE_REQUEST';
const RESPONSE_TYPE = 'XUI_BRIDGE_RESPONSE';

function nextRequestId() {
  return `xui-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function sendBridgeRequest(action, payload = {}, timeoutMs = 5000) {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Bridge is unavailable outside browser context.'));
  }

  const requestId = nextRequestId();

  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error('XUI browser bridge timed out. Is the extension installed?'));
    }, timeoutMs);

    const onMessage = (event) => {
      if (event.source !== window) return;
      const data = event.data;
      if (!data || data.channel !== BRIDGE_CHANNEL || data.type !== RESPONSE_TYPE) return;
      if (data.requestId !== requestId) return;
      cleanup();
      if (data.ok) {
        resolve(data.payload || {});
      } else {
        reject(new Error(data.error || 'XUI browser bridge request failed.'));
      }
    };

    const cleanup = () => {
      window.clearTimeout(timer);
      window.removeEventListener('message', onMessage);
    };

    window.addEventListener('message', onMessage);
    window.postMessage(
      {
        channel: BRIDGE_CHANNEL,
        type: REQUEST_TYPE,
        requestId,
        action,
        payload,
      },
      window.location.origin,
    );
  });
}

export async function isBridgeAvailable() {
  try {
    await sendBridgeRequest('PING', {}, 1000);
    return true;
  } catch (_error) {
    return false;
  }
}

export async function captureXCookies({ challengeId, challengeToken, importUrl, appOrigin }) {
  const payload = await sendBridgeRequest(
    'CAPTURE_AND_IMPORT',
    { challengeId, challengeToken, importUrl, appOrigin },
    10_000,
  );
  if (!payload || typeof payload !== 'object') {
    throw new Error('Bridge returned an invalid import response.');
  }
  return payload;
}
