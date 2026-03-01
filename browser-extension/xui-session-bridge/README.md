# XUI Session Bridge (Chrome/Edge)

This extension lets the Themes UI import X/Twitter cookies from the same browser profile where the app is open.

## Install (Unpacked)

1. Open `chrome://extensions` (or `edge://extensions`).
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select this folder: `browser-extension/xui-session-bridge`.
5. Refresh the Themes page.

## Usage

1. In the same browser profile, sign in to `https://x.com`.
2. Open Stock Scanner -> Themes -> Manage Sources.
3. Click **Connect From Current Browser** under **Twitter/X Session**.
4. If successful, status changes to `authenticated`.

## Troubleshooting

- `Bridge not detected`: extension not loaded on current page origin.
- `No usable cookies`: you are not signed in to X/Twitter in this browser profile.
- `Origin not allowed`: update `XUI_BRIDGE_ALLOWED_ORIGINS` on backend.
- `blocked_challenge` after import: X session requires additional verification on X.

