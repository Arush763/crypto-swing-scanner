# Live System Setup (Free Hosting)

## Architecture
- **GitHub Actions** — runs the scanner daily at midnight UTC (free, no server needed)
- **Telegram Bot** — sends alerts when signals fire
- **GitHub Pages** — live dashboard at `https://YOUR_USERNAME.github.io/YOUR_REPO`

---

## Step 1 — Push to GitHub

```bash
cd "Crypto Bot"
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/crypto-scanner.git
git push -u origin main
```

---

## Step 2 — Create a Telegram Bot

1. Open Telegram → search **@BotFather** → `/newbot`
2. Name it anything (e.g. `My Crypto Scanner`)
3. Copy the **Bot Token** (looks like `123456789:ABCdef...`)
4. Start a chat with your new bot, or add it to a channel
5. Get your **Chat ID**:
   - For personal chat: message `@userinfobot` — it replies with your ID
   - For a channel: add the bot as admin, then use `https://api.telegram.org/botTOKEN/getUpdates`

---

## Step 3 — Add Secrets to GitHub

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

Add two secrets:
| Name | Value |
|------|-------|
| `TELEGRAM_BOT_TOKEN` | Your bot token from Step 2 |
| `TELEGRAM_CHAT_ID` | Your chat/channel ID from Step 2 |

---

## Step 4 — Enable GitHub Pages

Go to your repo → **Settings → Pages**
- Source: `Deploy from a branch`
- Branch: `main` → `/dashboard`
- Click Save

Your dashboard will be live at:
`https://YOUR_USERNAME.github.io/YOUR_REPO`

---

## Step 5 — Trigger First Scan

Go to your repo → **Actions → Crypto Scanner → Run workflow**

This runs the scanner immediately, sends Telegram alerts, and updates the dashboard.

After that it runs automatically every day at midnight UTC.

---

## Changing the Scan Schedule

Edit `.github/workflows/scanner.yml`:
```yaml
- cron: "0 0 * * *"   # midnight UTC daily
- cron: "0 8 * * *"   # 8am UTC daily
- cron: "0 0 * * 1"   # every Monday
```

---

## Running Locally

```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python scripts/live_scan.py
```
