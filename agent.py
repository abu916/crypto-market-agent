import os
import sys
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from groq import Groq

# Force UTF-8 output so emojis print correctly on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables
load_dotenv()

# API keys
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
GMAIL_SENDER      = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
GMAIL_RECIPIENT   = os.getenv("GMAIL_RECIPIENT")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_crypto_news():
    """Fetches the latest crypto news from Finnhub."""
    url = f"https://finnhub.io/api/v1/news?category=crypto&token={FINNHUB_API_KEY}"
    print("📡 Fetching news from Finnhub...")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            news_items = response.json()[:15]
            formatted = []
            for item in news_items:
                headline = item.get("headline", "")
                summary  = item.get("summary", "")
                source   = item.get("source", "")
                if headline or summary:
                    formatted.append(f"Source: {source}\nHeadline: {headline}\nSummary: {summary}\n")
            return "\n".join(formatted)
        else:
            print(f"  Error {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"  Exception fetching news: {e}")
        return ""


def fetch_fear_greed_index():
    """Fetches the current Crypto Fear & Greed Index (free, no key needed)."""
    print("😨 Fetching Fear & Greed Index...")
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()["data"][0]
        return {
            "value": data["value"],
            "label": data["value_classification"],
        }
    except Exception as e:
        print(f"  Could not fetch Fear & Greed: {e}")
        return {"value": "N/A", "label": "Unknown"}


def fetch_crypto_prices():
    """Fetches live BTC and ETH prices from CoinGecko (free, no key needed)."""
    print("💰 Fetching live crypto prices from CoinGecko...")
    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd"
            "&include_24hr_change=true&include_market_cap=true"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        btc = data.get("bitcoin", {})
        eth = data.get("ethereum", {})
        return {
            "BTC": {
                "price":  f"${btc.get('usd', 'N/A'):,}",
                "change": f"{btc.get('usd_24h_change', 0):.2f}%",
                "mcap":   f"${btc.get('usd_market_cap', 0)/1e9:.1f}B",
                "up":     btc.get('usd_24h_change', 0) >= 0,
            },
            "ETH": {
                "price":  f"${eth.get('usd', 'N/A'):,}",
                "change": f"{eth.get('usd_24h_change', 0):.2f}%",
                "mcap":   f"${eth.get('usd_market_cap', 0)/1e9:.1f}B",
                "up":     eth.get('usd_24h_change', 0) >= 0,
            },
        }
    except Exception as e:
        print(f"  Could not fetch prices: {e}")
        return {}


# ─────────────────────────────────────────────
# AI ANALYSIS
# ─────────────────────────────────────────────

def analyze_news(news_content):
    """Analyzes news using Groq AI and returns a sorted JSON list."""
    print("🤖 Analyzing news with Groq AI (Llama 3.3 70B)...")

    system_prompt = """You are a crypto market intelligence AI agent.

Analyze the provided news and identify only HIGH-IMPACT events that affect cryptocurrency markets.

Focus on:
- Statements by Elon Musk, Donald Trump, Fed/ECB officials, SEC, BlackRock, IMF
- ETF approvals/rejections, regulatory bans or greenlight
- Whale movements, exchange inflows/outflows
- Macroeconomic data: CPI, interest rates, recession signals
- Geopolitical shocks: wars, sanctions

For EACH event return:
1. event: short title
2. summary: 1-2 sentence summary
3. sentiment: Bullish / Bearish / Neutral
4. impact: High / Medium / Low
5. confidence: number 0-100 (no % sign)
6. affected_assets: comma-separated e.g. BTC, ETH
7. reasoning: why it matters for crypto

IGNORE minor price updates, low-quality blogs, and duplicate news.
Sort results: High impact first, then Medium, then Low.

Return ONLY a valid JSON array, no markdown, no extra text."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Analyze this news:\n\n{news_content}"},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        result = completion.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        parsed = json.loads(result.strip())
        # Sort: High first, then Medium, then Low
        order = {"High": 0, "Medium": 1, "Low": 2}
        parsed.sort(key=lambda x: order.get(x.get("impact", "Low"), 3))
        return parsed
    except Exception as e:
        print(f"  AI analysis error: {e}")
        return []


# ─────────────────────────────────────────────
# EMAIL (HTML)
# ─────────────────────────────────────────────

SENTIMENT_COLORS = {
    "Bullish": {"bg": "#0d3320", "border": "#00c853", "badge": "#00c853", "text": "#a5d6a7"},
    "Bearish": {"bg": "#3b0d0d", "border": "#f44336", "badge": "#f44336", "text": "#ef9a9a"},
    "Neutral": {"bg": "#1a1a2e", "border": "#9e9e9e", "badge": "#9e9e9e", "text": "#e0e0e0"},
}

IMPACT_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

def build_html_email(events, fear_greed, prices):
    """Builds a beautiful dark-theme HTML email."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%B %d, %Y  %H:%M UTC")

    fg_value = fear_greed.get("value", "N/A")
    fg_label = fear_greed.get("label", "Unknown")

    # Fear & Greed color
    try:
        fgi = int(fg_value)
        if fgi <= 25:   fg_color = "#f44336"
        elif fgi <= 45: fg_color = "#ff9800"
        elif fgi <= 55: fg_color = "#9e9e9e"
        elif fgi <= 75: fg_color = "#8bc34a"
        else:           fg_color = "#00c853"
    except Exception:
        fg_color = "#9e9e9e"

    # Price cards
    price_cards_html = ""
    for coin, info in prices.items():
        arrow = "▲" if info["up"] else "▼"
        chg_color = "#00c853" if info["up"] else "#f44336"
        price_cards_html += f"""
        <td style="padding:0 8px;">
          <div style="background:#1e1e2e;border:1px solid #333;border-radius:12px;
                      padding:16px 22px;text-align:center;min-width:140px;">
            <div style="color:#888;font-size:12px;letter-spacing:1px;
                        text-transform:uppercase;">{coin}</div>
            <div style="color:#fff;font-size:22px;font-weight:700;
                        margin:6px 0;">{info['price']}</div>
            <div style="color:{chg_color};font-size:13px;">{arrow} {info['change']} 24h</div>
            <div style="color:#555;font-size:11px;margin-top:4px;">MCap {info['mcap']}</div>
          </div>
        </td>"""

    # Event cards
    event_cards_html = ""
    for i, ev in enumerate(events, 1):
        sentiment = ev.get("sentiment", "Neutral")
        c = SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["Neutral"])
        impact = ev.get("impact", "Low")
        imp_emoji = IMPACT_EMOJI.get(impact, "⚪")
        conf = ev.get("confidence", "N/A")
        assets = ev.get("affected_assets", "N/A")

        event_cards_html += f"""
        <div style="background:{c['bg']};border:1px solid {c['border']};
                    border-radius:14px;padding:20px 24px;margin-bottom:16px;">

          <div style="display:flex;justify-content:space-between;
                      align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
            <span style="background:{c['badge']};color:#000;font-size:11px;
                         font-weight:700;padding:3px 10px;border-radius:20px;
                         letter-spacing:0.5px;">{sentiment.upper()}</span>
            <span style="color:#aaa;font-size:12px;">
              {imp_emoji} {impact} Impact &nbsp;|&nbsp; Confidence: <b style="color:#fff;">{conf}%</b>
              &nbsp;|&nbsp; Assets: <b style="color:{c['border']};">{assets}</b>
            </span>
          </div>

          <div style="color:#fff;font-size:16px;font-weight:700;margin-bottom:8px;">
            {i}. {ev.get('event', 'N/A')}
          </div>
          <div style="color:{c['text']};font-size:14px;line-height:1.6;margin-bottom:10px;">
            {ev.get('summary', 'N/A')}
          </div>
          <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:10px 14px;">
            <span style="color:#888;font-size:12px;text-transform:uppercase;
                          letter-spacing:0.5px;">Why it matters: </span>
            <span style="color:#ccc;font-size:13px;">{ev.get('reasoning', 'N/A')}</span>
          </div>
        </div>"""

    high_count   = sum(1 for e in events if e.get("impact") == "High")
    bull_count   = sum(1 for e in events if e.get("sentiment") == "Bullish")
    bear_count   = sum(1 for e in events if e.get("sentiment") == "Bearish")
    bias = "🟢 BULLISH BIAS" if bull_count > bear_count else ("🔴 BEARISH BIAS" if bear_count > bull_count else "⚪ NEUTRAL BIAS")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#0a0a0f;font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:680px;margin:0 auto;padding:24px 16px;">

  <!-- HEADER -->
  <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
              border:1px solid #2a2a4a;border-radius:18px;
              padding:28px 30px;margin-bottom:20px;text-align:center;">
    <div style="font-size:28px;margin-bottom:6px;">📊</div>
    <h1 style="margin:0;color:#fff;font-size:22px;font-weight:800;
               letter-spacing:-0.5px;">Crypto Market Intelligence</h1>
    <p style="margin:6px 0 0;color:#666;font-size:13px;">{now}</p>
    <div style="margin-top:14px;display:inline-block;background:rgba(255,255,255,0.05);
                border-radius:10px;padding:6px 18px;font-size:13px;color:#aaa;">
      {len(events)} signals detected &nbsp;·&nbsp;
      {high_count} high impact &nbsp;·&nbsp; {bias}
    </div>
  </div>

  <!-- PRICE TICKER -->
  {'<table style="width:100%;border-collapse:separate;border-spacing:0;margin-bottom:20px;"><tr>' + price_cards_html + '</tr></table>' if prices else ''}

  <!-- FEAR & GREED -->
  <div style="background:#12121f;border:1px solid #2a2a4a;border-radius:14px;
              padding:16px 24px;margin-bottom:20px;display:flex;align-items:center;
              justify-content:space-between;flex-wrap:wrap;gap:10px;">
    <div>
      <div style="color:#888;font-size:11px;text-transform:uppercase;
                  letter-spacing:1px;">Fear & Greed Index</div>
      <div style="font-size:32px;font-weight:800;color:{fg_color};margin-top:4px;">
        {fg_value}
      </div>
    </div>
    <div style="background:{fg_color}22;border:1px solid {fg_color};border-radius:10px;
                padding:8px 18px;color:{fg_color};font-weight:700;font-size:15px;">
      {fg_label}
    </div>
  </div>

  <!-- EVENT CARDS -->
  <div style="margin-bottom:20px;">
    <h2 style="color:#fff;font-size:15px;font-weight:700;margin:0 0 14px;
               text-transform:uppercase;letter-spacing:1px;">
      📰 Market Signals
    </h2>
    {event_cards_html}
  </div>

  <!-- FOOTER -->
  <div style="text-align:center;color:#444;font-size:11px;padding:10px 0;">
    Generated by Crypto Market Intelligence Agent &nbsp;·&nbsp;
    Powered by Finnhub + Groq AI (Llama 3.3 70B)<br>
    <span style="color:#ff5722;">⚠ Not financial advice. Do your own research.</span>
  </div>

</div>
</body></html>"""
    return html


def send_email(events, fear_greed, prices):
    """Sends the HTML intelligence report via Gmail."""
    if not all([GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECIPIENT]):
        print("\n  Skipping email: Missing Gmail credentials in .env")
        return

    print("\n📧 Sending HTML email report...")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%b %d %H:%M UTC")

    msg = MIMEMultipart("alternative")
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = GMAIL_RECIPIENT
    msg["Subject"] = f"🔔 Crypto Intelligence Report — {now}"

    # Plain-text fallback
    plain_lines = [f"Crypto Market Intelligence Report — {now}\n"]
    for i, ev in enumerate(events, 1):
        plain_lines.append(
            f"Event {i}: {ev.get('event')}\n"
            f"Sentiment: {ev.get('sentiment')} | Impact: {ev.get('impact')} | "
            f"Confidence: {ev.get('confidence')}%\n"
            f"Summary: {ev.get('summary')}\n"
            f"Assets: {ev.get('affected_assets')}\n"
            f"Why: {ev.get('reasoning')}\n"
            + "-"*50
        )
    plain_body = "\n".join(plain_lines)

    html_body = build_html_email(events, fear_greed, prices)

    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"  ✅ HTML email sent to {GMAIL_RECIPIENT}")
    except Exception as e:
        print(f"  ❌ Email failed: {e}")


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(events):
    """Sends a concise Telegram alert for HIGH-impact events only."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n  Skipping Telegram: No bot token/chat ID in .env")
        return

    high_impact = [e for e in events if e.get("impact") == "High"]
    if not high_impact:
        print("\n  No High-impact events. Skipping Telegram alert.")
        return

    print(f"\n📱 Sending {len(high_impact)} High-impact alert(s) to Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for ev in high_impact:
        sentiment = ev.get("sentiment", "Neutral")
        emoji = "🟢" if sentiment == "Bullish" else ("🔴" if sentiment == "Bearish" else "⚪")
        text = (
            f"🔔 *CRYPTO ALERT — {sentiment.upper()}* {emoji}\n\n"
            f"*{ev.get('event', 'N/A')}*\n\n"
            f"_{ev.get('summary', 'N/A')}_\n\n"
            f"📊 Impact: *{ev.get('impact')}*  |  Confidence: *{ev.get('confidence')}%*\n"
            f"💎 Assets: `{ev.get('affected_assets', 'N/A')}`\n\n"
            f"💡 *Why:* {ev.get('reasoning', 'N/A')}\n\n"
            f"⚠️ _Not financial advice._"
        )
        try:
            r = requests.post(url, json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       text,
                "parse_mode": "Markdown",
            }, timeout=10)
            if r.status_code == 200:
                print(f"  ✅ Telegram alert sent: {ev.get('event')[:50]}...")
            else:
                print(f"  ❌ Telegram error: {r.text}")
        except Exception as e:
            print(f"  ❌ Telegram exception: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not FINNHUB_API_KEY or not GROQ_API_KEY:
        print("❌ Error: Missing FINNHUB_API_KEY or GROQ_API_KEY in .env")
        return

    print("\n" + "="*55)
    print("   🧠  CRYPTO MARKET INTELLIGENCE AGENT")
    print("="*55 + "\n")

    # 1. Fetch all data in parallel-ish
    news_content  = fetch_crypto_news()
    fear_greed    = fetch_fear_greed_index()
    prices        = fetch_crypto_prices()

    if not news_content:
        print("❌ No news fetched. Exiting.")
        return

    # 2. AI Analysis
    events = analyze_news(news_content)

    if not events:
        print("❌ No events extracted. Exiting.")
        return

    # 3. Print summary to console
    print(f"\n✅ Analysis complete — {len(events)} events found\n")
    print(f"   Fear & Greed: {fear_greed['value']} ({fear_greed['label']})")
    if prices:
        btc = prices.get("BTC", {})
        eth = prices.get("ETH", {})
        print(f"   BTC: {btc.get('price')} ({btc.get('change')})")
        print(f"   ETH: {eth.get('price')} ({eth.get('change')})")
    print()
    for ev in events:
        sentiment = ev.get("sentiment", "Neutral")
        emoji = "🟢" if sentiment == "Bullish" else ("🔴" if sentiment == "Bearish" else "⚪")
        print(f"  {emoji} [{ev.get('impact'):6}] {ev.get('event')}")

    # 4. Send HTML Email
    send_email(events, fear_greed, prices)

    # 5. Send Telegram alerts (High impact only)
    send_telegram(events)

    print("\n✅ Agent run complete.\n")


if __name__ == "__main__":
    main()
