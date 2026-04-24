import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from groq import Groq
# Load environment variables
load_dotenv()

# Initialize API clients
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def fetch_crypto_news():
    """Fetches the latest crypto news from Finnhub."""
    url = f"https://finnhub.io/api/v1/news?category=crypto&token={FINNHUB_API_KEY}"
    print(f"Fetching news from Finnhub...")
    response = requests.get(url)
    
    if response.status_code == 200:
        news_items = response.json()
        # Take the top 15 news items to avoid overloading the context window
        recent_news = news_items[:15]
        
        formatted_news = []
        for item in recent_news:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            source = item.get("source", "")
            if headline or summary:
                formatted_news.append(f"Source: {source}\nHeadline: {headline}\nSummary: {summary}\n")
        
        return "\n".join(formatted_news)
    else:
        print(f"Error fetching news: {response.status_code} - {response.text}")
        return ""

def analyze_news(news_content):
    """Analyzes the news content using Groq AI and returns the specified JSON format."""
    print("Analyzing news with Groq AI...")
    
    system_prompt = """You are a crypto market intelligence AI agent.

Your task is to analyze global news, tweets, and macroeconomic developments and identify events that can impact the cryptocurrency market.

Focus ONLY on high-impact signals such as:
- Statements or actions by influential figures (e.g., Elon Musk, Donald Trump, central bank officials)
- Economic data (inflation, interest rates, recession signals)
- Regulatory developments (SEC, government bans, ETF approvals)
- Major financial institutions (BlackRock, IMF, Federal Reserve)
- Large crypto movements (whales, exchange activity)
- Geopolitical events (wars, sanctions, instability)

For each event:
1. Summarize the news in 1-2 sentences
2. Classify sentiment: Bullish / Bearish / Neutral
3. Assign impact level: Low / Medium / High
4. Assign confidence score (0-100%)
5. Explain WHY it affects crypto markets
6. Mention affected assets (BTC, ETH, ALTCOINS, TOTAL MARKET)

Ignore:
- Low-quality blogs
- Repetitive or duplicate news
- Minor price updates without cause

Output format MUST be valid JSON matching this structure exactly:
[
  {
    "event": "",
    "summary": "",
    "sentiment": "",
    "impact": "",
    "confidence": "",
    "affected_assets": "",
    "reasoning": ""
  }
]

Be concise, analytical, and avoid speculation without evidence. Do not output anything outside of the JSON array.
"""

    user_prompt = f"Here is the latest news to analyze:\n\n{news_content}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        
        result = completion.choices[0].message.content
        
        # Try to parse it to ensure it's valid JSON
        try:
            # Strip any markdown formatting like ```json ... ```
            if result.strip().startswith("```json"):
                result = result.strip()[7:-3]
            elif result.strip().startswith("```"):
                result = result.strip()[3:-3]
                
            parsed_json = json.loads(result)
            return json.dumps(parsed_json, indent=2)
        except json.JSONDecodeError:
            print("Warning: Model did not return valid JSON. Raw output:")
            return result
            
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        return ""

def send_email(json_data):
    """Sends the JSON analysis as an email using Gmail SMTP."""
    sender_email = os.getenv("GMAIL_SENDER")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")
    recipient_email = os.getenv("GMAIL_RECIPIENT")

    if not sender_email or not sender_password or not recipient_email:
        print("\nSkipping email notification: Missing Gmail credentials in .env.")
        return

    print("\nSending email report...")
    subject = "Daily Crypto Market Intelligence Report"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    try:
        data = json.loads(json_data)
        formatted_text = "Here is your latest AI-generated market intelligence report:\n\n"
        for i, item in enumerate(data, 1):
            formatted_text += f"Event {i}: {item.get('event', 'N/A')}\n"
            formatted_text += f"Summary: {item.get('summary', 'N/A')}\n"
            formatted_text += f"Sentiment: {item.get('sentiment', 'N/A')}\n"
            formatted_text += f"Impact Level: {item.get('impact', 'N/A')}\n"
            formatted_text += f"Confidence: {item.get('confidence', 'N/A')}%\n"
            formatted_text += f"Affected Assets: {item.get('affected_assets', 'N/A')}\n"
            formatted_text += f"Reasoning: {item.get('reasoning', 'N/A')}\n"
            formatted_text += "-" * 50 + "\n\n"
        body = formatted_text
    except Exception:
        # Fallback to raw data if parsing fails
        body = f"Here is your latest AI-generated market intelligence report:\n\n{json_data}"

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"Email successfully sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    if not FINNHUB_API_KEY or not GROQ_API_KEY:
        print("Error: Missing API keys in .env file.")
        return

    # 1. Fetch News
    news_content = fetch_crypto_news()
    
    if not news_content:
        print("No news fetched. Exiting.")
        return
        
    print(f"Successfully fetched {len(news_content.split('Source: '))} news items.")

    # 2. Analyze News
    analysis_result = analyze_news(news_content)
    
    # 3. Output Result
    print("\n--- AI AGENT ANALYSIS RESULT ---\n")
    print(analysis_result)
    
    # 4. Send Email
    if analysis_result:
        send_email(analysis_result)

if __name__ == "__main__":
    main()
