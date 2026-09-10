"""Owner opt-in channels only. Never include secrets, raw exceptions or tax data."""
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from urllib.request import Request, urlopen


def send(message):
    channel = os.getenv("PUBLIC_REVIEW_ALERT_CHANNEL", "none").lower()
    if channel == "none":
        return False
    if channel == "telegram":
        token, chat = os.environ["PUBLIC_REVIEW_TELEGRAM_TOKEN"], os.environ["PUBLIC_REVIEW_TELEGRAM_CHAT_ID"]
        request = Request("https://api.telegram.org/bot" + token + "/sendMessage",
                          data=json.dumps({"chat_id": chat, "text": message}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=15) as response:
            if not json.loads(response.read()).get("ok"):
                raise RuntimeError("ALERT_DELIVERY_FAILED")
    elif channel == "email":
        mail = EmailMessage()
        mail["Subject"] = "Public portfolio: review update"
        mail["From"] = os.environ["PUBLIC_REVIEW_EMAIL_FROM"]
        mail["To"] = os.environ["PUBLIC_REVIEW_EMAIL_TO"]
        mail.set_content(message)
        with smtplib.SMTP_SSL(os.environ["PUBLIC_REVIEW_SMTP_HOST"],
                             int(os.getenv("PUBLIC_REVIEW_SMTP_PORT", "465")),
                             context=ssl.create_default_context(), timeout=15) as smtp:
            smtp.login(os.environ["PUBLIC_REVIEW_SMTP_USER"], os.environ["PUBLIC_REVIEW_SMTP_PASSWORD"])
            smtp.send_message(mail)
    else:
        raise ValueError("UNKNOWN_ALERT_CHANNEL")
    return True
