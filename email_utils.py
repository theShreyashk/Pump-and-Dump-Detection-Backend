import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_alert_email(receiver_email: str, stock: str, probability: int, risk: str, reason: str):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:
        print(f"Skipping email alert to {receiver_email} for {stock}. Credentials not configured.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"PUMP ALERT: High anomaly detected for {stock}"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    text = f"""
    Pump & Dump Alert!
    
    Stock: {stock}
    Pump Probability: {probability}%
    Dump Risk: {risk}
    
    Reason: {reason}
    
    Stay safe and do your own research before trading.
    """

    part1 = MIMEText(text, "plain")
    msg.attach(part1)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Successfully sent alert email for {stock} to {receiver_email}")
    except Exception as e:
        print(f"Failed to send email to {receiver_email}: {e}")
