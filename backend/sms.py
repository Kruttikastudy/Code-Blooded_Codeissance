from twilio.rest import Client

account_sid = "ACdb7e9b87e86f62b798cb969d05cd5446"
auth_token = "ece02d58a74f25c92e1d92290a6ff9b0"

client = Client(account_sid, auth_token)

def send_whatsapp_sos(location=None, custom_message=None):
    base_message = "🚨 SOS ALERT! Your friend needs help immediately."
    if location:
        base_message += f"\n📍 Location: {location}"
    if custom_message:
        base_message += f"\n📝 Note: {custom_message}"

    try:
        message = client.messages.create(
            from_="whatsapp:+14155238886",  # Twilio sandbox WhatsApp number
            body=base_message,
            to="whatsapp:+919326370332"    # User’s WhatsApp number
        )
        print(f"Your friend needs help.Kindly check on her.")
    except Exception as e:
        print(f"Error sending WhatsApp SOS: {e}")

# Example
send_whatsapp_sos(location="https://maps.google.com/?q=12.9716,77.5946", custom_message="Feeling unsafe.")