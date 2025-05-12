import requests

def send_telegram_alert(message):
    bot_token = "7260593253:AAHc14JfN4mlasfW2v4r5ZljWgcGXwC83QU"
    chat_id = "7445572516"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Telegram alert error: {e}")
        return False
