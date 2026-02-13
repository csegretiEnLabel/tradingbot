import os
import anthropic
from dotenv import load_dotenv

# Force load .env
load_dotenv(override=True)

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key loaded: {api_key if api_key else 'None'}")
if api_key:
    print(f"Masked Key: {api_key[:5]}...{api_key[-4:]}")

if not api_key:
    print("ERROR: No API key found in .env")
    exit(1)

client = anthropic.Anthropic(api_key=api_key)

try:
    print("Sending test request to Claude-3-5-Haiku...")
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=10,
        messages=[
            {"role": "user", "content": "Hello, world!"}
        ]
    )
    print("Success!")
    print(message.content[0].text)
except anthropic.APIStatusError as e:
    print(f"\nAPI Error: {e.status_code}")
    print(f"Type: {e.body.get('type')}")
    print(f"Message: {e.message}")
    if e.status_code == 400 and 'credit balance is too low' in str(e).lower():
        print("\nDIAGNOSIS: Your prepaid credits are exhausted or your payment method failed.")
        print("Please log in to console.anthropic.com -> Billing -> Add Funds.")
except Exception as e:
    print(f"\nUnexpected Error: {e}")
