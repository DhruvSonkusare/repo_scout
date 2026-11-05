import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
for m in client.models.list():
    actions = m.supported_actions or []
    if "generateContent" in actions or "embedContent" in actions:
        print(m.name, "->", ", ".join(a for a in actions if a in ("generateContent", "embedContent")))