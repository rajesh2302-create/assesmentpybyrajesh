import os
import hmac
import hashlib
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Load secrets from .env file (Nothing hardcoded)
load_dotenv()
API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# 2. Create requests.Session with Retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# 3. GET users through session & print first 5 users
url = "https://jsonplaceholder.typicode.com/users"
response = session.get(url, headers={"Authorization": f"Bearer {API_KEY}"})
users = response.json()

for user in users[:5]:
    print(f"{user.get('id')} - {user.get('name')} - {user.get('email')}")

# 4. Extract nested fields of the first user using .get() only
first_user = users[0] if users else {}
address = first_user.get("address", {})
geo = address.get("geo", {})
company = first_user.get("company", {})

city = address.get("city", "Not Available")
latitude = geo.get("lat", "Not Available")
# Using 'zip' key as per requirement (JSON has 'zipcode', so 'zip' safely defaults to Not Available)
zip_code = address.get("zip", "Not Available")
company_name = company.get("name", "Not Available")

print(f"City: {city} | Latitude: {latitude} | Zip: {zip_code} | Company: {company_name}")

# 5. HMAC-SHA256 Webhook Signature Verification
def verify_signature(payload: str, signature: str, secret: str) -> bool:
    if not secret:
        return False
    expected_sig = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_sig, signature)

# Testing Valid, Tampered, and Wrong-Secret cases
sample_payload = '{"event": "payment.success", "amount": 5000}'
valid_signature = hmac.new(
    WEBHOOK_SECRET.encode("utf-8"),
    sample_payload.encode("utf-8"),
    hashlib.sha256
).hexdigest()

is_valid = verify_signature(sample_payload, valid_signature, WEBHOOK_SECRET)
is_tampered = verify_signature('{"event": "payment.success", "amount": 99999}', valid_signature, WEBHOOK_SECRET)
is_wrong_secret = verify_signature(sample_payload, valid_signature, "wrong_secret_key")

print(f"Valid: {is_valid} | Tampered: {is_tampered} | Wrong secret: {is_wrong_secret}")