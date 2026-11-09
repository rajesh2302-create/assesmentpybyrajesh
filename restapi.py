import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()
app.add_middleware=(CORSMiddleware,
                    allow_origins=["*"],
                    allow_credential=True,
                    allow_methods=["*"],
                    allow_headers=["*"])

def explain_status(code):
    status_message={200:"ok - request succeesfull",
                    201:"created - resource created succeesfully",
                    400:"bad request - client error",
                    401:"unauthorizeed - authentication required",
                    404:"not found - resource does not exist",
                    500:"internal server error - server side problem"
                    }
    return status_message.get(code,"unknown status")
BASE_URL="https://jsonplaceholder.typicode.com"
response_users=requests.get(f"{BASE_URL}/users")
users=response_users.json()

for user in users[:5]:
    print(f"{user['id']}-{user['name']}-{user['email']}")

print("-" * 40)
@app.post("/posts")
def create_post():
 new_post ={"title":"data engineering task",
           "body":"learning rest api with python",
           "user id":1}
response_post =requests.post(f"{BASE_URL}/posts",json=new_post)
post_data=response_post.json()
print(f"post:{response_post.status_code},id:{post_data['id']}")
@app.put("/post/{post_id}")
def update_post(post_id: int):
 put_data={"id":1,"title":"updated title","body":"updated body","user id":1}
 response_put =requests.put(f"{BASE_URL}/posts/1",json=put_data)
def 
 patch_data={"title":"partially updated title"}

response_patch=requests.patch(f"{BASE_URL}/posts/1",json=patch_data)
response_delete=requests.delete(f"{BASE_URL}/posts/1")
print(f"PUT:{response_put.status_code}|PATCH: {response_patch.status_code}|DELETE:{response_delete.status_code}")
response_not_found=requests.get(f"{BASE_URL}/posts/9999")

if response_not_found.status_code==404:
    friendly_msg=explain_status(response_not_found.status_code)
    print(f"GET/posts/999->{response_not_found.status_code}{friendly_msg}")



 