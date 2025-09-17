from openai import OpenAI

apik = "YOUR_API_KEY"
# 调用 GPT-3.5 模型  
client = OpenAI(api_key=apik)
def ask_gpt3(client, messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        timeout = 60,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content

def ask_gpt4(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        timeout = 60,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content

messages = [{"role": "user", "content": "hello"}]

print(ask_gpt3(client, messages))