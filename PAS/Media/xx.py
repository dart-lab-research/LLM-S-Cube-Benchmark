import ollama

# response = ollama.chat(
#     model='llama3.1',
#     messages=[{
#         'role': 'user',
#         'content': 'What is 1+1?'
#     }]
# )
messages1 = [{"role": "user", "content": "Hello ollama! My name is YCY, remember that!"}]
messages2 = [{"role": "user", "content": "What is my name"}]
def chat(messages):
        
    stream = ollama.chat(
        model='llama3.1:8b',
        messages=messages,
        stream=True,
        )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans
ans = ""
questions = ["I am jack. Remember my name.", "Tell me what is my name?", "1+1 = ?", "3*3 = ?"]  
messages = []
for question in questions:  
        # 将用户问题添加到对话历史  
    messages.append({"role": "user", "content": question})  

    # 调用 chat 函数获取模型的回答  
    response = chat(messages)  

    # 将模型的回答添加到对话历史  
    messages.append({"role": "assistant", "content": response})  

    # 打印问题和回答  
    print(f"Question: {question}")  
    print(f"Model Response: {response}")  
    print("-" * 50)  # 分隔线  

