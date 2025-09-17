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
        model='llama3.1',
        messages=messages,
        stream=True,
        )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans
ans = ""
print(chat(messages2))
print("1111111111111111111111111111")
print(chat(messages2))
#ans = ans + chat(messages1)+'\n'+chat(messages2)
#print(ans)
# messages = []  

# # List of questions  
# questions = [  
#     'Why is the sky blue? Your answer should be simple.',  
#     'What causes rainbows?',  
#     'How does photosynthesis work?',  
#     'Why do leaves change color in the fall?'  
# ]  
# stream = ollama.embed(model = 'llama3.1', input = questions)
# ans = ""
# for chunk in stream:  
#     # Check if the chunk has the expected structure  
#     if isinstance(chunk, dict) and 'message' in chunk:  
#         ans += chunk['message']['content']  # Access the content  
#     else:  
#         print("Unexpected chunk structure:", chunk)  
# cnt = 1
# # Iterate through each question  
# for question in questions:  
#     # Add the current user question to the messages  
#     messages.append({  
#         'role': 'user',  
#         'content': question,  
#     })  
    
#     # Call the chat model with the entire conversation so far  
#     response = ollama.chat(model='llama3.1', messages=messages)  
    
#     # Append the model's response to the messages to maintain context  
#     messages.append({  
#         'role': 'assistant',  
#         'content': response['message']['content'],  
#     })  
    
#     # Print the model's response  
#     print(f"No.{cnt}",response['message']['content'])
#     cnt = cnt+1