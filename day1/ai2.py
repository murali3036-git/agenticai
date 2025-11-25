from openai import OpenAI
import logging
log = logging.getLogger(OpenAI.__module__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
client = OpenAI(base_url='http://localhost:11434/v1/',api_key='ollama') # Placeholder API key for Ollama)

chat_completion = client.chat.completions.create(messages=[{'role': 'user','content': 'Explain the concept of quantum entanglement.'}],model='llama3')                                              # Use the name of the model pulled with Ollama)

print(chat_completion.choices[0].message.content)
