from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2:1b")  # you can swap with "mistral", "gemma", etc.

response = llm.invoke("Explain photosynthesis like I'm 10 years old.")
print(response)
