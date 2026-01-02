from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.7)

response = chat_model.invoke("Hello, how are you?")
print(response)
