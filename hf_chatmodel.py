from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    provider="auto"
)
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("Tell me about the burewala city.")
print(response.content)
