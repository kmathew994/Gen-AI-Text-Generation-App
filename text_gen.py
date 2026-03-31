import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()


client = ChatCompletionsClient(
    endpoint=os.getenv("endpoint"),
    credential=AzureKeyCredential(os.getenv("api_key")),
    api_version="2024-05-01-preview"
)
model_name = "grok-4-1-fast-reasoning"

response = client.complete(
    messages=[
        SystemMessage(content="You are an AI assistant that provides insightful and inspiring motivational quotes. Your goal is to uplift and encourage users. You should always be positive and thoughtful. Do not generate quotes that are generic or cliché."),
        UserMessage(content="I'm an artist feeling discouraged with my work. Give me a powerful quote about perseverance for the creative field"),
    ],
    model=model_name
    )

print(response.choices[0].message.content)
