import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from deepeval.models import DeepEvalBaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

model = AzureChatOpenAI(
    temperature=0,
    model=os.getenv("LLM_MODEL_NAME"),
    openai_api_type="azure",
    openai_api_version=os.getenv("LLM_API_VERSION"),
    azure_endpoint=os.getenv("LLM_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


def main():

    systemPrompt = """You are an adeptus mechanicus knowledge seeker.
    You work in pare with another adept. You both should find and structure data from sources.
    You task is to decompose a broad topic into subtopics to assist your colleague, who will seek data for each subtopic from your response.
    You will receive a broad topic.
    You should decompose it in subtopics. And then decompose each subtopic further. Until subtopic is covering a small, precise amount of knowledge. Or a limit of depth is reached.
    Try to find as much subtopics for each parent node as possible.
    Use provided list of sources for data search.
    Your response should be presented as a graph using map.
    Each node should contain fields:
    1. topic - a subtopic name;
    2. parent - parent subtopic name;
    3. subtopics - list of subtopics for current node;
    4. level of depth
    5. keywords - list of keywords related to subtopic to help your colleague in knowledge retrieval.
    The format of response is Json."""

    input = """Topic: Java collections.
    Level of depth: 4"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", systemPrompt),
            ("user", "{input}"),
        ]
    )

    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    print(chain.invoke(input))


if __name__ == "__main__":
    main()
