import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

def test_langsmith():
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # safe, supported, cheap
        temperature=0
    )

    response = llm.invoke(
        [HumanMessage(content="Say hello in one short sentence.")]
    )

    print(f"LLM response: {response.content}")

if __name__ == "__main__":
    test_langsmith()
