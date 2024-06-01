from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

from prompt_toolkit import prompt

from chatty.data import DOCS
from chatty.prompt import (
    create_prompt_with_history,
    create_prompt_with_retriever,
    get_session_history,
    say_something,
)
from chatty.store import ChattyVectorStore

load_dotenv()

CHAT_MAX_SIZE = 20

if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-3.5-turbo")
    docs = DOCS["movies"]
    store = ChattyVectorStore(docs)
    chain = create_prompt_with_history(model)
    runnable_with_history = RunnableWithMessageHistory(chain, get_session_history)
    runnable_with_retriever = create_prompt_with_retriever(model, store.retriever)
    config = {"configurable": {"session_id": "abc"}}

    msg_count = 0
    msg = ""

    while msg_count <= CHAT_MAX_SIZE and msg.lower() != "bye":
        msg_count += 1
        msg = prompt(">> : ")
        response = say_something(msg, config, runnable_with_retriever)
        print(response.content)
