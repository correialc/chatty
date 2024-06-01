from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from prompt_toolkit import prompt

from chatty.data import MoviesDataSample
from chatty.prompt import (
    create_prompt_with_history,
    create_prompt_with_retriever,
    create_prompt_with_self_retriever,
    say_something,
)
from chatty.store import ChattyVectorStore

load_dotenv()

CHAT_MAX_SIZE = 20

if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-3.5-turbo")
    sample = MoviesDataSample()
    store = ChattyVectorStore(sample.docs)
    runnable_with_history = create_prompt_with_history(model)
    runnable_with_retriever = create_prompt_with_retriever(model, store.retriever)
    runnable_with_self_retriever = create_prompt_with_self_retriever(
        model,
        store.vectorstore,
        sample.document_content_description,
        sample.metadata_field_info,
    )
    config = {"configurable": {"session_id": "abc"}}

    msg_count = 0
    msg = ""

    while msg_count <= CHAT_MAX_SIZE and msg.lower() != "bye":
        msg_count += 1
        msg = prompt(">> : ")
        response = say_something(msg, config, runnable_with_self_retriever)
        print(response.content)
