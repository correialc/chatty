from typing import Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def say_something(message: str, config: dict[str, dict[str, str]]) -> Any:
    print(message)
    return with_message_history.invoke(
            [HumanMessage(content=message)],
            config=config,
        )    

if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    with_message_history = RunnableWithMessageHistory(model, get_session_history)
    
    config = {"configurable": {"session_id": "abc"}}
    response = say_something("My name is Leandro", config)
    print(response.content)
    response = say_something("Do you know my name?", config)
    print(response.content)
    