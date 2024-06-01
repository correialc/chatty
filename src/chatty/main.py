from typing import Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

from prompt_toolkit import prompt

load_dotenv()
store = {}

CHAT_MAX_SIZE = 20
BASE_INPUT = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
)


def create_prompt_template(model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_INPUT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def say_something(
    message: str,
    config: dict[str, dict[str, str]],
    runnable: RunnableWithMessageHistory,
) -> Any:
    print(message)
    return runnable.invoke(
        [HumanMessage(content=message)],
        config=config,
    )


if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = create_prompt_template(model)
    runnable_with_history = RunnableWithMessageHistory(chain, get_session_history)
    config = {"configurable": {"session_id": "abc"}}

    msg_count = 0
    msg = ""

    while msg_count <= CHAT_MAX_SIZE and msg.lower() != "bye":
        msg_count += 1
        msg = prompt(">> : ")
        response = say_something(msg, config, runnable_with_history)
        print(response.content)
