from typing import Any, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import ChatMessageHistory

history = {}

BASE_INPUT = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
)

MESSAGE_TEMPLATE = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
"""


def create_prompt_with_history(model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_INPUT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model


def create_prompt_with_retriever(model, retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", MESSAGE_TEMPLATE),
        ]
    )

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

    return chain


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history:
        history[session_id] = ChatMessageHistory()
    return history[session_id]


def say_something(
    message: str,
    config: dict[str, dict[str, str]],
    runnable: Union[RunnableWithMessageHistory, VectorStoreRetriever],
) -> Any:
    print(message)
    return runnable.invoke(
        message,
        config=config,
    )
