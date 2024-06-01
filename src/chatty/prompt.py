from typing import Any, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
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


def create_prompt_with_history(model: BaseLanguageModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_INPUT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model

    return RunnableWithMessageHistory(chain, get_session_history)


def create_prompt_with_retriever(model: BaseLanguageModel, retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", MESSAGE_TEMPLATE),
        ]
    )

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

    return chain


def create_prompt_with_self_retriever(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
):
    retriever =  SelfQueryRetriever.from_llm(
        model,
        vectorstore,
        document_content_description,
        metadata_field_info,
        search_type="similarity",
        search_kwargs={"k": 1},
    )

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
    runnable: Union[RunnableWithMessageHistory, VectorStoreRetriever, SelfQueryRetriever],
) -> Any:
    print(message)
    return runnable.invoke(
        message,
        config=config,
    )
