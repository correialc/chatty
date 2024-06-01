from typing import Sequence
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class ChattyVectorStore:
    def __init__(self, docs: Sequence[Document]) -> None:
        self.vectorstore = ChattyVectorStore.build_vectorstore_from_docs(docs)

    @staticmethod
    def build_vectorstore_from_docs(docs: Sequence[Document]):
        return Chroma.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
        )

    @property
    def retriever(self):
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        )
