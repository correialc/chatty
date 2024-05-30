from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-3.5-turbo")
    r = model.invoke([HumanMessage(content="Hi! I'm Leandro")])
    print(r.content)
