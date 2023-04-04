import os

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model_name="gpt-4")

multiple_input_prompt = PromptTemplate(
    input_variables=["user_query", "context"], 
    template="""Distill the key ideas, insights, or principles from the following context related to the user's query.
    User query: {user_query}
    
    Context: {context}
    """
)

