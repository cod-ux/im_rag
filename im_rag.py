from langchain.embeddings import SentenceTransformerEmbeddings
import streamlit as st
from langchain.vectorstores.faiss import FAISS
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

import os
import toml


model_name = "text-embedding-ada-002"
model_name_2 = "text-embedding-3-large"

path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/embeddings/"
github_path = "embeddings/"

secrets_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/secrets.toml"
github_secrets = "secrets.toml"

os.environ["OPENAI_API_KEY"] = toml.load(github_secrets)["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(model=model_name_2)
db = FAISS.load_local('/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/embeddings/faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Load llm for chain

llm = ChatOpenAI(
    model_name = "gpt-3.5-turbo",
)


#system_prompt = PromptTemplate(input_variables=["context"], template=sys_template)
#chain.combine_documents_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=system_prompt)

# Create retrieval chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    metadata={"application_type": "question_answering"},
)

prompt_wrapper = """Answer this question about Surya in an accurate manner: {}"""

## Get Query

opening_content = """
Hello, I'm a chatbot that can tell you what you want to know about Surya's Project Portfolio, his CV or his education. Here are some example questions to get you started:

1. How did Surya build the Pregnancy predictor project?
2. What is his educational background?
3. What is the Guilty proejct?

Ask away...

"""

if 'key' not in st.session_state:
    st.session_state.key = 'value'
    

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{
        "role": "assisstant", 
        "content": opening_content
        }]

# History

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Say Something")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    

if st.session_state.messages[-1]["role"] != "assistant":
  if query:
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        placeholder = st.empty()
        response = chain.invoke(prompt_wrapper.format(query))["result"]
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
        placeholder.markdown(response)
        



