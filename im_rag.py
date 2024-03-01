from langchain.embeddings import SentenceTransformerEmbeddings
import streamlit as st
from langchain.vectorstores import FAISS
from openai import OpenAI

embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
db = FAISS.load_local('/Users/suryaganesan/vscode/ml/learning/Gen AI/faiss_index', embeddings)
api_key = "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"


client = OpenAI(api_key=api_key)

def gather_results(query, db):
    results = db.similarity_search_with_relevance_scores(query, k=3)

    return results

def pass_to_llm(results, query, success):
    mod_results = [doc.page_content for doc, rel_sc in results]
    seperator = '\\n'
    context = seperator.join(mod_results)

    system_prompt = "You are a helpful chatbot that replies to user questions on Surya Ganesan's education and professional experience"

    user_prompt_1 = f"""
    Here is the user's question:
    {query}

    Here is some relevant text from Surya's CV to answer the user's question. Use the information below to answer their question:
    {context}
    """
    user_prompt_0 = f"""
    Here is the user's question:
    {query}

    There is not any relevant answer to this question. But use the following information from your CV to give an acceptable answer:
    {context}

    """

    if success == 1:
       response = client.chat.completions.create(
          model = "gpt-3.5-turbo",
          messages = [{'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_prompt_1}
          ]
       )

    else:
        response = client.chat.completions.create(
          model = "gpt-3.5-turbo",
          messages = [{'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_prompt_0}
          ]
       )

    return response.choices[0].message.content


## Get Query

opening_content = """
Hello, I'm a chatbot that can tell you what you want to know about Surya's Project Portfolio, his CV or his education. Here are some example questions to get you started:

1. How did Surya build the Pregnancy predictor project?
2. What is his educational background?
3. What is the Guilty proejct?

Ask away...

"""

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
        results = gather_results(query=query, db=db)
        score_a = results[0][1]
        doc_a = results[0][0]
        if score_a < 0.3:
            response = pass_to_llm(results, query, 0)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            placeholder.markdown(response)
        
        else:
            response = pass_to_llm(results, query, 1)
            placeholder = st.empty()
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            placeholder.markdown(response)



## Work to be done
# Unfuck the prompt template
# Craft a better CV & portfolio doc, include cover letter, Include read me of projects
