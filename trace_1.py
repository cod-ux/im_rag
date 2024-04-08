from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.trace import (
    SpanEvaluations,
    DocumentEvaluations,
)
from phoenix.trace.dsl.helpers import SpanQuery
from phoenix.session.evaluation import (
    get_qa_with_reference,
    get_retrieved_documents
)
from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)

from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
    context_precision,
    answer_correctness,
    context_recall,
    context_relevancy
)

import os
import phoenix as px
import pandas as pd
import toml
from datasets import Dataset

# Launch Phoenix
session = px.launch_app()

# Load vectorstore as retriever for chain
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
db = FAISS.load_local('/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/embeddings/faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

secrets_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/secrets.toml"

# Load llm for chain
os.environ["OPENAI_API_KEY"] = toml.load(secrets_path)["OPENAI_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    return_source_documents=True,
    metadata={"application_type": "question_answering"},
)

LangChainInstrumentor().instrument()

print("Session Url: ", session.url)

queries_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx"
query_df = pd.read_excel(queries_path)
queries = query_df["queries"].tolist()

for query in queries:
    result = chain.invoke(query)

termination_input = input("Terminate program")
