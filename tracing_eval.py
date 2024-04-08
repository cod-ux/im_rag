from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from phoenix.trace.langchain import LangChainInstrumentor
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
from phoenix.trace import (
    SpanEvaluations,
    DocumentEvaluations,
)

import os
import phoenix as px
import pandas as pd
import toml


# Start Phoenix server
session = px.launch_app()

# Load vectorstore as retriever for chain
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
db = FAISS.load_local('/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/embeddings/faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

secrets_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/secrets.toml"

# Load llm for chain
os.environ["OPENAI_API_KEY"] = toml.load(secrets_path)["OPENAI_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = ChatOpenAI(
    model_name = "gpt-3.5-turbo",
)

# Create retrieval chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    metadata={"application_type": "question_answering"},
)

# Instrument chain object
LangChainInstrumentor().instrument()
print("Session url: ", session.url)

# Load query inputs
excel_df = pd.read_excel("/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx")
queries = excel_df["queries"].tolist()

for query in queries:
      chain.invoke(query)
      print(f"One Query completed")


# Load QA's and retrieved docs for eval
qa_with_ref = get_qa_with_reference(px.Client()) # pandas df
documents_retrieved = get_retrieved_documents(px.Client())

# Load evaluators
eval_model = OpenAIModel(model="gpt-3.5-turbo")

hall_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
rel_evaluator = RelevanceEvaluator(eval_model)

# Run evaluations

hallucination_eval, qa_correctness_eval = run_evals(
    dataframe=qa_with_ref,
    evaluators=[hall_evaluator, qa_correctness_evaluator],
    provide_explanation=True
)

relevance_eval = run_evals(
    dataframe=documents_retrieved,
    evaluators=[rel_evaluator],
    provide_explanation=True,
)[0]

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval),
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval)
)

evals_file = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/evals/evals.xlsx"
"""if os.path.exists(evals_file):
    os.remove(evals_file)

# Export evaluations to excel
hall_eval_xl = pd.merge(qa_with_ref, hallucination_eval, on="context.span_id", how="left")
qa_corr_xl = pd.merge(qa_with_ref, qa_correctness_eval, on="context.span_id", how="left")
rel_eval_xl = pd.merge(documents_retrieved, relevance_eval, on="context.span_id", how="left")

with pd.ExcelWriter(evals_file) as writer:
    hall_eval_xl.to_excel(writer, sheet_name="hallucination", index=True)
    qa_corr_xl.to_excel(writer, sheet_name="qa_correctness", index=True)
    rel_eval_xl.to_excel(writer, sheet_name="relevance", index=True)
"""

termination_input = input("Terminate program")

print("...Program terminated")
