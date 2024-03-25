from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.trace import (
    SpanEvaluations,
    DocumentEvaluations,
)
from phoenix.trace.dsl.helpers import SpanQuery
from phoenix.session.evaluation import get_qa_with_reference
from phoenix.trace import SpanEvaluations

from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
)

import os
import phoenix as px
import pandas as pd
import toml
from datasets import Dataset

embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
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
    return_source_documents=True,
    metadata={"application_type": "question_answering"},
)

LangChainInstrumentor().instrument()

session = px.launch_app()
print(session.url)

# Load queries
queries_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx"
query_df = pd.read_excel(queries_path)
queries = query_df["queries"].tolist()

# Generate eval scores
answer = []
contexts = []
for query in queries:
    result = chain.invoke(query)
    answer.append(result["result"])
    contexts.append([r.page_content for r in result["source_documents"]])

results = {
    "question": [q for q in queries],
    "answer": [a for a in answer],
    "contexts": [c for c in contexts]
}

spans_df = get_qa_with_reference(px.Client())

ragas_eval_ds = Dataset.from_dict(results)
ragas_eval_df = pd.DataFrame(results)

evaluation_result = evaluate(
    dataset=ragas_eval_ds,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_utilization
    ]
)


eval_scores_df = pd.DataFrame(evaluation_result.scores)

# Log eval scores against span_ids
eval_scores_df.index = pd.Index(
    list(reversed(spans_df.index.to_list())), name=spans_df.index.name
)

for eval_name in eval_scores_df.columns:
    evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
    evals = SpanEvaluations(eval_name, evals_df)
    px.Client().log_evaluations(evals)

print(session.url)
x = input("Terminate...")
px.close_app()

evals_output = pd.merge(spans_df[["input", "output", "reference"]], eval_scores_df, on=spans_df.index.name)

out_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/evals/ragas_evals.xlsx"
with pd.ExcelWriter(out_path) as writer:
    evals_output.to_excel(writer, sheet_name="evals", index=True)

