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

print("Session Url: ", session.url)

# Load queries
queries_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx"
query_df = pd.read_excel(queries_path)
queries = query_df["queries"].tolist()

# Generate ragas database for eval
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

# Generate response dataframes for tracing eval
qa_with_ref = get_qa_with_reference(px.Client()) # pandas df
documents_retrieved = get_retrieved_documents(px.Client())

"""   Conducting Tracing evaluation   """

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


#Log evaluations

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval),
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval)
)


"""   Conducting Ragas evaluation   """

# Load dataset 

ragas_eval_ds = Dataset.from_dict(results)
ragas_eval_df = pd.DataFrame(results)

# Run evalutions
evaluation_result = evaluate(
    dataset=ragas_eval_ds,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_utilization
    ]
)

# Index and log evaluations

eval_scores_df = pd.DataFrame(evaluation_result.scores)

# Log eval scores against span_ids
eval_scores_df.index = pd.Index(
    list(reversed(qa_with_ref.index.to_list())), name=qa_with_ref.index.name
)

for eval_name in eval_scores_df.columns:
    evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
    evals = SpanEvaluations(eval_name, evals_df)
    px.Client().log_evaluations(evals)

termination_input = input("Terminate program")

# Export evaluations to excel
evals_file = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/evals/combined_evals.xlsx"

hall_eval_xl = pd.merge(qa_with_ref, hallucination_eval, on="context.span_id", how="left")
qa_corr_xl = pd.merge(qa_with_ref, qa_correctness_eval, on="context.span_id", how="left")
rel_eval_xl = pd.merge(documents_retrieved, relevance_eval, on="context.span_id", how="left")

ragas_output = pd.merge(qa_with_ref[["input", "output", "reference"]], eval_scores_df, on=qa_with_ref.index.name)

combined_evals = pd.merge(qa_with_ref, hallucination_eval[["label", "score"]], on="context.span_id", how="left")
combined_evals.rename(columns={"label": "hallucination", "score": "h_score"}, inplace=True)

combined_evals = pd.merge(combined_evals, qa_correctness_eval[["label", "score"]], on="context.span_id", how="left")
combined_evals.rename(columns={"label": "qa correctness", "score": "qa_score"}, inplace=True)

combined_evals = pd.merge(combined_evals, ragas_output[["faithfulness", "answer_relevancy", "context_utilization"]], on="context.span_id", how="left")

with pd.ExcelWriter(evals_file) as writer:
    combined_evals.to_excel(writer, sheet_name="combined_evals", index=True)
    rel_eval_xl.to_excel(writer, sheet_name="relevance", index=True)


print("...Program terminated")