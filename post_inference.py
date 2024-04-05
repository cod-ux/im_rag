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
from phoenix.trace.dsl.helpers import SpanQuery

import os
import phoenix as px
import pandas as pd
import time
import ast

emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Requirements: query_df, corpus_df 
corpus_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/corpus.xlsx"
queries_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx"
reference_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/evals/combined_evals.xlsx"

# query df - query text, query vector, answer text

query_df = pd.read_excel(queries_path)

query_vector = []
for query in query_df["queries"]:
    query_vector.append(emb_model.encode(query))

query_df["query_vector"] = query_vector

# reference_df - text, vector

reference_df = pd.read_excel(reference_path)

response_vector = []
responses = []
for response in reference_df["output"]:
    result_dict = ast.literal_eval(response)
    response_vector.append(emb_model.encode(result_dict["result"]))
    responses.append(result_dict["result"])

response_vector = response_vector[::-1]
response = response[::-1]

query_df["response"] = response_vector
query_df["response vector"] = response_vector

# corpus_df - text, vector
corpus_df = pd.read_excel(corpus_path)

text_vector = []
for text in corpus_df["text"]:
    text_vector.append(emb_model.encode(text))

corpus_df["text_vector"] = text_vector

query_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="queries", vector_column_name="query_vector"
    ),
    response_column_names="ground truth"
)

corpus_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="text", vector_column_name="text_vector"
    )
)

reference_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="response", vector_column_name="response vector"
    )
)

session = px.launch_app(
    primary=px.Dataset(query_df, query_schema, "query"),
    corpus=px.Dataset(corpus_df.reset_index(drop=True), corpus_schema, "corpus"),
    reference=px.Dataset(query_df, reference_schema, "response")
)

keyboard_input = input("Terminate program with keyboard input:  ")

"""
1. UI to capture all queries - done
2. Test data set to evaluate - done
3. Automated testing with excel input and output metrics - done
4. Run Inference api for visualizing embeddings - done
5. Test with Ragas - done
"""