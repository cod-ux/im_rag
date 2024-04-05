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

emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Requirements: query_df, corpus_df 
corpus_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/corpus.xlsx"
queries_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx"

# query df - query text, query vector, answer text

query_df = pd.read_excel(queries_path)

query_vector = []
for query in query_df["queries"]:
    query_vector.append(emb_model.encode(query))

query_df["query_vector"] = query_vector

# reference_df - text, vector

truth_vector = []
for truth in query_df["ground truth"]:
    truth_vector.append(emb_model.encode(truth))

query_df["truth vector"] = truth_vector

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
        raw_data_column_name="ground truth", vector_column_name="truth vector"
    )
)

session = px.launch_app(
    primary=px.Dataset(query_df, query_schema, "query"),
    corpus=px.Dataset(corpus_df.reset_index(drop=True), corpus_schema, "corpus"),
    reference=px.Dataset(query_df, reference_schema, "ground truth")
)

keyboard_input = input("Terminate program with keyboard input:  ")

"""
1. UI to capture all queries - done
2. Test data set to evaluate - done
3. Automated testing with excel input and output metrics - done
4. Run Inference api for visualizing embeddings - done
5. Test with Ragas - done
"""