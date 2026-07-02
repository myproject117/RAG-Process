import inngest
import logging
from fastapi import FastAPI
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGQueryResult
)

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="rag_inngest_pdf",
    name="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> dict: #RAGChunkAndSrc
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(chunks=chunks, source_id=source_id).dict()


    def _upsert(chunks_and_src: dict) -> dict:
        chunks = chunks_and_src['chunks']
        source_id = chunks_and_src['source_id']
        vecs = embed_texts(chunks)
        ids = [abs(hash(f"{source_id}: {i}")) % (2**31) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        qdrant = QdrantStorage()        
        qdrant.upsert(ids, vecs, payloads)

        return RAGUpsertResult(ingested=len(chunks)).dict()


    #RAGChunkAndSrc 
    chunks_and_src: dict = await ctx.step.run(step_id="load-and-chunk", handler=lambda: _load(ctx))
    
    # RAGUpsertResult
    ingested: dict = await ctx.step.run(step_id="embed-and-upsert", handler=lambda: _upsert(chunks_and_src))
    
    return ingested


@inngest_client.create_function(
    fn_id="rag_query_pdf",
    name="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)


async def rag_quesry_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int):
        query_vec = embed_texts([question])[0]
        print(f"Query vector dimension: {len(query_vec)}")
        
        store = QdrantStorage()
        # try:
        #     collection_info = store.client.get_collection(collection_name="docs")
        #     print(f"Points in collection: {collection_info.points_count}")
        # except Exception as e:
        #     print(f"Error getting collection info: {e}")
        
        found = store.search(query_vec, top_k)
        print(f"Search results: {found}")
        return RAGSearchResult(context=found["context"], sources=found["sources"]).dict()

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(step_id="embed-and-search", handler=lambda: _search(question, top_k))

    context_block = "\n\n".join(f"- {c}" for c in found['context'])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
       auth_key = os.getenv("OPENAI_API_KEY"),
       model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context"},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]['content'].strip()
    return {
            "answer": answer, 
            "sources": found['sources'], 
            "num_contexts": len(found['context'])
           }

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_inngest_pdf, rag_quesry_pdf_ai])


# for route in app.routes:
#     print(f"Route: {route.path}")

# Output
# --------
# Route: /openapi.json
# Route: /docs
# Route: /docs/oauth2-redirect
# Route: /redoc
# Route: /api/inngest
# Route: /api/inngest
# Route: /api/inngest
