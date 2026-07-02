from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k=5):
        # Try to find the right search method
        if hasattr(self.client, 'search'):
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k
            )
        elif hasattr(self.client, 'query_points'):
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                with_payload=True,
                limit=top_k
            )
        else:
            raise AttributeError("No search method found on QdrantClient")

        # Debug: print the raw results
        # print(f"Raw results type: {type(results)}")
        # print(f"Raw results: {results}")
        # print(f"Results length: {len(results) if hasattr(results, '__len__') else 'N/A'}")

        context = []
        sources = set()

        for res in results.points:
            print(f"Result item: {res}, type: {type(res)}")
            payload = getattr(res, 'payload', None) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                context.append(text)
                sources.add(source)
        
        return {"context": context, "sources": list(sources)}