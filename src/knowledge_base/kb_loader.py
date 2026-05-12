from __future__ import annotations

import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.core.config import settings
from src.core.logger import logger


@dataclass
class Chunk:
    """A single STG guideline chunk with its text, metadata and embedding."""
    uuid: str
    text: str
    chapter: str
    section: str
    chunk_id: str
    embedding: np.ndarray = field(repr=False)


class KnowledgeBase:
    """
    In-memory knowledge base loaded from Athuman's ChromaDB artefacts.

    Provides:
      - vector_search(query_embedding, k) → top-k chunks by cosine similarity
      - All chunks as a list for BM25 indexing
    """

    _DIM: int = 384
    _NODE_SIZE: int = 1676
    _VECTOR_OFFSET: int = 140

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None
        self._loaded = False

    def load(self) -> None:
        """Load all KB artefacts. Called once at startup."""
        if self._loaded:
            return

        t0 = time.perf_counter()
        logger.info("Loading knowledge base from ChromaDB artefacts...")

        pickle_path = (
            f"{settings.CHROMA_DB_PATH}/"
            "fa328576-6557-427f-9cfd-0fccb62ab19e/index_metadata.pickle"
        )
        with open(pickle_path, "rb") as f:
            meta = pickle.load(f)
        label_to_uuid: Dict[int, str] = meta["label_to_id"]

        bin_path = (
            f"{settings.CHROMA_DB_PATH}/"
            "fa328576-6557-427f-9cfd-0fccb62ab19e/data_level0.bin"
        )
        with open(bin_path, "rb") as f:
            raw = f.read()

        total_nodes = len(raw) // self._NODE_SIZE
        uuid_to_vec: Dict[str, np.ndarray] = {}
        for label in range(1, total_nodes + 1):
            uuid = label_to_uuid.get(label)
            if uuid is None:
                continue
            offset = (label - 1) * self._NODE_SIZE + self._VECTOR_OFFSET
            vec = np.frombuffer(
                raw[offset: offset + self._DIM * 4], dtype=np.float32
            ).copy()
            uuid_to_vec[uuid] = vec

        logger.info(f"Loaded {len(uuid_to_vec)} vectors from HNSW binary")

        sqlite_path = f"{settings.CHROMA_DB_PATH}/chroma.sqlite3"
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                e.embedding_id,
                MAX(CASE WHEN em.key='chroma:document' THEN em.string_value END) as doc,
                MAX(CASE WHEN em.key='chapter'         THEN em.string_value END) as chapter,
                MAX(CASE WHEN em.key='section'         THEN em.string_value END) as section,
                MAX(CASE WHEN em.key='chunk_id'        THEN em.string_value END) as chunk_id
            FROM embeddings e
            LEFT JOIN embedding_metadata em ON e.id = em.id
            GROUP BY e.embedding_id
        """)
        rows = cur.fetchall()
        conn.close()

        logger.info(f"Loaded {len(rows)} rows from SQLite")

        chunks: List[Chunk] = []
        for (uuid, doc, chapter, section, chunk_id) in rows:
            vec = uuid_to_vec.get(uuid)
            if vec is None or not doc:
                continue
            chunks.append(Chunk(
                uuid=uuid,
                text=doc,
                chapter=chapter or "",
                section=section or "",
                chunk_id=chunk_id or uuid,
                embedding=vec,
            ))

        self.chunks = chunks

        self._matrix = np.stack([c.embedding for c in chunks], axis=0)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = self._matrix / norms

        self._loaded = True
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"Knowledge base ready: {len(self.chunks)} chunks, "
            f"matrix {self._matrix.shape}, loaded in {elapsed}ms ✓"
        )

    def vector_search(self, query_embedding: List[float], k: int = 5) -> List[tuple]:
        """
        Find top-k chunks by cosine similarity to query embedding.

        Args:
            query_embedding: 384-dim float list from embedder.encode_query()
            k: number of results to return

        Returns:
            List of (score, Chunk) tuples sorted by descending similarity.
        """
        if not self._loaded:
            raise RuntimeError("KnowledgeBase.load() must be called before searching.")

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        scores = self._matrix @ q
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(float(scores[i]), self.chunks[i]) for i in top_indices]

    @property
    def texts(self) -> List[str]:
        """All chunk texts — used to build BM25 index."""
        return [c.text for c in self.chunks]


kb: KnowledgeBase = KnowledgeBase()
