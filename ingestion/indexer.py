"""
ingestion/indexer.py
====================
Persiste e consulta chunks de editais no ChromaDB.

Usa duas colecoes separadas para implementar o Parent-Child retrieval:

  editais_children  - chunks pequenos (~175 tokens), indexados por embedding.
                      Usados na busca semantica. Cada child conhece seu parent_id.

  editais_parents   - chunks grandes (~700 tokens), armazenados sem embedding.
                      Recuperados por ID apos a busca nos children.
                      Sao eles que vao para o contexto do LLM.

Essa separacao garante que a busca semantica opera em textos curtos e precisos,
enquanto o LLM recebe o contexto completo da clausula para gerar a resposta.

Dependencias:
    pip install chromadb
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

from ingestion.chunker import Chunk, ChunkResult


# Embedding functions

class OllamaEmbedder(EmbeddingFunction):
    """
    Gera embeddings via Ollama rodando localmente.

    Requer Ollama instalado e o modelo baixado:
        ollama pull nomic-embed-text

    O modelo nomic-embed-text e multilingual e funciona bem com portugues,
    ao contrario do all-MiniLM-L6-v2 que e treinado principalmente em ingles.

    Uso:
        indexer = EditalIndexer(embedding_fn=OllamaEmbedder())
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.url = "http://127.0.0.1:11434/api/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = requests.post(
                self.url,
                json={"model": self.model_name, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" in data:
                embeddings.append(data["embedding"])
            elif "embeddings" in data:
                embeddings.append(data["embeddings"][0])
        return embeddings


class HashEmbedder(EmbeddingFunction):
    """
    Embedding deterministico baseado em hash — apenas para testes unitarios.
    Nao captura semantica real. Nao usar em producao.
    """

    def __init__(self, dims: int = 384):
        self.dims = dims

    def __call__(self, input: Documents) -> Embeddings:
        result = []
        for text in input:
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            vec = rng.standard_normal(self.dims).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-9)
            result.append(vec.tolist())
        return result


# Configuracao

COLLECTION_CHILDREN = "editais_children"
COLLECTION_PARENTS  = "editais_parents"
BATCH_SIZE = 100   # numero de chunks por requisicao ao ChromaDB


# Dataclasses de saida

@dataclass
class IndexSummary:
    """Estatisticas retornadas apos uma operacao de indexacao."""
    edital_id: str
    source_file: str
    parents_indexed: int
    children_indexed: int
    parents_skipped: int    
    children_skipped: int
    duration_seconds: float
    collection_children: str
    collection_parents: str
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 56,
            f"INDEXACAO - {self.source_file}",
            "=" * 56,
            f"Edital ID         : {self.edital_id}",
            f"Parents indexados : {self.parents_indexed}"
            + (f" ({self.parents_skipped} ja existiam)" if self.parents_skipped else ""),
            f"Children indexados: {self.children_indexed}"
            + (f" ({self.children_skipped} ja existiam)" if self.children_skipped else ""),
            f"Tempo             : {self.duration_seconds:.2f}s",
        ]
        for w in self.warnings:
            lines.append(f"Aviso: {w}")
        lines.append("=" * 56)
        return "\n".join(lines)


@dataclass
class SearchResult:
    """Resultado de busca semantica com child e parent expandido."""
    child_id: str
    child_text: str
    child_score: float      # distancia coseno (menor = mais similar)

    parent_id: str
    parent_text: str        # contexto completo da clausula, enviado ao LLM

    chunk_type: str
    section_title: str
    clause_id: str
    page_start: int
    contains_date: bool
    contains_value: bool
    source_file: str
    edital_id: str

# Indexador principal

class EditalIndexer:
    """
    Gerencia a indexacao e busca de editais no ChromaDB.

    Inicializa duas colecoes no banco (criando-as se nao existirem) e
    expoe metodos para indexar ChunkResults, buscar por similaridade
    semantica com filtros de metadados, e remover editais do indice.

    Parâmetros:
        persist_dir  : diretorio onde o ChromaDB salva os dados em disco.
        embedding_fn : funcao de embedding. Use OllamaEmbedder() em producao
                       e HashEmbedder() em testes unitarios.
    """

    def __init__(
        self,
        persist_dir: str | Path = "./chroma_db",
        embedding_fn: Optional[EmbeddingFunction] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_fn = embedding_fn

        self._col_children = self._get_or_create_collection(COLLECTION_CHILDREN, embed=True)
        self._col_parents  = self._get_or_create_collection(COLLECTION_PARENTS,  embed=False)

    # API publica

    def index(self, chunk_result: ChunkResult) -> IndexSummary:
        """
        Indexa um ChunkResult completo no ChromaDB.

        E uma operacao idempotente: chunks cujo ID ja existe na colecao
        sao ignorados (deduplicacao). Isso permite re-indexar o mesmo edital
        sem gerar duplicatas.

        Parents sao indexados primeiro porque os children referenciam seus IDs.
        """
        start = time.time()
        warnings: list[str] = []

        p_indexed, p_skipped = self._upsert_chunks(self._col_parents,  chunk_result.parents,  embed=False)
        c_indexed, c_skipped = self._upsert_chunks(self._col_children, chunk_result.children, embed=True)

        if c_indexed == 0 and c_skipped == 0:
            warnings.append("Nenhum child indexado. Verifique o ChunkResult.")

        return IndexSummary(
            edital_id=chunk_result.edital_id,
            source_file=chunk_result.source_file,
            parents_indexed=p_indexed,
            children_indexed=c_indexed,
            parents_skipped=p_skipped,
            children_skipped=c_skipped,
            duration_seconds=round(time.time() - start, 2),
            collection_children=COLLECTION_CHILDREN,
            collection_parents=COLLECTION_PARENTS,
            warnings=warnings,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        chunk_type: Optional[str] = None,
        edital_id: Optional[str] = None,
        contains_date: Optional[bool] = None,
        contains_value: Optional[bool] = None,
    ) -> list[SearchResult]:
        """
        Busca semantica hibrida: embedding + filtro de metadados.

        O filtro e opcional. Quando informado, o ChromaDB restringe a busca
        apenas aos chunks que satisfazem a condicao antes de calcular
        similaridade — isso e mais eficiente que filtrar pos-busca.

        Retorna lista de SearchResult com o child encontrado e o parent
        correspondente ja expandido (lookup por ID na colecao de parents).
        """
        where = self._build_where_filter(chunk_type, edital_id, contains_date, contains_value)

        query_kwargs = {
            "query_texts": [query],
            "n_results":   min(n_results, self._col_children.count() or 1),
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw = self._col_children.query(**query_kwargs)
        except Exception:
            return []

        if not raw["ids"] or not raw["ids"][0]:
            return []

        results: list[SearchResult] = []
        seen_parents: set[str] = set()

        for i, child_id in enumerate(raw["ids"][0]):
            meta       = raw["metadatas"][0][i]
            child_text = raw["documents"][0][i]
            score      = raw["distances"][0][i]
            parent_id  = meta.get("parent_id", "")

            # Deduplicar por parent: se dois children do mesmo parent
            # aparecem no resultado, retornamos apenas o de menor score.
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            results.append(SearchResult(
                child_id=child_id,
                child_text=child_text,
                child_score=round(score, 4),
                parent_id=parent_id,
                parent_text=self._fetch_parent_text(parent_id),
                chunk_type=meta.get("chunk_type", "outro"),
                section_title=meta.get("section_title", ""),
                clause_id=meta.get("clause_id", ""),
                page_start=int(meta.get("page_start", 0)),
                contains_date=bool(meta.get("contains_date", 0)),
                contains_value=bool(meta.get("contains_value", 0)),
                source_file=meta.get("source_file", ""),
                edital_id=meta.get("edital_id", ""),
            ))

        return results

    def delete_edital(self, edital_id: str) -> dict:
        """
        Remove todos os chunks de um edital das duas colecoes.
        Util para re-indexar um edital com conteudo atualizado.
        """
        where    = {"edital_id": {"$eq": edital_id}}
        c_before = self._col_children.count()
        p_before = self._col_parents.count()

        self._col_children.delete(where=where)
        self._col_parents.delete(where=where)

        return {
            "edital_id":        edital_id,
            "children_removed": c_before - self._col_children.count(),
            "parents_removed":  p_before - self._col_parents.count(),
        }

    def stats(self) -> dict:
        """Retorna contagem de chunks nas duas colecoes e o diretorio do banco."""
        return {
            "children_total": self._col_children.count(),
            "parents_total":  self._col_parents.count(),
            "persist_dir":    str(self.persist_dir),
        }

    # Internos

    def _get_or_create_collection(self, name: str, embed: bool) -> chromadb.Collection:
        """
        Cria ou recupera uma colecao ChromaDB.
        A distancia coseno e usada porque embeddings normalizados se comportam
        melhor com coseno do que com distancia euclidiana para tarefas semanticas.
        """
        kwargs: dict = {"name": name}
        if embed and self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn

        return self._client.get_or_create_collection(
            **kwargs,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert_chunks(
        self,
        collection: chromadb.Collection,
        chunks: list[Chunk],
        embed: bool,
    ) -> tuple[int, int]:
        """
        Insere chunks em batches, pulando os que ja existem.
        Retorna (n_inseridos, n_pulados).

        Parents sao inseridos com embedding dummy ([0.0]) porque nunca sao
        buscados por similaridade — apenas recuperados por ID.
        """
        if not chunks:
            return 0, 0

        existing  = self._get_existing_ids(collection, [c.id for c in chunks])
        to_insert = [c for c in chunks if c.id not in existing]
        n_skipped = len(chunks) - len(to_insert)

        if not to_insert:
            return 0, n_skipped

        total_batches = (len(to_insert) + BATCH_SIZE - 1) // BATCH_SIZE

        for i, batch in enumerate(self._batches(to_insert, BATCH_SIZE), 1):
            print(f"   Lote {i}/{total_batches} ({len(batch)} chunks)...")

            ids       = [c.id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [self._sanitize_metadata(c.metadata) for c in batch]

            if embed:
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
            else:
                collection.add(
                    ids=ids, documents=documents, metadatas=metadatas,
                    embeddings=[[0.0]] * len(ids),
                )

        return len(to_insert), n_skipped

    def _get_existing_ids(self, collection: chromadb.Collection, ids: list[str]) -> set[str]:
        """Consulta o ChromaDB para saber quais IDs ja existem na colecao."""
        try:
            return set(collection.get(ids=ids, include=[])["ids"])
        except Exception:
            return set()

    def _fetch_parent_text(self, parent_id: str) -> str:
        """Recupera o texto de um parent pelo ID. Retorna string vazia se nao encontrado."""
        if not parent_id:
            return ""
        try:
            result = self._col_parents.get(ids=[parent_id], include=["documents"])
            if result["documents"]:
                return result["documents"][0]
        except Exception:
            pass
        return ""

    def _build_where_filter(
        self,
        chunk_type:    Optional[str],
        edital_id:     Optional[str],
        contains_date: Optional[bool],
        contains_value: Optional[bool],
    ) -> Optional[dict]:
        """
        Monta o filtro 'where' do ChromaDB combinando as condicoes informadas.
        Condicoes multiplas sao unidas com $and.
        Retorna None se nenhum filtro for informado (busca sem restricao).
        """
        conditions = []
        if chunk_type:
            conditions.append({"chunk_type":  {"$eq": chunk_type}})
        if edital_id:
            conditions.append({"edital_id":   {"$eq": edital_id}})
        if contains_date is not None:
            conditions.append({"contains_date":  {"$eq": int(contains_date)}})
        if contains_value is not None:
            conditions.append({"contains_value": {"$eq": int(contains_value)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """
        Prepara metadados para o ChromaDB.
        O ChromaDB aceita apenas str, int e float como valores de metadado.
        Booleanos precisam ser convertidos para int para funcionar em filtros.
        Listas sao convertidas para string separada por virgula.
        """
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, bool):
                clean[k] = int(v)
            elif isinstance(v, list):
                clean[k] = ",".join(str(x) for x in v)
            elif isinstance(v, (str, int, float)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _batches(items: list, size: int):
        """Divide uma lista em sublistas de tamanho maximo 'size'."""
        for i in range(0, len(items), size):
            yield items[i: i + size]


if __name__ == "__main__":
    import sys
    from ingestion.pdf_extractor import EditalExtractor
    from ingestion.chunker import EditalChunker

    if len(sys.argv) < 2:
        print("Uso: python indexer.py <caminho_do_edital.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Iniciando pipeline completo para: {pdf_path}")

    with EditalExtractor(pdf_path) as ext:
        extraction = ext.extract()
    print(f"Blocos extraidos: {len(extraction.blocks)}")

    chunker      = EditalChunker()
    chunk_result = chunker.chunk(extraction)
    print(f"Parents: {len(chunk_result.parents)} | Children: {len(chunk_result.children)}")

    indexer = EditalIndexer(persist_dir="./chroma_db", embedding_fn=OllamaEmbedder())
    summary = indexer.index(chunk_result)
    print(summary)

    stats = indexer.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")