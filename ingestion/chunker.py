"""
ingestion/chunker.py
====================
Divide os blocos extraidos em chunks hierarquicos para indexacao no ChromaDB.

Estrategia Parent-Child:
  - Parent (~700 tokens): contexto completo de uma clausula, enviado ao LLM.
  - Child  (~175 tokens): fragmento menor, indexado por embedding para busca.

O fluxo e: extrair secoes logicas -> gerar parents -> subdividir em children.
Na recuperacao, o sistema busca pelo child (semantica precisa) e expande para
o parent (contexto completo) antes de montar o prompt.

Dependencias: nenhuma alem do modulo pdf_extractor.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from ingestion.pdf_extractor import (
    ChunkType,
    ExtractionResult,
    ExtractedBlock,
    HeadingLevel,
)


# Configuracoes

PARENT_TARGET_TOKENS  = 700
PARENT_OVERLAP_TOKENS = 100
CHILD_TARGET_TOKENS   = 175
CHILD_OVERLAP_TOKENS  = 30

# Dataclasses

@dataclass
class Chunk:
    """
    Unidade pronta para indexacao no ChromaDB.
    O campo 'metadata' contem todos os campos filtráveis.
    O campo 'text' e o que sera embedding-ado (child) ou enviado ao LLM (parent).
    """
    id: str
    text: str
    token_count: int
    level: str              # "parent" ou "child"
    parent_id: Optional[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    """Resultado completo do chunking de um edital."""
    parents: list[Chunk]
    children: list[Chunk]
    edital_id: str
    source_file: str
    stats: dict = field(default_factory=dict)

    @property
    def all_chunks(self) -> list[Chunk]:
        return self.parents + self.children


# Tokenizacao e divisao de texto

def count_tokens(text: str) -> int:
    """
    Estimativa de tokens baseada em contagem de palavras.
    O fator 1.35 aproxima bem o comportamento de tokenizadores como o do Llama
    para textos em portugues (onde palavras tendem a gerar mais de 1 token).
    Para producao com OpenAI embeddings, substitua por tiktoken.
    """
    return max(1, int(len(text.split()) * 1.35))


def split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Divide o texto em janelas de tamanho max_tokens com sobreposicao.

    A divisao e feita por palavras (nao por tokens brutos) para evitar cortar
    no meio de uma palavra. O overlap garante que o contexto de uma janela
    esteja presente no inicio da proxima, evitando perda de continuidade.

    Apos a divisao, verifica se algum chunk comeca com virgula ou numero de
    resolucao (fragmento de lista) e, se sim, prefixa com as ultimas 20
    palavras do chunk anterior para restaurar o contexto.
    """
    words = text.split()
    if not words:
        return []

    words_per_chunk = int(max_tokens / 1.35)
    overlap_words   = int(overlap_tokens / 1.35)

    # Garantir que o avanco nunca seja zero 
    advance = words_per_chunk - overlap_words
    if advance <= 0:
        advance = max(1, words_per_chunk // 2)

    chunks = []
    i = 0
    total = len(words)

    while i < total:
        end = min(i + words_per_chunk, total)
        chunks.append(" ".join(words[i:end]))
        if end == total:
            break
        i += advance

    # Correcao de fragmentos: chunks que comecam com virgula ou "NNN/AAAA"
    # indicam que a divisao cortou no meio de uma lista — prefixar com contexto anterior
    cleaned = []
    for idx, chunk in enumerate(chunks):
        first_word = chunk.lstrip().split()[0] if chunk.split() else ""
        is_fragment = (
            first_word.startswith(",")
            or bool(re.match(r"^\d{3}/\d{4}", first_word))
        )
        if is_fragment and idx > 0:
            prefix = " ".join(chunks[idx - 1].split()[-20:])
            chunk  = prefix + " " + chunk
        cleaned.append(chunk)

    return cleaned


# Chunker principal

class EditalChunker:
    """
    Gera pares Parent-Child a partir dos blocos extraidos pelo EditalExtractor.

    Etapas:
      1. Agrupar blocos em secoes logicas (cada secao inicia em um heading).
      2. Converter cada secao em um ou mais Parents (se longa, divide com overlap).
      3. Subdividir cada Parent em Children menores para indexacao semantica.
      4. Propagar todos os metadados da secao para parents e children.
    """

    def __init__(
        self,
        parent_target_tokens:  int = PARENT_TARGET_TOKENS,
        parent_overlap_tokens: int = PARENT_OVERLAP_TOKENS,
        child_target_tokens:   int = CHILD_TARGET_TOKENS,
        child_overlap_tokens:  int = CHILD_OVERLAP_TOKENS,
    ):
        self.parent_target  = parent_target_tokens
        self.parent_overlap = parent_overlap_tokens
        self.child_target   = child_target_tokens
        self.child_overlap  = child_overlap_tokens

    def chunk(self, extraction: ExtractionResult) -> ChunkResult:
        """Ponto de entrada. Recebe ExtractionResult e retorna ChunkResult."""
        sections = self._group_into_sections(extraction.blocks)

        parents:  list[Chunk] = []
        children: list[Chunk] = []

        for section in sections:
            section_parents = self._build_parents(section, extraction)
            for parent in section_parents:
                parents.append(parent)
                children.extend(self._build_children(parent, extraction))

        return ChunkResult(
            parents=parents,
            children=children,
            edital_id=extraction.edital_id,
            source_file=extraction.source_file,
            stats=self._compute_stats(parents, children),
        )

    # Agrupamento em secoes

    def _group_into_sections(self, blocks: list[ExtractedBlock]) -> list[list[ExtractedBlock]]:
        """
        Agrupa blocos consecutivos em secoes logicas.

        Uma nova secao comeca sempre que encontramos um heading de nivel SECAO
        ou CLAUSULA. Blocos de paragrafo sao sempre anexados a secao corrente,
        nunca ficam isolados (preserva contexto da clausula que os precede).
        """
        sections: list[list[ExtractedBlock]] = []
        current:  list[ExtractedBlock] = []

        for block in blocks:
            is_boundary = block.heading_level in (HeadingLevel.SECAO, HeadingLevel.CLAUSULA)
            if is_boundary and current:
                sections.append(current)
                current = []
            current.append(block)

        if current:
            sections.append(current)

        return sections

    # Construcao dos Parents

    def _build_parents(self, blocks: list[ExtractedBlock], extraction: ExtractionResult) -> list[Chunk]:
        """
        Converte uma secao logica em um ou mais chunks Parent.

        Se a secao cabe em parent_target tokens, gera um unico Parent.
        Se for maior, divide com overlap para nao perder continuidade entre janelas.

        Os metadados do Parent incluem localizacao (pagina, secao, clausula) e
        semantica (chunk_type dominante, flags de data/valor). Esses metadados
        sao herdados por todos os Children gerados a partir deste Parent.
        """
        text = self._join_blocks(blocks)
        rep  = blocks[0]

        types    = list({b.chunk_type for b in blocks})
        dominant = self._dominant_type(types)
        pages    = sorted({b.page_number for b in blocks})

        base_metadata = {
            "section_title":  rep.section_title[:200],
            "clause_id":      rep.clause_id,
            "hierarchy_path": rep.hierarchy_path[:300],
            "page_start":     pages[0],
            "page_end":       pages[-1],
            "chunk_type":     dominant.value,
            "contains_date":  any(b.contains_date  for b in blocks),
            "contains_value": any(b.contains_value for b in blocks),
            "edital_id":      extraction.edital_id,
            "source_file":    extraction.source_file,
            "chunk_level":    "parent",
        }

        windows = split_by_tokens(text, self.parent_target, self.parent_overlap)

        parents = []
        for idx, window in enumerate(windows):
            parents.append(Chunk(
                id=self._make_id(window, extraction.edital_id),
                text=window,
                token_count=count_tokens(window),
                level="parent",
                parent_id=None,
                metadata={**base_metadata, "window_index": idx},
            ))

        return parents

    # Construcao dos Children

    def _build_children(self, parent: Chunk, extraction: ExtractionResult) -> list[Chunk]:
        """
        Subdivide um Parent em Children menores para indexacao semantica.

        Cada Child herda todos os metadados do Parent e adiciona:
          - parent_id: referencia ao Parent para expansao na recuperacao
          - child_index: posicao dentro do Parent

        Children com menos de 10 tokens sao descartados (fragmentos residuais).
        """
        texts = split_by_tokens(parent.text, self.child_target, self.child_overlap)

        children = []
        for idx, text in enumerate(texts):
            if count_tokens(text) < 10:
                continue
            children.append(Chunk(
                id=self._make_id(text, extraction.edital_id),
                text=text,
                token_count=count_tokens(text),
                level="child",
                parent_id=parent.id,
                metadata={
                    **parent.metadata,
                    "chunk_level": "child",
                    "parent_id":   parent.id,
                    "child_index": idx,
                },
            ))

        return children

    # Utilitarios internos

    def _join_blocks(self, blocks: list[ExtractedBlock]) -> str:
        """
        Concatena os blocos de uma secao em texto corrido.
        Headings recebem quebra dupla para preservar a estrutura visual
        (facilita a leitura do contexto pelo LLM).
        """
        parts = []
        for b in blocks:
            if b.heading_level in (HeadingLevel.SECAO, HeadingLevel.CLAUSULA):
                parts.append(f"\n\n{b.text}")
            else:
                parts.append(b.text)

        return re.sub(r"\n{3,}", "\n\n", " ".join(parts)).strip()

    def _dominant_type(self, types: list[ChunkType]) -> ChunkType:
        """
        Retorna o ChunkType de maior prioridade presente na secao.
        Usado para classificar o Parent quando a secao contem blocos mistos.
        """
        priority = [
            ChunkType.PRAZO,
            ChunkType.DOCUMENTO_OBRIGATORIO,
            ChunkType.CRITERIO_DESCLASSIFICACAO,
            ChunkType.OBJETO,
            ChunkType.CABECALHO,
            ChunkType.OUTRO,
        ]
        for ct in priority:
            if ct in types:
                return ct
        return ChunkType.OUTRO

    def _make_id(self, text: str, edital_id: str) -> str:
        """ID deterministico: SHA-256 do edital_id + texto, truncado em 12 chars."""
        return hashlib.sha256((edital_id + text).encode()).hexdigest()[:12]

    def _compute_stats(self, parents: list[Chunk], children: list[Chunk]) -> dict:
        """Calcula estatisticas basicas de tokens para o relatorio."""
        if not parents or not children:
            return {}

        def s(lst):
            return {"count": len(lst), "min": min(lst), "max": max(lst),
                    "mean": round(sum(lst) / len(lst), 1)}

        return {
            "parents":  s([p.token_count for p in parents]),
            "children": s([c.token_count for c in children]),
        }


# Relatorio de diagnostico

def print_chunk_report(result: ChunkResult) -> None:
    s = result.stats
    print("=" * 60)
    print(f"RELATORIO DE CHUNKING - {result.source_file}")
    print("=" * 60)
    if s:
        p, c = s["parents"], s["children"]
        print(f"Parents  : {p['count']} chunks | tokens: min={p['min']} mean={p['mean']} max={p['max']}")
        print(f"Children : {c['count']} chunks | tokens: min={c['min']} mean={c['mean']} max={c['max']}")
        print(f"Media de children por parent: {round(c['count']/p['count'], 1)}")
    print("=" * 60)


if __name__ == "__main__":
    import sys, time
    from pathlib import Path
    from ingestion.pdf_extractor import EditalExtractor

    path_pdf = r"C:\Users\Gustavo\Downloads\DOM-5250-27.12.2021-Ed-Extra.pdf"

    if not Path(path_pdf).exists():
        print(f"Arquivo nao encontrado: {path_pdf}")
        sys.exit(1)

    start = time.time()

    with EditalExtractor(path_pdf) as ext:
        extraction = ext.extract()
    print(f"Extracao: {len(extraction.blocks)} blocos em {time.time()-start:.2f}s")

    t = time.time()
    chunker = EditalChunker()
    result  = chunker.chunk(extraction)
    print(f"Chunking: {len(result.parents)} parents, {len(result.children)} children em {time.time()-t:.2f}s")

    print_chunk_report(result)