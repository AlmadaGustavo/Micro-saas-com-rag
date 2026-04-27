"""
retriever.py
============
Pipeline RAG completo para perguntas sobre editais publicos.

Fluxo de uma pergunta:
  1. Detectar a intencao (prazo? documento? criterio?) para pre-filtrar a busca.
  2. Busca hibrida: busca filtrada por chunk_type + busca semantica livre,
     mescladas e deduplicadas por parent_id.
  3. Expandir cada child recuperado para seu parent (contexto completo).
  4. Montar o prompt com os trechos numerados e metadados de localizacao.
  5. Chamar o LLM via Ollama e retornar a resposta com rastreabilidade de fontes.

Dependencias:
    pip install langchain-core langchain-ollama
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from ingestion.indexer import EditalIndexer, SearchResult
from ingestion.chunker import ChunkType


# Configuracao

DEFAULT_MODEL     = "llama3"
DEFAULT_BASE_URL  = "http://localhost:11434"
DEFAULT_N_RESULTS = 4
SCORE_THRESHOLD   = 0.55  # distancia coseno maxima para considerar um chunk relevante


# Roteador de intencao

# Cada entrada e uma lista de padroes regex associada a um ChunkType.
# O detect_intent conta quantos padroes casam por tipo e usa isso como score.
_INTENT_PATTERNS = [
    (
        [r"\bprazo\b", r"\bdata\b", r"\bquando\b", r"\bencerra\b",
         r"\blimite\b", r"\bvalidade\b", r"\babertura\b", r"\bvigencia\b"],
        ChunkType.PRAZO.value,
    ),
    (
        [r"\bdocumentos?\b", r"\bcertid[ao]\b", r"\batestado\b", r"\bcomprovante\b",
         r"\bhabilitacao\b", r"\bobrigatori", r"\bexigid", r"\brequisito\b", r"\bapresentar\b"],
        ChunkType.DOCUMENTO_OBRIGATORIO.value,
    ),
    (
        [r"\bdesclassifica", r"\binabilita", r"\bimpedimento\b",
         r"\bveda\b", r"\bproibid", r"\belimina", r"\bexclui"],
        ChunkType.CRITERIO_DESCLASSIFICACAO.value,
    ),
    (
        [r"\bobjeto\b", r"\bcontratacao\b", r"\baquisicao\b",
         r"\bfornecimento\b", r"\bservico\b"],
        ChunkType.OBJETO.value,
    ),
]

# Em caso de empate de score, este e o tipo que prevalece.
# Criterio de desclassificacao tem prioridade maxima porque perguntas como
# "quando o candidato e desclassificado?" ativam tanto "quando" (prazo)
# quanto "desclassificado" (criterio), e o criterio e mais especifico.
_INTENT_PRIORITY = [
    ChunkType.CRITERIO_DESCLASSIFICACAO.value,
    ChunkType.PRAZO.value,
    ChunkType.DOCUMENTO_OBRIGATORIO.value,
    ChunkType.OBJETO.value,
]


def detect_intent(question: str) -> Optional[str]:
    """
    Detecta o tipo semantico mais provavel de uma pergunta.

    Retorna o ChunkType vencedor ou None se a pergunta for ambigua.
    A regra de ativacao exige ao menos 2 padroes casados, ou 1 padrao
    em perguntas curtas (ate 10 palavras) onde a intencao e mais clara.

    Retornar None faz o retriever usar busca semantica pura sem filtro,
    o que e mais seguro para perguntas abertas.
    """
    q = question.lower()
    scores: dict[str, int] = {}

    for patterns, chunk_type in _INTENT_PATTERNS:
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            scores[chunk_type] = score

    if not scores:
        return None

    max_score = max(scores.values())
    best = next(
        (p for p in _INTENT_PRIORITY if scores.get(p, 0) == max_score),
        max(scores, key=lambda k: scores[k]),
    )

    if max_score >= 2:
        return best
    if max_score == 1 and len(question.split()) <= 10:
        return best

    return None


# Prompt do LLM

_TEMPLATE = """Voce e um especialista em analise de editais publicos brasileiros.
Responda a pergunta com base EXCLUSIVAMENTE nos trechos do edital abaixo.

REGRAS:
1. Use apenas as informacoes dos trechos. Nunca invente dados.
2. Se a informacao nao estiver nos trechos, diga: "Nao encontrei essa informacao nos trechos recuperados do edital."
3. Ao citar prazos, datas ou valores, indique sempre a pagina ou clausula de origem.
4. Seja objetivo e direto. Use bullet points para listas.
5. Responda em portugues brasileiro.

TRECHOS DO EDITAL:
{context}

PERGUNTA: {question}

RESPOSTA:"""

PROMPT = PromptTemplate(input_variables=["context", "question"], template=_TEMPLATE)


# Dataclasses de saida

@dataclass
class SourceReference:
    """Referencia de fonte para exibicao na interface."""
    chunk_type: str
    section_title: str
    clause_id: str
    page: int
    score: float
    excerpt: str        # primeiros 200 chars do child 
    source_file: str


@dataclass
class RAGResponse:
    """Resposta completa do pipeline, incluindo metadados de rastreabilidade."""
    question: str
    answer: str
    sources: list
    detected_intent: Optional[str]
    n_chunks_retrieved: int
    chunks_above_threshold: int     # chunks com score abaixo do SCORE_THRESHOLD
    duration_seconds: float
    warning: Optional[str] = None   # aviso quando nenhum chunk e suficientemente relevante

    def __str__(self):
        lines = [
            f"PERGUNTA : {self.question}",
            f"INTENCAO : {self.detected_intent or 'geral'}",
            f"CHUNKS   : {self.chunks_above_threshold}/{self.n_chunks_retrieved} relevantes",
            f"TEMPO    : {self.duration_seconds:.2f}s",
            "",
            "RESPOSTA:",
            self.answer,
        ]
        if self.warning:
            lines.append(f"\nAviso: {self.warning}")
        if self.sources:
            lines.append("\nFONTES:")
            for i, s in enumerate(self.sources, 1):
                lines.append(
                    f"  [{i}] p.{s.page} | {s.chunk_type} | score={s.score} | {s.excerpt[:80]}..."
                )
        return "\n".join(lines)


# Retriever principal

class EditalRetriever:
    """
    Pipeline RAG para perguntas sobre editais publicos.

    Combina roteamento de intencao, busca hibrida no ChromaDB, expansao
    parent-child e geracao via Ollama em um unico metodo .ask().

    Parametros:
        indexer   : EditalIndexer inicializado com embedding.
        model     : modelo Ollama a usar (ex: "llama3", "mistral").
        base_url  : URL do servidor Ollama.
        n_results : numero de chunks a recuperar por busca.
        edital_id : restringir buscas a um edital especifico (suporte multi-tenant).
    """

    def __init__(
        self,
        indexer:   EditalIndexer,
        model:     str = DEFAULT_MODEL,
        base_url:  str = DEFAULT_BASE_URL,
        n_results: int = DEFAULT_N_RESULTS,
        edital_id: Optional[str] = None,
    ):
        self.indexer   = indexer
        self.n_results = n_results
        self.edital_id = edital_id

        self._llm = OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=0.0,    # deterministico, importante para docs juridicos
            num_predict=1024,
        )
        self._chain = PROMPT | self._llm | StrOutputParser()

    def ask(self, question: str) -> RAGResponse:
        """Recebe uma pergunta em linguagem natural e retorna RAGResponse completo."""
        start   = time.time()
        intent  = detect_intent(question)
        results = self._hybrid_search(question, intent)

        relevant = [r for r in results if r.child_score <= SCORE_THRESHOLD]
        context, sources = self._build_context(relevant or results)

        warning = None
        if not relevant:
            warning = (
                f"Nenhum chunk com score <= {SCORE_THRESHOLD}. "
                "A resposta pode ser imprecisa."
            )

        answer = self._generate(question, context)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            detected_intent=intent,
            n_chunks_retrieved=len(results),
            chunks_above_threshold=len(relevant),
            duration_seconds=round(time.time() - start, 2),
            warning=warning,
        )

    def ask_stream(self, question: str):
        """
        Versao streaming para uso no Streamlit.
        Faz yield de tokens do LLM a medida que sao gerados.

        Uso no Streamlit:
            placeholder = st.empty()
            full = ""
            for token in retriever.ask_stream(pergunta):
                full += token
                placeholder.markdown(full + "...")
            placeholder.markdown(full)
        """
        intent   = detect_intent(question)
        results  = self._hybrid_search(question, intent)
        relevant = [r for r in results if r.child_score <= SCORE_THRESHOLD]
        context, _ = self._build_context(relevant or results)
        prompt_text = PROMPT.format(context=context, question=question)
        for token in self._llm.stream(prompt_text):
            yield token

    # Busca hibrida

    def _hybrid_search(self, question: str, intent: Optional[str]) -> list[SearchResult]:
        """
        Realiza busca em duas camadas e mescla os resultados.

        Camada 1 (filtrada): busca apenas nos chunks do tipo detectado.
        Camada 2 (livre): busca semantica sem restricao de tipo.

        Os resultados das duas camadas sao deduplicados por parent_id e
        ordenados por score. O cap de n_results * 2 evita um contexto
        excessivamente longo no prompt.
        """
        seen:   set[str]          = set()
        merged: list[SearchResult] = []

        # Camada 1: filtrada por intencao
        if intent:
            for r in self.indexer.search(
                query=question, n_results=self.n_results,
                chunk_type=intent, edital_id=self.edital_id,
            ):
                if r.parent_id not in seen:
                    seen.add(r.parent_id)
                    merged.append(r)

        # Camada 2: semantica pura
        for r in self.indexer.search(
            query=question, n_results=self.n_results, edital_id=self.edital_id,
        ):
            if r.parent_id not in seen:
                seen.add(r.parent_id)
                merged.append(r)

        merged.sort(key=lambda r: r.child_score)
        return merged[: self.n_results * 2]

    # Montagem do contexto

    def _build_context(self, results: list[SearchResult]) -> tuple[str, list[SourceReference]]:
        """
        Monta o bloco de contexto que sera enviado ao LLM.

        Usa o texto do PARENT (nao do child) para dar contexto completo ao LLM.
        O child foi usado apenas para a busca semantica.

        Cada trecho recebe um cabecalho com pagina, clausula e tipo,
        permitindo que o LLM cite a fonte corretamente na resposta.
        """
        parts:   list[str]             = []
        sources: list[SourceReference] = []

        for i, r in enumerate(results, 1):
            text   = r.parent_text or r.child_text
            header = f"[Trecho {i}]"
            if r.section_title:
                header += f" | Secao: {r.section_title[:80]}"
            if r.clause_id:
                header += f" | Clausula: {r.clause_id}"
            header += f" | Pagina: {r.page_start} | Tipo: {r.chunk_type}"

            parts.append(f"{header}\n{text}")
            sources.append(SourceReference(
                chunk_type=r.chunk_type,
                section_title=r.section_title,
                clause_id=r.clause_id,
                page=r.page_start,
                score=r.child_score,
                excerpt=r.child_text[:200],
                source_file=r.source_file,
            ))

        return "\n\n---\n\n".join(parts), sources

    # Geracao

    def _generate(self, question: str, context: str) -> str:
        """Invoca o LLM com o prompt montado. Retorna mensagem de erro se o Ollama estiver indisponivel."""
        try:
            return self._chain.invoke({"context": context, "question": question})
        except Exception as e:
            return (
                f"Erro ao chamar o modelo: {e}\n"
                "Verifique se o Ollama esta rodando: ollama serve"
            )


if __name__ == "__main__":
    import sys
    from ingestion.indexer import EditalIndexer, OllamaEmbedder

    indexer = EditalIndexer(persist_dir="./chroma_db", embedding_fn=OllamaEmbedder())
    stats   = indexer.stats()

    if stats["children_total"] == 0:
        print("ChromaDB vazio. Execute o indexer.py primeiro.")
        sys.exit(1)

    print(f"ChromaDB: {stats['children_total']} children | {stats['parents_total']} parents")

    model     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    retriever = EditalRetriever(indexer=indexer, model=model)
    print(f"Modelo: {model} | Threshold: {SCORE_THRESHOLD}\n")

    demo = [
        "Qual o prazo para entrega das propostas?",
        "Quais documentos sao obrigatorios para habilitacao?",
        "Quais sao os criterios de desclassificacao?",
    ]

    for q in demo:
        print(f"\n{'='*60}")
        print(retriever.ask(q))

    print(f"\n{'='*60}\nModo interativo ('sair' para encerrar):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("sair", "exit", "quit", ""):
            break
        print(retriever.ask(q))