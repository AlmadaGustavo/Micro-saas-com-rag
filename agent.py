"""
agent.py
========
Orquestração em Grafo (LangGraph) para o pipeline Agentic RAG de Editais Públicos.
"""

from typing import List, Optional, TypedDict
from typing_extensions import Literal
import time
import re
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END

from ingestion.indexer import EditalIndexer, SearchResult

# ===========================================================================
# 1. ESTRUTURAS COMPARTILHADAS (Trazidas para cá para evitar import circular)
# ===========================================================================

SCORE_THRESHOLD = 0.55

@dataclass
class SourceReference:
    """Referencia de fonte para exibicao na interface."""
    chunk_type: str
    section_title: str
    clause_id: str
    page: int
    score: float
    excerpt: str
    source_file: str

@dataclass
class RAGResponse:
    """Resposta completa do pipeline, incluindo metadados de rastreabilidade."""
    question: str
    answer: str
    sources: list
    detected_intent: Optional[str]
    n_chunks_retrieved: int
    chunks_above_threshold: int
    duration_seconds: float
    warning: Optional[str] = None

# Padrões do Roteador por Regex
_INTENT_PATTERNS = [
    ([r"\bprazo\b", r"\bdata\b", r"\bquando\b", r"\bencerra\b", r"\blimite\b", r"\bvalidade\b", r"\babertura\b", r"\bvigencia\b"], "prazo"),
    ([r"\bdocumentos?\b", r"\bcertid[ao]\b", r"\batestado\b", r"\bcomprovante\b", r"\bhabilitacao\b", r"\bobrigatori", r"\bexigid", r"\brequisito\b", r"\bapresentar\b"], "documento_obrigatorio"),
    ([r"\bdesclassifica", r"\binabilita", r"\bimpedimento\b", r"\bveda\b", r"\bproibid", r"\belimina", r"\bexclui"], "criterio_desclassificacao"),
    ([r"\bobjeto\b", r"\bcontratacao\b", r"\baquisicao\b", r"\bfornecimento\b", r"\bservico\b"], "objeto"),
]

_INTENT_PRIORITY = ["criterio_desclassificacao", "prazo", "documento_obrigatorio", "objeto"]

def detect_intent(question: str) -> Optional[str]:
    q = question.lower()
    scores: dict[str, int] = {}
    for patterns, chunk_type in _INTENT_PATTERNS:
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            scores[chunk_type] = score
    if not scores:
        return None
    max_score = max(scores.values())
    best = next((p for p in _INTENT_PRIORITY if scores.get(p, 0) == max_score), max(scores, key=lambda k: scores[k]))
    if max_score >= 2 or (max_score == 1 and len(question.split()) <= 10):
        return best
    return None

# Prompt Mestre
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

# ===========================================================================
# 2. DEFINIÇÃO DO ESTADO E DO GRAFO (Continua igual)
# ===========================================================================

class AgentState(TypedDict):
    question: str
    current_query: str
    intent: Optional[str]
    results: List[SearchResult]
    relevant_chunks: List[SearchResult]
    loop_count: int
    answer: str
    sources: list

# Configurações do Ciclo
DEFAULT_MODEL = "llama3"
MAX_LOOPS = 2  # Proteção contra loops infinitos de reescrita

class EditalAgentGraph:
    """
    Substitui a execução linear por uma máquina de estados agêntica.
    Navega dinamicamente entre busca, crítica e reformulação de perguntas.
    """

    def __init__(self, indexer: EditalIndexer, model: str = DEFAULT_MODEL, edital_id: Optional[str] = None):
        self.indexer = indexer
        self.edital_id = edital_id  # Mantém o isolamento multi-tenant do app.py
        self._llm = OllamaLLM(model=model, temperature=0.0)  # Temperatura zero para auditoria
        self.graph = self._build_graph()

    # ==========================================
    # NÓS DE EXECUÇÃO (NODES)
    # ==========================================

    def node_route_and_search(self, state: AgentState) -> dict:
        """Determina o foco temático e interage de forma híbrida com o ChromaDB."""
        question = state["question"]
        current_query = state.get("current_query", question)
        loop_count = state.get("loop_count", 0)

        # Na primeira iteração, tenta o roteamento leve via regex do retriever.py
        intent = state.get("intent") if loop_count > 0 else detect_intent(current_query)
        
        seen = set()
        merged = []
        n_results = 4

        # Integração direta e segura com o método search() do seu indexer.py
        if intent:
            for r in self.indexer.search(query=current_query, n_results=n_results, chunk_type=intent, edital_id=self.edital_id):
                if r.parent_id not in seen:
                    seen.add(r.parent_id)
                    merged.append(r)

        for r in self.indexer.search(query=current_query, n_results=n_results, edital_id=self.edital_id):
            if r.parent_id not in seen:
                seen.add(r.parent_id)
                merged.append(r)

        merged.sort(key=lambda x: x.child_score)
        results = merged[:n_results * 2]

        return {
            "current_query": current_query,
            "intent": intent,
            "results": results,
            "loop_count": loop_count + 1
        }

    def node_grade_documents(self, state: AgentState) -> dict:
        """Agente de Crítica: Filtra alucinações avaliando a utilidade real dos blocos."""
        question = state["question"]
        results = state["results"]
        relevant_chunks = []

        # Casamento perfeito com a métrica de distância cossena definida no seu retriever
        hard_filtered = [r for r in results if r.child_score <= SCORE_THRESHOLD]
        chunks_to_grade = hard_filtered if hard_filtered else results

        for chunk in chunks_to_grade:
            text = chunk.parent_text or chunk.child_text
            prompt = f"""Você é um auditor jurídico rigoroso. Avalie se o trecho do edital abaixo responde diretamente ou traz contexto essencial para a pergunta do usuário.
            
            Trecho: {text}
            Pergunta: {question}
            
            Responda APENAS com a palavra 'SIM' se for relevante ou 'NAO' se for irrelevante. Não mude a grafia.
            Veredito:"""
            
            verdict = self._llm.invoke(prompt).strip().upper()
            if "SIM" in verdict:
                relevant_chunks.append(chunk)

        return {"relevant_chunks": relevant_chunks}

    def node_rewrite_query(self, state: AgentState) -> dict:
        """Agente de Tradução: Transforma termos leigos para a linguagem burocrática de licitações."""
        current_query = state["current_query"]
        
        prompt = f"""Você é um consultor especialista em licitações públicas brasileiras. O sistema falhou em localizar respostas para a busca: '{current_query}'.
        Reescreva essa dúvida usando termos formais de editais, jargões jurídicos (ex: certidões ao invés de papéis) e sinônimos da Lei 14.133 para otimizar os vetores.
        Retorne EXCLUSIVAMENTE a nova linha de pergunta, sem preâmbulos ou aspas.
        Nova Pergunta:"""
        
        new_query = self._llm.invoke(prompt).strip()
        return {"current_query": new_query}

    def node_generate_answer(self, state: AgentState) -> dict:
        """Nó de Síntese: Monta o contexto final enriquecido e executa o prompt mestre."""
        question = state["question"]
        final_chunks = state["relevant_chunks"] if state["relevant_chunks"] else state["results"]

        parts = []
        sources = []
        for i, r in enumerate(final_chunks, 1):
            text = r.parent_text or r.child_text
            header = f"[Trecho {i}]"
            if r.section_title: header += f" | Secao: {r.section_title[:80]}"
            if r.clause_id: header += f" | Clausula: {r.clause_id}"
            header += f" | Pagina: {r.page_start} | Tipo: {r.chunk_type}"

            parts.append(f"{header}\n{text}")
            sources.append(SourceReference(
                chunk_type=r.chunk_type, section_title=r.section_title, clause_id=r.clause_id,
                page=r.page_start, score=r.child_score, excerpt=r.child_text[:200], source_file=r.source_file
            ))

        context = "\n\n---\n\n".join(parts)
        prompt_text = PROMPT.format(context=context, question=question)
        answer = self._llm.invoke(prompt_text)

        return {"answer": answer, "sources": sources}

    # ==========================================
    # LOGICA CONDICIONAL DE ROTEAMENTO
    # ==========================================

    def decide_to_generate(self, state: AgentState) -> Literal["generate", "rewrite", "force_generate"]:
        if state["relevant_chunks"]:
            return "generate"
        
        if state["loop_count"] >= MAX_LOOPS:
            return "force_generate"
            
        return "rewrite"

    # ==========================================
    # COMPILAÇÃO DA ESTRUTURA DO GRAFO
    # ==========================================

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Mapeamento de Nós
        workflow.add_node("route_and_search", self.node_route_and_search)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("rewrite_query", self.node_rewrite_query)
        workflow.add_node("generate_answer", self.node_generate_answer)

        # Conexões fixas
        workflow.add_edge(START, "route_and_search")
        workflow.add_edge("route_and_search", "grade_documents")

        # Conexões Dinâmicas/Bordas Condicionais
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate_answer",
                "rewrite": "rewrite_query",
                "force_generate": "generate_answer"
            }
        )

        workflow.add_edge("rewrite_query", "route_and_search")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # ==========================================
    # MÉTODO ASK COMPATÍVEL COM APP.PY
    # ==========================================

    def ask(self, question: str) -> RAGResponse:
        start_time = time.time()
        
        inputs = {"question": question, "loop_count": 0}
        output = self.graph.invoke(inputs)

        warning = None
        if not output.get("relevant_chunks"):
            warning = "O agente considerou os trechos recuperados imprecisos após tentativas de reescrita. Resposta gerada sob risco de omissão."

        return RAGResponse(
            question=question,
            answer=output["answer"],
            sources=output["sources"],
            detected_intent=output["intent"],
            n_chunks_retrieved=len(output["results"]),
            chunks_above_threshold=len(output["relevant_chunks"]),
            duration_seconds=round(time.time() - start_time, 2),
            warning=warning
        )