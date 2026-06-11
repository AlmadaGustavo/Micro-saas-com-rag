"""
app.py
======
Interface Streamlit para o sistema RAG Agêntico de análise de editais públicos.

Execução:
    streamlit run app.py
"""

import os
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página — deve ser o primeiro comando Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EditalIA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS customizado — tema institucional sóbrio com acento azul-governo
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset e base */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Fundo geral */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2a3a;
}

/* Header hero */
.hero-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #1a6bb5 0%, transparent 70%);
    opacity: 0.3;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e8f4ff;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #6b8cad;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: #1a3a5c;
    border: 1px solid #1e6bb5;
    color: #4da6ff;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Cards de métricas */
.metric-card {
    background: #161b27;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4da6ff;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b8cad;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* Mensagens do chat */
.chat-message {
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 16px;
    line-height: 1.7;
    font-size: 0.94rem;
}
.chat-user {
    background: #1a2f4a;
    border-left: 3px solid #1e6bb5;
    color: #c8d8e8;
}
.chat-assistant {
    background: #161b27;
    border: 1px solid #1e2a3a;
    border-left: 3px solid #2ecc71;
    color: #e8eaf0;
}
.chat-warning {
    background: #1f1a0e;
    border-left: 3px solid #f39c12;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.82rem;
    color: #c8a640;
    margin-top: 8px;
}

/* Intent badge */
.intent-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 6px;
}
.intent-prazo        { background: #1a3a5c; color: #4da6ff; border: 1px solid #1e5a8a; }
.intent-documento    { background: #1a3a2a; color: #4dbb7a; border: 1px solid #1e6b40; }
.intent-criterio     { background: #3a1a1a; color: #ff6b6b; border: 1px solid #6b2020; }
.intent-objeto       { background: #2a2a1a; color: #d4aa4d; border: 1px solid #6b5a20; }
.intent-geral        { background: #1e1e1e; color: #8a8a8a; border: 1px solid #333; }

/* Sources */
.source-card {
    background: #0f1117;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.source-page  { color: #4da6ff; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
.source-score { color: #6b8cad; font-family: 'JetBrains Mono', monospace; }
.source-type  { color: #2ecc71; }
.source-text  { color: #8a9ab0; margin-top: 4px; font-size: 0.78rem; line-height: 1.5; }

/* Upload area */
.upload-zone {
    border: 2px dashed #1e3a5f;
    border-radius: 10px;
    padding: 28px 20px;
    text-align: center;
    background: #0d1520;
    transition: border-color 0.2s;
}

/* Botões */
.stButton > button {
    background: #1a4a7a;
    color: #e8f4ff;
    border: 1px solid #1e6bb5;
    border-radius: 6px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #1e5a9a;
    border-color: #4da6ff;
    color: #ffffff;
}

/* Input de texto */
.stTextInput > div > div > input,
.stChatInput > div > div > textarea {
    background: #161b27 !important;
    border: 1px solid #1e2a3a !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* Spinner */
.stSpinner { color: #4da6ff !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #161b27 !important;
    border: 1px solid #1e2a3a !important;
    color: #e8eaf0 !important;
}

/* Divider */
hr { border-color: #1e2a3a !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Imports do pipeline — Adaptado para carregar o Grafo Agêntico do LangGraph
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline(model: str, persist_dir: str):
    """
    Inicializa o indexer e o agente em grafo uma única vez (cache do Streamlit).
    Re-executa apenas se model ou persist_dir mudarem.
    """
    from ingestion.indexer import EditalIndexer, OllamaEmbedder
    from agent import EditalAgentGraph  # Carrega nossa arquitetura agêntica compilada

    indexer = EditalIndexer(
        persist_dir=persist_dir,
        embedding_fn=OllamaEmbedder(),
    )
    # Instanciamos o grafo agêntico encapsulado que criamos no passo anterior
    agent_graph = EditalAgentGraph(indexer=indexer, model=model)
    return indexer, agent_graph


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

INTENT_LABELS = {
    "prazo":                    ("prazo",     "intent-prazo"),
    "documento_obrigatorio":    ("documento", "intent-documento"),
    "criterio_desclassificacao": ("critério",  "intent-criterio"),
    "objeto":                    ("objeto",    "intent-objeto"),
    None:                        ("geral",     "intent-geral"),
}

CHUNK_TYPE_PT = {
    "prazo":                     "Prazo",
    "documento_obrigatorio":     "Documento obrigatório",
    "criterio_desclassificacao": "Critério de desclassificação",
    "objeto":                    "Objeto",
    "outro":                     "Geral",
}


def render_intent_badge(intent: Optional[str]) -> str:
    label, css_class = INTENT_LABELS.get(intent, ("geral", "intent-geral"))
    return f'<span class="intent-badge {css_class}">{label}</span>'


def render_sources(sources) -> str:
    if not sources:
        return ""
    html = "<div style='margin-top:12px'>"
    for s in sources:
        type_label = CHUNK_TYPE_PT.get(s.chunk_type, s.chunk_type)
        score_color = "#2ecc71" if s.score < 0.35 else "#f39c12" if s.score < 0.50 else "#e74c3c"
        html += f"""
        <div class="source-card">
            <span class="source-page">p.{s.page}</span>
            &nbsp;·&nbsp;
            <span class="source-type">{type_label}</span>
            &nbsp;·&nbsp;
            <span style="color:{score_color};font-family:'JetBrains Mono',monospace;font-size:0.78rem">
                score {s.score:.3f}
            </span>
            {f'&nbsp;·&nbsp;<span style="color:#8a9ab0;font-size:0.78rem">{s.clause_id}</span>' if s.clause_id else ''}
            <div class="source-text">{s.excerpt[:180]}...</div>
        </div>"""
    html += "</div>"
    return html


def process_pdf(uploaded_file, indexer) -> dict:
    """Salva o PDF em temp, roda o pipeline de ingestão e retorna estatísticas."""
    from ingestion.pdf_extractor import EditalExtractor
    from ingestion.chunker import EditalChunker

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with EditalExtractor(tmp_path) as ext:
            extraction = ext.extract()

        chunker = EditalChunker()
        chunk_result = chunker.chunk(extraction)

        summary = indexer.index(chunk_result)

        return {
            "edital_id":   extraction.edital_id,
            "pages":       extraction.total_pages,
            "blocks":      len(extraction.blocks),
            "parents":     summary.parents_indexed,
            "children":    summary.children_indexed,
            "skipped":     summary.children_skipped,
            "duration":    summary.duration_seconds,
            "is_scanned":  extraction.is_scanned,
            "warnings":    extraction.warnings,
        }
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "messages":        [],       # histórico do chat
        "indexed_editais": {},       # edital_id → nome do arquivo
        "active_edital":   None,     # edital_id selecionado para perguntas
        "pipeline_ready":  False,
        "model":           "llama3",
        "persist_dir":     "./chroma_db",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0">
        <div style="font-size:1.3rem;font-weight:700;color:#e8f4ff">⚖️ EditalIA</div>
        <div style="font-size:0.75rem;color:#6b8cad;margin-top:2px">Análise Inteligente de Editais</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Configurações do modelo
    st.markdown("**⚙️ Configurações**")
    model = st.selectbox(
        "Modelo LLM",
        ["llama3", "mistral", "gemma2", "llama3.1", "phi3"],
        index=0,
        help="Modelos disponíveis no Ollama local",
    )
    st.session_state["model"] = model

    persist_dir = st.text_input(
        "Diretório ChromaDB",
        value="./chroma_db",
        help="Onde os vetores são persistidos",
    )
    st.session_state["persist_dir"] = persist_dir

    st.divider()

    # Upload de PDF
    st.markdown("**📄 Carregar Edital**")
    uploaded = st.file_uploader(
        "Selecione um PDF",
        type=["pdf"],
        help="Editais de licitação, concurso público ou vestibular",
        label_visibility="collapsed",
    )

    if uploaded:
        file_hash = hashlib.md5(uploaded.name.encode()).hexdigest()[:6]
        already_indexed = any(
            uploaded.name in name
            for name in st.session_state["indexed_editais"].values()
        )

        if already_indexed:
            st.success(f"✓ {uploaded.name} já indexado")
        else:
            if st.button("🔍 Processar e Indexar", use_container_width=True):
                try:
                    indexer, _ = load_pipeline(
                        st.session_state["model"],
                        st.session_state["persist_dir"],
                    )
                    with st.spinner("Extraindo texto e indexando..."):
                        stats = process_pdf(uploaded, indexer)

                    edital_id = stats["edital_id"]
                    st.session_state["indexed_editais"][edital_id] = uploaded.name
                    st.session_state["active_edital"] = edital_id
                    st.session_state["pipeline_ready"] = True

                    st.success(f"✓ Indexado em {stats['duration']:.1f}s")

                    col1, col2 = st.columns(2)
                    col1.metric("Páginas", stats["pages"])
                    col2.metric("Chunks", stats["children"])

                    if stats["is_scanned"]:
                        st.warning("PDF escaneado — qualidade reduzida")
                    for w in stats["warnings"]:
                        st.warning(w)

                except Exception as e:
                    st.error(f"Erro na indexação: {e}")

    st.divider()

    # Editais indexados
    if st.session_state["indexed_editais"]:
        st.markdown("**📚 Editais Indexados**")
        options = {v: k for k, v in st.session_state["indexed_editais"].items()}
        selected_name = st.selectbox(
            "Ativo",
            list(options.keys()),
            label_visibility="collapsed",
        )
        st.session_state["active_edital"] = options[selected_name]
        st.session_state["pipeline_ready"] = True

        if st.button("🗑️ Remover edital ativo", use_container_width=True):
            try:
                indexer, _ = load_pipeline(
                    st.session_state["model"],
                    st.session_state["persist_dir"],
                )
                edital_id = st.session_state["active_edital"]
                indexer.delete_edital(edital_id)
                del st.session_state["indexed_editais"][edital_id]
                st.session_state["active_edital"] = None
                st.session_state["pipeline_ready"] = False
                st.session_state["messages"] = []
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao remover: {e}")

    st.divider()

    # Estatísticas do banco
    if st.session_state["pipeline_ready"]:
        try:
            indexer, _ = load_pipeline(
                st.session_state["model"],
                st.session_state["persist_dir"],
            )
            stats_db = indexer.stats()
            st.markdown("**📊 ChromaDB**")
            c1, c2 = st.columns(2)
            c1.metric("Children", stats_db["children_total"])
            c2.metric("Parents",  stats_db["parents_total"])
        except Exception:
            pass

    # Botão limpar conversa
    if st.session_state["messages"]:
        st.divider()
        if st.button("🧹 Limpar conversa", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Conteúdo principal
# ---------------------------------------------------------------------------

# Hero header
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">Agentic RAG · LangGraph · Powered by Ollama</div>
    <div class="hero-title">EditalIA</div>
    <div class="hero-subtitle">
        Análise inteligente de editais através de agentes autônomos de reflexão gráfica.<br>
        O sistema avalia a qualidade do contexto e reescreve queries dinamicamente em caso de falhas.
    </div>
</div>
""", unsafe_allow_html=True)

# Estado: sem edital indexado
if not st.session_state["pipeline_ready"]:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem;margin-bottom:12px">📄</div>
        <div style="font-size:1rem;font-weight:600;color:#c8d8e8;margin-bottom:6px">
            Nenhum edital carregado
        </div>
        <div style="font-size:0.85rem;color:#6b8cad">
            Use o painel lateral para fazer upload de um PDF e indexá-lo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Perguntas de exemplo como inspiração
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**💡 Exemplos de perguntas que você poderá fazer:**")
    exemplos = [
        "📅 Qual o prazo para entrega das propostas?",
        "📋 Quais documentos são obrigatórios para habilitação?",
        "❌ Quais são os critérios de desclassificação?",
        "🎯 Qual o objeto desta licitação?",
        "💰 Qual o valor estimado do contrato?",
        "📝 Quais são os requisitos para inscrição?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(exemplos):
        cols[i % 2].markdown(
            f'<div style="background:#161b27;border:1px solid #1e2a3a;border-radius:6px;'
            f'padding:10px 14px;margin-bottom:8px;font-size:0.85rem;color:#8a9ab0">{ex}</div>',
            unsafe_allow_html=True,
        )

else:
    # ── Chat principal ──────────────────────────────────────────────────
    active_name = st.session_state["indexed_editais"].get(
        st.session_state["active_edital"], "Edital"
    )
    st.markdown(
        f'<div style="font-size:0.82rem;color:#6b8cad;margin-bottom:16px">'
        f'📄 Edital ativo: <span style="color:#4da6ff">{active_name}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Renderizar histórico
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message chat-user">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            intent_badge = render_intent_badge(msg.get("intent"))
            meta = (
                f'<div style="font-size:0.75rem;color:#6b8cad;margin-bottom:8px">'
                f'{intent_badge}'
                f'<span style="color:#6b8cad">'
                f'{msg.get("chunks",0)} chunks relevantes · '
                f'{msg.get("duration",0):.1f}s'
                f'</span></div>'
            )
            st.markdown(
                f'<div class="chat-message chat-assistant">'
                f'{meta}'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if msg.get("warning"):
                st.markdown(
                    f'<div class="chat-warning">⚠ {msg["warning"]}</div>',
                    unsafe_allow_html=True,
                )
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} fonte(s) recuperada(s)"):
                    st.markdown(render_sources(msg["sources"]), unsafe_allow_html=True)

    # Input do chat
    question = st.chat_input(
        "Faça uma pergunta sobre o edital...",
        disabled=not st.session_state["pipeline_ready"],
    )

    if question:
        # Adicionar mensagem do usuário
        st.session_state["messages"].append({"role": "user", "content": question})
        st.markdown(
            f'<div class="chat-message chat-user">🧑 {question}</div>',
            unsafe_allow_html=True,
        )

        # Gerar resposta via Agente LangGraph
        try:
            _, agent_graph = load_pipeline(
                st.session_state["model"],
                st.session_state["persist_dir"],
            )

            # Restringir as buscas do Grafo ao edital ativo configurado na sessão
            agent_graph.edital_id = st.session_state["active_edital"]

            # Usamos o st.spinner enquanto o LangGraph navega pelos nós (Search, Grade, Rewrite)
            with st.spinner("Agente analisando o edital e avaliando consistência..."):
                response = agent_graph.ask(question)

            # Armazenar no histórico o payload retornado pelo grafo (idêntico ao RAGResponse original)
            st.session_state["messages"].append({
                "role":    "assistant",
                "content": response.answer,
                "intent":  response.detected_intent,
                "chunks":  response.chunks_above_threshold,
                "duration": response.duration_seconds,
                "sources": response.sources,
                "warning": response.warning,
            })

            st.rerun()

        except Exception as e:
            st.error(f"Erro na execução do loop agêntico: {e}")
            st.info("Verifique se as instâncias do ChromaDB e Ollama estão operacionais.")