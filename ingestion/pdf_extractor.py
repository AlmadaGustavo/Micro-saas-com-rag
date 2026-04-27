"""
ingestion/pdf_extractor.py
==========================
Extrai blocos de texto estruturados de PDFs de editais públicos.

O PyMuPDF fornece cada span com seus atributos tipograficos (tamanho de fonte,
negrito, posicao). A partir disso, reconstruimos a hierarquia do documento
(Secao > Clausula > Paragrafo) sem depender de tags HTML ou estrutura previa.

Cada bloco extraido carrega metadados de localizacao (pagina, clausula, secao)
e semanticos (tipo do chunk, presenca de datas/valores) que alimentam o chunker.

Dependencias:
    pip install pymupdf
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# Enums

class ChunkType(str, Enum):
    """Classificacao semantica do bloco. Usado como filtro no ChromaDB."""
    PRAZO = "prazo"
    DOCUMENTO_OBRIGATORIO = "documento_obrigatorio"
    CRITERIO_DESCLASSIFICACAO = "criterio_desclassificacao"
    OBJETO = "objeto"
    CABECALHO = "cabecalho"
    OUTRO = "outro"


class HeadingLevel(int, Enum):
    """Nivel hierarquico do bloco, reconstruido a partir da tipografia."""
    SECAO = 1       # "5. HABILITACAO"          - fonte grande ou negrito destacado
    CLAUSULA = 2    # "5.1 Documentos exigidos"  - fonte media com negrito
    SUBCLAUSULA = 3 # "5.1.1 Certidoes"          - negrito no tamanho do corpo
    PARAGRAFO = 4   # Texto corrido              - sem destaque tipografico


# Padroes de classificacao semantica

# Cada ChunkType tem uma lista de padroes regex. O classificador conta quantos
# padroes casam e usa isso como score — o tipo com mais matches vence.
_PATTERNS: dict[ChunkType, list[str]] = {
    ChunkType.PRAZO: [
        r"\bprazo.{0,30}\d{2}/\d{2}/\d{4}",
        r"\bdata.{0,20}limit",
        r"\bencerr.{0,20}(em|ate|no dia)",
        r"\bvalidade\b.{0,30}\d+.{0,10}dias",
        r"\bate o dia\b",
        r"\babertura.{0,20}(sessao|proposta)",
    ],
    ChunkType.DOCUMENTO_OBRIGATORIO: [
        r"\bdevera\s+apresentar\b",
        r"\be\s+obrigatorio\b",
        r"\bdocumentacao\s+exigida\b",
        r"\bcomprovante\s+de\b",
        r"\bcertidao\s+negativa\b",
        r"\batestado\s+de\s+capacidade",
        r"\bhabilitacao\s+juridica\b",
    ],
    ChunkType.CRITERIO_DESCLASSIFICACAO: [
        r"\bsera desclassificad",
        r"\bimplicara.{0,20}desclassif",
        r"\binabilitad",
        r"\bsera excluíd",
        r"\bimpediment",
        r"\bveda\b",
        r"\bproibid",
    ],
    ChunkType.OBJETO: [
        r"\bobjeto\b",
        r"\bcontratacao\b",
        r"\baquisicao\b",
        r"\bfornecimento\b",
        r"\bprestacao de servicos\b",
    ],
}

# Detectores de entidades para os flags booleanos do bloco
_RE_DATE  = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_RE_VALUE = re.compile(r"R\$\s*[\d.,]+")
_RE_CNPJ  = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")

# Detecta titulos de secao no padrao "1. DAS INSCRICOES" ou "2 - DO OBJETO"
_SECTION_RE = re.compile(r"^\d+[\.\-]\s+[A-ZAAEEIIOOUU\u00C0-\u00DC]{3,}", re.UNICODE)

# Detecta rodapes de diario oficial para nao confundi-los com secoes
_RODAPE_RE = re.compile(
    r"(diario oficial|dom\s+\d+|edicao extra|pagina\s+\d+|"
    r"segunda-feira|terca-feira|quarta-feira|quinta-feira|sexta-feira)",
    re.IGNORECASE,
)

# Dataclasses de saida

@dataclass
class ExtractedBlock:
    """
    Representa um paragrafo ou clausula extraido do PDF, com todo o contexto
    estrutural e semantico necessario para o chunker e o indexador.
    """
    text: str

    # Onde este bloco esta no documento
    page_number: int
    section_title: str      # titulo da secao pai 
    clause_id: str          # numero da clausula 
    hierarchy_path: str     # caminho completo 
    heading_level: HeadingLevel

    # Classificacao semantica
    chunk_type: ChunkType
    contains_date: bool
    contains_value: bool
    contains_cnpj: bool
    referenced_clauses: list[str]  # clausulas mencionadas no texto ("conforme 3.1.4")

    # Atributos tipograficos do span representativo
    font_size: float
    is_bold: bool

    # Identificacao do documento de origem
    edital_id: str      # SHA-256 (8 chars) do arquivo
    source_file: str    # nome do arquivo


@dataclass
class ExtractionResult:
    """Resultado completo da extracao de um edital."""
    blocks: list[ExtractedBlock]
    edital_id: str
    source_file: str
    total_pages: int
    is_scanned: bool        # True se o PDF nao tem texto nativo 
    doc_metadata: dict      # metadados nativos do PDF 
    warnings: list[str] = field(default_factory=list)


# Extrator principal

class EditalExtractor:
    """
    Abre um PDF de edital e extrai seus blocos de texto com contexto estrutural.

    O processo tem quatro etapas:
      1. Diagnosticar se o PDF tem texto nativo ou e escaneado.
      2. Extrair todos os spans com atributos tipograficos via PyMuPDF.
      3. Calcular as estatisticas de fonte para identificar o corpo e headings.
      4. Agrupar spans em blocos coerentes e classificar cada um.

    Uso:
        with EditalExtractor("edital.pdf") as ext:
            result = ext.extract()
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF nao encontrado: {self.pdf_path}")

        self._doc: fitz.Document = fitz.open(str(self.pdf_path))
        self._edital_id = self._compute_id()
        self._warnings: list[str] = []

    def extract(self) -> ExtractionResult:
        """Executa o pipeline completo e retorna o ExtractionResult."""
        is_scanned = self._detect_scanned()

        if is_scanned:
            self._warnings.append(
                "PDF parece ser escaneado (texto nativo ausente). "
                "A qualidade da extracao sera inferior. "
                "Considere pre-processar com OCR (pytesseract)."
            )

        raw_spans  = self._extract_spans()
        font_stats = self._compute_font_stats(raw_spans)
        blocks     = self._build_blocks(raw_spans, font_stats)

        return ExtractionResult(
            blocks=blocks,
            edital_id=self._edital_id,
            source_file=self.pdf_path.name,
            total_pages=len(self._doc),
            is_scanned=is_scanned,
            doc_metadata=self._doc.metadata,
            warnings=self._warnings,
        )

    def close(self):
        self._doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # Diagnostico

    def _detect_scanned(self) -> bool:
        """
        Amostra ate 5 paginas e conta os caracteres extraidos.
        Media abaixo de 50 chars/pagina indica PDF escaneado (sem texto nativo).
        """
        sample = min(5, len(self._doc))
        total  = sum(len(self._doc[i].get_text("text")) for i in range(sample))
        return (total / sample) < 50

    # Extracao de spans

    def _extract_spans(self) -> list[dict]:
        """
        Extrai todos os spans do PDF com seus atributos tipograficos.

        Um span e a unidade minima do PyMuPDF: texto continuo com a mesma fonte,
        tamanho e estilo. Usamos get_text("dict") em vez de get_text("text")
        justamente para ter acesso a esses atributos.

        Retorna lista de dicts com: text, size, bold, page, bbox, font.
        """
        all_spans = []

        for page_num, page in enumerate(self._doc, start=1):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = texto, 1 = imagem
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()

                        # Limpar caracteres de substituicao (encoding corrompido)
                        text = text.replace("\ufffd", " ")
                        text = unicodedata.normalize("NFC", text)

                        if not text:
                            continue

                        flags   = span.get("flags", 0)
                        is_bold = bool(flags & 2**4)  # bit 4 = negrito no PyMuPDF

                        all_spans.append({
                            "text": text,
                            "size": round(span.get("size", 12.0), 1),
                            "bold": is_bold,
                            "page": page_num,
                            "bbox": span.get("bbox", (0, 0, 0, 0)),
                            "font": span.get("font", ""),
                        })

        return all_spans

    # Estatisticas de fonte

    def _compute_font_stats(self, spans: list[dict]) -> dict:
        """
        Determina o tamanho de fonte do corpo do texto e os tamanhos de heading.

        Em vez de assumir que o corpo e sempre 12pt (o que varia por orgao),
        pesamos cada tamanho pelo comprimento do texto que o usa. O tamanho
        mais frequente (em caracteres) e o corpo. Tudo acima e heading.

        Retorna dict com 'body_size' (float) e 'heading_sizes' (list, desc).
        """
        from collections import Counter

        size_counter: Counter = Counter()
        for span in spans:
            size_counter[span["size"]] += len(span["text"])

        if not size_counter:
            return {"body_size": 12.0, "heading_sizes": []}

        body_size     = size_counter.most_common(1)[0][0]
        heading_sizes = sorted(
            [s for s in size_counter if s > body_size + 0.5],
            reverse=True,
        )

        return {"body_size": body_size, "heading_sizes": heading_sizes}

    # Construcao dos blocos

    def _build_blocks(self, spans: list[dict], font_stats: dict) -> list[ExtractedBlock]:
        """
        Agrupa spans consecutivos de mesmo estilo em blocos coerentes.

        A logica e simples: enquanto o estilo (tamanho + negrito) e a pagina
        nao mudam, acumulamos spans no buffer. Quando muda, fazemos flush e
        criamos um ExtractedBlock com o texto concatenado.

        O flush tambem atualiza o contexto hierarquico (secao, clausula atual)
        que e herdado pelos blocos subsequentes.
        """
        body_size     = font_stats["body_size"]
        heading_sizes = font_stats["heading_sizes"]

        current_section   = ""
        current_clause_id = ""
        hierarchy_stack: list[str] = []

        blocks: list[ExtractedBlock] = []
        buffer_spans: list[dict] = []

        def flush_buffer():
            nonlocal current_section, current_clause_id, hierarchy_stack

            if not buffer_spans:
                return

            combined_text = " ".join(s["text"] for s in buffer_spans).strip()
            combined_text = re.sub(r"\s+", " ", combined_text)

            # Descartar fragmentos muito curtos
            if len(combined_text.split()) < 3:
                buffer_spans.clear()
                return

            rep   = buffer_spans[0]
            level = self._classify_heading_level(rep, body_size, heading_sizes)

            # Atualizar secao atual se o bloco e um titulo de secao valido
            if _SECTION_RE.match(combined_text.strip()) and not _RODAPE_RE.search(combined_text):
                current_section = combined_text[:120]

            # Atualizar hierarquia tipografica
            if level in (HeadingLevel.SECAO, HeadingLevel.CLAUSULA, HeadingLevel.SUBCLAUSULA):
                detected_id = self._detect_clause_id(combined_text)

                if level == HeadingLevel.SECAO:
                    current_section = combined_text[:120]
                    hierarchy_stack = [detected_id or combined_text[:40]]
                elif level == HeadingLevel.CLAUSULA:
                    hierarchy_stack = hierarchy_stack[:1]
                    hierarchy_stack.append(detected_id or combined_text[:40])
                else:
                    hierarchy_stack = hierarchy_stack[:2]
                    hierarchy_stack.append(detected_id or combined_text[:40])

                if detected_id:
                    current_clause_id = detected_id

            hierarchy_path = " > ".join(h for h in hierarchy_stack if h)

            blocks.append(ExtractedBlock(
                text=combined_text,
                page_number=rep["page"],
                section_title=current_section,
                clause_id=current_clause_id,
                hierarchy_path=hierarchy_path,
                heading_level=level,
                chunk_type=self._classify_chunk_type(combined_text, rep["page"]),
                contains_date=bool(_RE_DATE.search(combined_text)),
                contains_value=bool(_RE_VALUE.search(combined_text)),
                contains_cnpj=bool(_RE_CNPJ.search(combined_text)),
                referenced_clauses=self._extract_referenced_clauses(combined_text),
                font_size=rep["size"],
                is_bold=rep["bold"],
                edital_id=self._edital_id,
                source_file=self.pdf_path.name,
            ))
            buffer_spans.clear()

        prev_page = -1
        prev_size = -1.0
        prev_bold = False

        for span in spans:
            same_page  = span["page"] == prev_page
            same_style = (
                abs(span["size"] - prev_size) < 0.5
                and span["bold"] == prev_bold
            )

            if buffer_spans and (not same_page or not same_style):
                flush_buffer()

            buffer_spans.append(span)
            prev_page = span["page"]
            prev_size = span["size"]
            prev_bold = span["bold"]

        flush_buffer()

        return blocks

    # ------------------------------------------------------------------
    # Classificadores
    # ------------------------------------------------------------------

    def _classify_heading_level(
        self, span: dict, body_size: float, heading_sizes: list[float]
    ) -> HeadingLevel:
        """
        Determina o nivel hierarquico de um span combinando tres criterios:
          1. Tamanho relativo ao corpo (maior = heading mais alto)
          2. Negrito
          3. Padrao numerico no inicio ("5.", "5.1", "5.1.1")

        A combinacao e necessaria porque editais municipais frequentemente
        usam o mesmo tamanho de fonte para corpo e clausulas, diferenciando
        apenas pelo negrito e numeracao.
        """
        size = span["size"]
        bold = span["bold"]
        text = span["text"].strip()

        has_top_level_num = bool(re.match(r"^\d+\s*[\.\-]\s+[A-Z\u00C0-\u00DC]", text))
        has_clause_num    = bool(re.match(r"^\d+\.\d+", text))

        if len(heading_sizes) >= 2 and size >= heading_sizes[0]:
            return HeadingLevel.SECAO
        if (len(heading_sizes) >= 1 and size >= heading_sizes[-1]) or has_top_level_num:
            return HeadingLevel.SECAO
        if bold and has_clause_num:
            depth = text.split()[0].count(".")
            return HeadingLevel.CLAUSULA if depth == 1 else HeadingLevel.SUBCLAUSULA
        if bold and size > body_size:
            return HeadingLevel.CLAUSULA
        if bold:
            return HeadingLevel.SUBCLAUSULA

        if _SECTION_RE.match(text.strip()):
            depth = text.strip().split()[0].count(".")
            return HeadingLevel.SECAO if depth == 0 else HeadingLevel.CLAUSULA

        return HeadingLevel.PARAGRAFO

    def _classify_chunk_type(self, text: str, page_number: int = 99) -> ChunkType:
        """
        Classifica o tipo semantico do bloco por contagem de matches de regex.

        O tipo com mais padroes casados vence. Em empate, uma ordem de
        prioridade desempata (criterio > prazo > documento > objeto).

        Blocos nas primeiras duas paginas que parecem cabecalho institucional
        (com "edital no", "resolucao no" etc.) sao classificados como CABECALHO
        antes de qualquer outra verificacao.
        """
        text_lower = text.lower()

        if page_number <= 2:
            HEADER_SIGNALS = [
                r"\bedital\s+n[o\u00ba\u00b0]",
                r"\bresolucao\s+n[o\u00ba\u00b0]",
                r"\bconsider",
                r"\bportaria\s+n[o\u00ba\u00b0]",
            ]
            if any(re.search(p, text_lower) for p in HEADER_SIGNALS):
                return ChunkType.CABECALHO

        scores: dict[ChunkType, int] = {}
        for chunk_type, patterns in _PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text_lower))
            if score > 0:
                scores[chunk_type] = score

        if not scores:
            return ChunkType.OUTRO

        PRIORITY = [
            ChunkType.PRAZO,
            ChunkType.CRITERIO_DESCLASSIFICACAO,
            ChunkType.DOCUMENTO_OBRIGATORIO,
            ChunkType.OBJETO,
            ChunkType.CABECALHO,
            ChunkType.OUTRO,
        ]
        max_score  = max(scores.values())
        candidates = [ct for ct, s in scores.items() if s == max_score]

        for ct in PRIORITY:
            if ct in candidates:
                return ct

        return ChunkType.OUTRO

    def _detect_clause_id(self, text: str) -> str:
        """
        Extrai o identificador numerico de uma clausula do inicio do texto.
        Ex: "5.1.2 Certidoes obrigatorias..." -> "5.1.2"
        Retorna string vazia se nenhum padrao for encontrado.
        """
        match = re.match(r"^(\d+(?:\.\d+){0,4})\s*[\.\-]?\s+\S", text.strip())
        return match.group(1) if match else ""

    def _extract_referenced_clauses(self, text: str) -> list[str]:
        """
        Detecta referencias cruzadas a outras clausulas dentro do bloco.
        Ex: "conforme disposto no item 8.2.1" -> ["8.2.1"]

        Isso e importante para o RAG: quando um chunk referencia outra clausula,
        o retriever pode buscar aquela clausula tambem e incluir no contexto.
        """
        pattern = re.compile(
            r"(?:item|subitem|clausula|secao|artigo|inciso)\s+(\d+(?:\.\d+){0,4})",
            re.IGNORECASE,
        )
        refs      = pattern.findall(text)
        bare_refs = re.findall(r"\((?:ver\s+)?(\d+\.\d+(?:\.\d+)?)\)", text, re.IGNORECASE)

        return list(dict.fromkeys(refs + bare_refs))  # deduplica preservando ordem

    def _compute_id(self) -> str:
        """SHA-256 (8 chars) do conteudo do arquivo. Identificador estavel do edital."""
        return hashlib.sha256(self.pdf_path.read_bytes()).hexdigest()[:8]


# Relatorio de diagnostico 

def print_extraction_report(result: ExtractionResult) -> None:
    from collections import Counter

    print("=" * 60)
    print(f"RELATORIO DE EXTRACAO - {result.source_file}")
    print("=" * 60)
    print(f"Edital ID : {result.edital_id}")
    print(f"Paginas   : {result.total_pages}")
    print(f"Escaneado : {'SIM - qualidade reduzida' if result.is_scanned else 'Nao'}")
    print(f"Blocos    : {len(result.blocks)}")

    type_counts = Counter(b.chunk_type for b in result.blocks)
    print("\nDistribuicao de chunk_type:")
    for ct, count in type_counts.most_common():
        print(f"  {ct.value:<32} {count:>4} blocos")

    if result.warnings:
        print("\nAvisos:")
        for w in result.warnings:
            print(f"  - {w}")

    print("\nAmostra (primeiros 3 blocos nao-outro):")
    shown = 0
    for block in result.blocks:
        if block.chunk_type == ChunkType.OUTRO:
            continue
        print(f"\n  [{block.chunk_type.value}] p.{block.page_number} | {block.hierarchy_path}")
        print(f"  {block.text[:200]}...")
        shown += 1
        if shown >= 3:
            break

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python pdf_extractor.py <caminho_do_edital.pdf>")
        sys.exit(1)

    with EditalExtractor(sys.argv[1]) as extractor:
        result = extractor.extract()
        print_extraction_report(result)