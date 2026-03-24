import streamlit as st
import tempfile
import json
import re
from pathlib import Path

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = "--------------"  # ← paste your Groq key here


# LANGCHAIN — LOADERS, SPLITTER, EMBEDDINGS, VECTOR STORE


@st.cache_resource(show_spinner="Loading embedding model & vector store…")
def get_vectorstore():
    """
    Build a persistent LangChain Chroma vector store with HuggingFace embeddings.
    Cached so it's only created once per session.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # In-memory Chroma collection
    vectorstore = Chroma(
        collection_name="research_papers",
        embedding_function=embeddings,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_llm():
    """Return a cached ChatGroq LLM instance."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )


# DOCUMENT LOADER
def load_pdf(file_path: str):
    """Load a PDF using LangChain's PyMuPDFLoader — preserves page metadata."""
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    return loader.load()   # list of Document(page_content, metadata)
def load_docx(file_path: str):
    """Load a Word document using LangChain's Docx2txtLoader."""
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
    return loader.load()
def load_arxiv(arxiv_id: str):
    """
    Load a paper directly from arXiv using LangChain's ArxivLoader.
    Returns Document objects with title, authors, summary in metadata.
    """
    from langchain_community.document_loaders import ArxivLoader
    loader = ArxivLoader(
        query=arxiv_id.strip(),
        load_max_docs=1,
        load_all_available_meta=True,
    )
    return loader.load()
def split_documents(docs):
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    This is smarter than simple word-count splitting — it tries to split at
    paragraph → sentence → word boundaries in that order.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # characters per chunk
        chunk_overlap=150,       # overlap to preserve context across boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

# LANGGRAPH — states and nodes

from typing import TypedDict, List, Any

class ResearchState(TypedDict):
    """
    LangGraph state schema. Every node reads from and writes to this dict.
    LangGraph passes this state between nodes automatically.
    """
    question: str               # user's question
    documents: List[Any]        # retrieved Document chunks
    generation: str             # final LLM answer
    paper_names: List[str]      # names of papers in scope


def retrieve_node(state: ResearchState) -> ResearchState:
    """
    Node 1 — RETRIEVE
    Embeds the question and queries ChromaDB for the most relevant chunks.
    Uses MMR (Maximal Marginal Relevance) to reduce redundancy in results.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",          # diversity-aware retrieval
        search_kwargs={"k": 6, "fetch_k": 20},
    )
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def grade_node(state: ResearchState) -> ResearchState:
    """
    Node 2 — GRADE (filter)
    Asks the LLM to score each retrieved chunk as relevant or not.
    Filters out chunks that don't actually help answer the question.
    This is the 'corrective RAG' pattern — improves answer quality.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader. Given a question and a document chunk, "
         "respond with only 'yes' if the chunk is relevant to the question, "
         "or 'no' if it is not. No explanation."),
        ("human", "Question: {question}\n\nChunk:\n{document}"),
    ])
    chain = prompt | llm | StrOutputParser()

    filtered = []
    for doc in state["documents"]:
        score = chain.invoke({
            "question": state["question"],
            "document": doc.page_content[:400],
        })
        if "yes" in score.lower():
            filtered.append(doc)

    # Fall back to all docs if everything was filtered out
    return {**state, "documents": filtered if filtered else state["documents"]}


def generate_node(state: ResearchState) -> ResearchState:
    """
    Node 3 — GENERATE
    Builds a RAG prompt with retrieved context and calls the LLM.
    Uses LangChain's LCEL pipe syntax: prompt | llm | parser
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm()

    # Format retrieved chunks into a single context block
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', doc.metadata.get('Title', 'Unknown'))}]\n{doc.page_content}"
        for doc in state["documents"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert research assistant with deep knowledge of academic papers. "
         "Answer the question using ONLY the provided context from research papers. "
         "Always cite the paper source when referencing a specific claim. "
         "If multiple papers agree or disagree, explicitly mention this. "
         "If the context is insufficient, say so clearly."),
        ("human", "Context from papers:\n{context}\n\nQuestion: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "generation": answer}


def build_rag_graph():
    """
    Assemble the LangGraph StateGraph with 3 nodes:
    retrieve → grade → generate
    Each node is a pure function that receives and returns ResearchState.
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade",    grade_node)
    graph.add_node("generate", generate_node)

    # Define edges (execution order)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade",    "generate")
    graph.add_edge("generate", END)

    return graph.compile()


@st.cache_resource(show_spinner=False)
def get_rag_graph():
    """Cache the compiled LangGraph so it's only built once."""
    return build_rag_graph()



# Analysis of relationships between th epapers using lcel(langchain expression language) 
def analyse_relationships(papers: dict) -> list:
    """
    Uses a LangChain LCEL chain (prompt | llm | parser) to identify
    relationships between papers and return structured JSON.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm()

    summaries = "\n\n".join(
        f"Paper ID: {pid}\nTitle: {info['title']}\nExcerpt:\n{info['preview'][:600]}"
        for pid, info in papers.items()
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research graph analyst. Analyse the given research paper summaries "
         "and identify pairwise relationships between them. "
         "Respond ONLY with a valid JSON array — no prose, no markdown fences. "
         "Each item must have exactly these keys: "
         '"source" (paper id), "target" (paper id), '
         '"relation" (one of: builds_on | contradicts | uses_method | similar_approach), '
         '"reason" (one concise sentence explaining why).'),
        ("human", "Papers:\n{summaries}\n\nReturn the JSON array of relationships:"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"summaries": summaries})
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return []


# visualization of knowledge grapg using networkx and pyvis


def build_graph_html(papers: dict, relationships: list) -> str:
    from pyvis.network import Network
    import networkx as nx

    G = nx.DiGraph()
    for pid, info in papers.items():
        label = info["title"][:40] + ("…" if len(info["title"]) > 40 else "")
        G.add_node(pid, label=label)

    color_map = {
        "builds_on":        "#38bdf8",
        "contradicts":      "#f87171",
        "uses_method":      "#34d399",
        "similar_approach": "#a78bfa",
    }
    for rel in relationships:
        if rel.get("source") in papers and rel.get("target") in papers:
            G.add_edge(
                rel["source"], rel["target"],
                title=f"{rel['relation']}: {rel.get('reason', '')}",
                color=color_map.get(rel["relation"], "#94a3b8"),
                label=rel["relation"].replace("_", " "),
            )

    net = Network(height="500px", width="100%", directed=True)
    net.from_nx(G)
    net.set_options("""{
        "nodes": {"shape": "dot", "size": 22, "font": {"size": 14}},
        "edges": {"arrows": "to", "font": {"size": 11},
                  "smooth": {"type": "curvedCW", "roundness": 0.25}},
        "physics": {"stabilization": {"iterations": 150},
                    "barnesHut": {"gravitationalConstant": -4000}}
    }""")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return Path(f.name).read_text()


# SESSION STATE

if "papers" not in st.session_state:
    st.session_state.papers = {}
    # papers[paper_id] = {title, preview, source, authors}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "relationships" not in st.session_state:
    st.session_state.relationships = []



# SIDEBAR
with st.sidebar:
    st.title("🔬 ResearchLens AI")
    st.caption("LangChain + LangGraph RAG pipeline")
    st.divider()

    st.metric("Papers Loaded", len(st.session_state.papers))
    st.metric("Relationships Found", len(st.session_state.relationships))

    if st.session_state.papers:
        st.divider()
        st.subheader("📄 Loaded Papers")
        for pid, info in st.session_state.papers.items():
            st.write(f"📝 {info['title'][:48]}")
            st.caption(info.get("source", "uploaded"))

        st.divider()
        if st.button("🗑 Clear Everything", use_container_width=True):
            st.session_state.papers = {}
            st.session_state.chat_history = []
            st.session_state.relationships = []
            # Reset vector store
            get_vectorstore.clear()
            st.rerun()


# MAIN UI
st.title("🔬 ResearchLens AI")
st.caption("Powered by LangChain · LangGraph · LLaMA 3.3 70B · ChromaDB")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Ingest Papers",
    "💬 Chat",
    "🔗 Relationship Graph",
    "📊 Analysis",
])



# TAB 1 — INGEST
with tab1:
    st.subheader("Add Research Papers")
    st.caption("LangChain document loaders handle PDF, DOCX, and arXiv automatically.")

    col_upload, col_arxiv = st.columns(2, gap="large")

    with col_upload:
        st.markdown("#### 📎 Upload Files (PDF / DOCX)")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        process_btn = st.button("▶ Process Uploaded Files", use_container_width=True)

        if process_btn:
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
            else:
                vectorstore = get_vectorstore()
                progress = st.progress(0, text="Starting…")

                for i, file in enumerate(uploaded_files):
                    progress.progress((i + 1) / len(uploaded_files),
                                      text=f"Loading {file.name}…")
                    suffix = Path(file.name).suffix.lower()

                    # Save upload to a temp file so LangChain loaders can open it
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    # LangChain document loader — returns list of Document objects
                    raw_docs = load_pdf(tmp_path) if suffix == ".pdf" else load_docx(tmp_path)

                    # Add paper name to each document's metadata
                    for doc in raw_docs:
                        doc.metadata["paper_name"] = file.name
                        doc.metadata["source"] = file.name

                    # LangChain splitter — smart recursive splitting
                    chunks = split_documents(raw_docs)

                    # Add chunks to ChromaDB via LangChain
                    vectorstore.add_documents(chunks)

                    # Store paper info in session state
                    paper_id = re.sub(r"[^a-z0-9_]", "_", file.name.lower())[:40]
                    preview = " ".join(raw_docs[0].page_content.split()[:200]) if raw_docs else ""
                    st.session_state.papers[paper_id] = {
                        "title": file.name,
                        "preview": preview,
                        "source": "upload",
                    }

                progress.empty()
                st.success(f"✅ {len(uploaded_files)} file(s) ingested — {len(chunks)} chunks indexed!")
                st.rerun()

    with col_arxiv:
        st.markdown("#### 🔭 arXiv Papers")
        arxiv_input = st.text_area(
            "arXiv IDs (one per line)",
            placeholder="2303.08774\n2310.06825\n1706.03762",
            height=130,
            label_visibility="visible",
        )
        fetch_btn = st.button("▶ Fetch from arXiv", use_container_width=True)

        if fetch_btn:
            if not arxiv_input.strip():
                st.warning("Enter at least one arXiv ID.")
            else:
                vectorstore = get_vectorstore()
                ids = [x.strip() for x in arxiv_input.strip().splitlines() if x.strip()]

                for arxiv_id in ids:
                    with st.spinner(f"Loading arXiv:{arxiv_id} via LangChain ArxivLoader…"):
                        try:
                            # LangChain ArxivLoader — fetches title, authors, abstract automatically
                            raw_docs = load_arxiv(arxiv_id)

                            for doc in raw_docs:
                                doc.metadata["paper_name"] = doc.metadata.get("Title", arxiv_id)
                                doc.metadata["source"] = f"arXiv:{arxiv_id}"

                            chunks = split_documents(raw_docs)
                            vectorstore.add_documents(chunks)

                            title = raw_docs[0].metadata.get("Title", arxiv_id) if raw_docs else arxiv_id
                            authors = raw_docs[0].metadata.get("Authors", "") if raw_docs else ""
                            preview = " ".join(raw_docs[0].page_content.split()[:200]) if raw_docs else ""

                            paper_id = re.sub(r"[^a-z0-9_]", "_", title.lower())[:40]
                            st.session_state.papers[paper_id] = {
                                "title": title,
                                "preview": preview,
                                "source": f"arXiv:{arxiv_id}",
                                "authors": authors,
                            }
                            st.success(f"✅ {title[:70]}")
                        except Exception as e:
                            st.error(f"Failed to fetch {arxiv_id}: {e}")
                st.rerun()

    # Papers knowledge base
    if st.session_state.papers:
        st.divider()
        st.subheader("📚 Papers in Knowledge Base")
        for pid, info in st.session_state.papers.items():
            with st.expander(f"📄 {info['title']}"):
                st.write(f"**Source:** {info.get('source', '—')}")
                if info.get("authors"):
                    st.write(f"**Authors:** {info['authors']}")
                st.write("**Preview:**")
                st.caption(info.get("preview", "")[:400] + "…")


# TAB 2 — CHAT (LangGraph RAG pipeline)
with tab2:
    st.subheader("💬 Chat with Your Papers")
    st.caption("Powered by LangGraph: retrieve → grade → generate")

    if not st.session_state.papers:
        st.info("👆 Go to **Ingest Papers** tab and add some papers first.")
    else:
        # Suggested starter questions
        if not st.session_state.chat_history:
            st.write("**Try asking:**")
            suggestions = [
                "What are the main contributions of each paper?",
                "Which methods are commonly used across papers?",
                "What are the key research gaps identified?",
                "How do the evaluation metrics compare?",
            ]
            c1, c2 = st.columns(2)
            for i, sug in enumerate(suggestions):
                with (c1 if i % 2 == 0 else c2):
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        st.session_state["_pending_q"] = sug
            st.divider()

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        question = st.chat_input("Ask a question about your papers…")

        # Handle suggestion button click
        if "_pending_q" in st.session_state:
            question = st.session_state.pop("_pending_q")

        if question:
            # Show user message
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            # Run LangGraph pipeline
            with st.chat_message("assistant"):
                with st.spinner("Running LangGraph pipeline: retrieve → grade → generate…"):
                    rag_graph = get_rag_graph()
                    result = rag_graph.invoke({
                        "question": question,
                        "documents": [],
                        "generation": "",
                        "paper_names": list(st.session_state.papers.keys()),
                    })
                    answer = result["generation"]

                st.write(answer)

                # Show which documents were used (transparency)
                if result.get("documents"):
                    with st.expander("📎 Sources used"):
                        for doc in result["documents"]:
                            src = doc.metadata.get("source", doc.metadata.get("paper_name", "Unknown"))
                            page = doc.metadata.get("page", "")
                            st.caption(f"• {src}" + (f" — page {page}" if page else ""))

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

# TAB 3 — RELATIONSHIP GRAPH
with tab3:
    st.subheader("🔗 Research Relationship Graph")
    st.caption("LangChain LCEL chain analyses how papers relate to each other.")

    if len(st.session_state.papers) < 2:
        st.info("Add at least **2 papers** to analyse relationships.")
    else:
        if st.button("🔍 Analyse Relationships"):
            with st.spinner("Running LangChain relationship analysis…"):
                rels = analyse_relationships(st.session_state.papers)
                st.session_state.relationships = rels
            st.success(f"Found {len(rels)} relationship(s)!")

        if st.session_state.relationships:
            st.divider()

            # Legend
            st.write("**Legend:**")
            l1, l2, l3, l4 = st.columns(4)
            l1.info("🔵 Builds On")
            l2.error("🔴 Contradicts")
            l3.success("🟢 Uses Method")
            l4.warning("🟡 Similar Approach")

            # Interactive graph
            st.divider()
            try:
                html = build_graph_html(st.session_state.papers, st.session_state.relationships)
                st.components.v1.html(html, height=520, scrolling=False)
            except Exception as e:
                st.error(f"Graph error: {e}")

            # Relationship details
            st.divider()
            st.subheader("Detected Relationships")
            icons = {
                "builds_on": "🔵",
                "contradicts": "🔴",
                "uses_method": "🟢",
                "similar_approach": "🟡",
            }
            for rel in st.session_state.relationships:
                src = st.session_state.papers.get(rel.get("source", ""), {}).get("title", rel.get("source", "?"))
                tgt = st.session_state.papers.get(rel.get("target", ""), {}).get("title", rel.get("target", "?"))
                icon = icons.get(rel.get("relation", ""), "⚪")
                label = rel.get("relation", "").replace("_", " ").title()
                with st.expander(f"{icon} {src[:38]} → {label} → {tgt[:38]}"):
                    c1, c2 = st.columns(2)
                    c1.write(f"**From:** {src}")
                    c2.write(f"**To:** {tgt}")
                    st.write(f"**Relation:** {label}")
                    st.info(rel.get("reason", "No reason provided."))


# TAB 4 — DEEP ANALYSIS
with tab4:
    st.subheader("📊 Deep Research Analysis")
    st.caption("Structured analysis using the full LangGraph RAG pipeline with broader context.")

    if not st.session_state.papers:
        st.info("Add papers first to run analysis.")
    else:
        analysis_type = st.selectbox("Choose an analysis type", [
            "Common methods & techniques across all papers",
            "Research gaps and open problems",
            "Timeline of contributions (who built on what)",
            "Contradictions and disagreements between papers",
            "Practical applications and use-cases",
        ])

        if st.button("▶ Run Analysis"):
            query_map = {
                "Common methods & techniques across all papers":
                    "What methods, techniques, and algorithms are used across these papers? Identify commonalities and differences.",
                "Research gaps and open problems":
                    "What research gaps, limitations, and open problems are identified across these papers?",
                "Timeline of contributions (who built on what)":
                    "Describe the progression of ideas across these papers. Which papers build on earlier work and how?",
                "Contradictions and disagreements between papers":
                    "What claims, findings, or conclusions do these papers disagree about or contradict each other on?",
                "Practical applications and use-cases":
                    "What real-world applications, use-cases, and deployment scenarios are mentioned or implied by these papers?",
            }

            with st.spinner(f"Running LangGraph analysis: {analysis_type}…"):
                rag_graph = get_rag_graph()
                result = rag_graph.invoke({
                    "question": query_map[analysis_type],
                    "documents": [],
                    "generation": "",
                    "paper_names": list(st.session_state.papers.keys()),
                })

            st.divider()
            st.subheader(f"Results: {analysis_type}")
            st.write(result["generation"])

            if result.get("documents"):
                with st.expander("📎 Sources used"):
                    for doc in result["documents"]:
                        src = doc.metadata.get("source", doc.metadata.get("paper_name", "Unknown"))
                        page = doc.metadata.get("page", "")
                        st.caption(f"• {src}" + (f" — page {page}" if page else ""))
