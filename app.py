"""
app.py — chatbot Streamlit sur un CSV de votre choix (Lab 1b, Partie 4).

Lancement :  streamlit run app.py

A ADAPTER (section Configuration) : le chemin du CSV et les noms de colonnes.
Exemple de CSV : data/my_dataset.csv avec les colonnes
    title, description, category, price
"""

import csv
from pathlib import Path
from typing import Literal

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration  (<- à adapter à VOS données)
# --------------------------------------------------------------------------

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434"
DATA_PATH = Path("data/my_dataset.csv")

COL_TITLE = "title"            # colonne courte (nom de l'entrée)
COL_TEXT = "description"       # colonne de texte long
COL_CATEGORY = "category"      # colonne catégorielle
COL_NUMBER = "price"           # colonne numérique

MAX_TURNS = 10                 # nombre de messages gardés (hors système)

SYSTEM_PROMPT = (
    "You are an assistant for a small catalogue of entries. "
    "Use your tools for every factual question about the catalogue: never "
    "invent entries. If a question is not about the catalogue, say so briefly "
    "and do not call any tool. Keep answers to two sentences."
)

st.set_page_config(page_title="Mon chatbot", page_icon="💬")


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------

@st.cache_data
def load_rows():
    with open(DATA_PATH, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


ROWS = load_rows()


def _num(value):
    """Convertit en float, ou None si la cellule est vide / invalide."""
    try:
        return float(str(value).replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------
# Schéma Pydantic (extraction structurée depuis une ligne)
# Principe 5 : le raisonnement vient AVANT la décision, car le modèle génère
# les champs dans l'ordre : il "réfléchit" avant de s'engager.
# --------------------------------------------------------------------------

class EntryProfile(BaseModel):
    """Fiche structurée extraite d'une ligne du catalogue."""
    reasoning: str = Field(description="One sentence justifying the audience level.")
    audience: Literal["beginner", "intermediate", "expert"] = Field(
        description="Closed set: who this entry is aimed at."
    )
    main_topic: str = Field(description="The main topic in at most 4 words.")


def extract_profile(row: dict) -> EntryProfile:
    """Extraction typée d'une ligne (Literal = valeurs fermées, validables)."""
    llm = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_URL)
    structured = llm.with_structured_output(EntryProfile)
    return structured.invoke(
        f"Title: {row[COL_TITLE]}\nDescription: {row[COL_TEXT]}"
    )


# --------------------------------------------------------------------------
# Outils — le nom et la docstring SONT le prompt (principe 9)
# --------------------------------------------------------------------------

@tool
def search_data(keyword: str) -> str:
    """Find catalogue entries whose title or description mentions a keyword.
    Use this whenever the user asks what exists on a subject.
    Only knows this catalogue, nothing else."""
    k = (keyword or "").strip().lower()
    if not k:
        return "Empty keyword."
    hits = [r for r in ROWS
            if k in (r[COL_TITLE] or "").lower() or k in (r[COL_TEXT] or "").lower()]
    if not hits:
        return f"No entry mentions '{keyword}'."
    return "\n".join(f"- {r[COL_TITLE]}" for r in hits[:5])


@tool
def count_entries() -> int:
    """Return the total number of entries in the catalogue.
    Use this for questions about the size of the catalogue."""
    return len(ROWS)


@tool
def number_stats(operation: Literal["max", "min", "average"],
                 category: str = "") -> str:
    """Compute the max, min or average of the numeric column, optionally for one
    category. Use this for any question about the highest, lowest or average
    value. Never guess a number: call this tool."""
    rows = ROWS
    if category:
        rows = [r for r in rows
                if (r[COL_CATEGORY] or "").lower() == category.strip().lower()]
        if not rows:
            return f"No entry in category '{category}'."
    pairs = [(_num(r[COL_NUMBER]), r[COL_TITLE]) for r in rows]
    pairs = [(v, t) for v, t in pairs if v is not None]
    if not pairs:
        return "No numeric values found."
    if operation == "average":
        return f"Average: {sum(v for v, _ in pairs) / len(pairs):.2f}"
    best = max(pairs) if operation == "max" else min(pairs)
    return f"{operation}: {best[0]} ({best[1]})"


TOOLBANK = [search_data, count_entries, number_stats]
TOOLS = {t.name: t for t in TOOLBANK}


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------

@st.cache_resource
def get_llm_with_tools():
    llm = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_URL)
    return llm.bind_tools(TOOLBANK)


llm_tools = get_llm_with_tools()


def answer(messages):
    """Un aller-retour : demander, exécuter les outils si besoin, redemander."""
    reply = llm_tools.invoke(messages)
    if not reply.tool_calls:
        return reply, []

    traces = []
    messages = messages + [reply]
    for call in reply.tool_calls:
        try:
            result = TOOLS[call["name"]].invoke(call["args"])   # <- exécution réelle
        except Exception as exc:                                # arguments invalides
            result = f"Tool error: {exc}"
        traces.append(f"{call['name']}({call['args']}) -> {str(result)[:120]}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return llm_tools.invoke(messages), traces


def truncate(messages):
    """Garde le message système + les MAX_TURNS derniers messages."""
    return messages[:1] + messages[1:][-MAX_TURNS:]


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

st.title("💬 Mon chatbot")
st.caption(f"{len(ROWS)} entrées · modèle {MODEL} · en local")

with st.sidebar:
    st.subheader("À propos")
    st.write("Assistant sur un petit catalogue : recherche par mot-clé, "
             "comptage, max / min / moyenne de la colonne numérique.")
    if st.button("Effacer la conversation"):
        st.session_state.pop("messages", None)
        st.rerun()

# L'état de la conversation : la liste de messages du notebook 1a.
if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content=SYSTEM_PROMPT)]

for m in st.session_state["messages"]:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        st.chat_message("assistant").write(m.content)

if question := st.chat_input("Posez votre question"):
    st.chat_message("user").write(question)
    st.session_state["messages"].append(HumanMessage(content=question))
    st.session_state["messages"] = truncate(st.session_state["messages"])

    with st.chat_message("assistant"):
        with st.spinner("..."):
            reply, traces = answer(st.session_state["messages"])
        st.write(reply.content)
        if traces:
            with st.expander("Outils appelés"):
                for t in traces:
                    st.code(t, language=None)

    st.session_state["messages"].append(AIMessage(content=reply.content))
