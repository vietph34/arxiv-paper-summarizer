import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import ArxivLoader, TextLoader
template = """Summarize this text: {text}

"""
prompt = PromptTemplate(input_variables=['text'], template=template)

def load_llm():
    llm = OpenAI(temperature=0.5)
    chain = load_summarize_chain(llm,"map_reduce")
    return llm, chain
llm, chain = load_llm()
st.set_page_config(page_title="Arxiv Paper Summarizer", page_icon="🦃")
st.header("Arxiv Paper Summarizer")

col1, col2 = st.columns(2)

with col1:
    st.write("This tool will help you summarize any arxiv paper. This tool is powered by Langchain.")
with col2:
    st.image(image="./ArXiv_logo_2022.svg.png")


# col1, col2 = st.columns(2)
# with col1:
#     option_tone = st.selectbox("Which tone woudl you like your email to have?",
#                                ("Formal", "Informal"))
# with col2:
#     option_dialect = st.selectbox("Which English Dialect would you like?", 
#                                   ("American English", "British English"))

def get_text():
    input_text = st.text_area(label="", placeholder="Arxiv ID...", key="id_input")
    return input_text

st.markdown("## Enter The Arxiv ID")
article_id = get_text()
st.markdown("### Your Summarized Version")
if article_id:
    if article_id.startswith("arxiv"):
        article_id = article_id.split()[1]
        loader = ArxivLoader(query=article_id)
        docs = loader.load_and_split()
        summary = chain.run(docs)
    else:
        model_input = prompt.format(text=article_id)
        docs = model_input
        summary = llm(docs)
    # formated_email = llm(model_input)
    st.write(summary)

