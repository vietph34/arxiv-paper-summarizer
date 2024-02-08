import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import ArxivLoader
template = """Article: { text }
You will generate increasingly concise, entity-dense summaries of the
above article.
Repeat the following 2 steps 5 times.
Step 1. Identify 1-3 informative entities (";" delimited) from the article
which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every
entity and detail from the previous summary plus the missing entities.
A missing entity is:
- relevant to the main story,
- specific yet concise (5 words or fewer),
- novel (not in the previous summary),
- faithful (present in the article),
- anywhere (can be located anywhere in the article).
Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly
non-specific, containing little information beyond the entities marked
as missing. Use overly verbose language and fillers (e.g., "this article
discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and
make space for additional entities.
- Make space with fusion, compression, and removal of uninformative
phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained,
i.e., easily understood without the article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made,
add fewer new entities.
Remember, use the exact same number of words for each summary.
Answer in JSON. The JSON should be a list (length 5) of dictionaries whose
keys are "Missing_Entities" and "Denser_Summary".
"""
prompt = PromptTemplate(input_variables=['text'], template=template)

def load_llm():
    llm = OpenAI(temperature=0.5)
    chain = load_summarize_chain(llm,"map_reduce")
    return chain
chain = load_llm()
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
    input_text = st.text_area(label="", placeholder="Your Email...", key="email_input")
    return input_text

st.markdown("## Enter The Arxiv ID")
article_id = get_text()
st.markdown("### Your Summarized Version")
if article_id:
    loader = ArxivLoader(query=article_id)
    docs = loader.load_and_split()
    summary = chain.run(docs)
    # formated_email = llm(model_input)
    st.write(summary)

