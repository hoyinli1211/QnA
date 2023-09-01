from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

#Sidebar
st.sidebar.title("Instructions:")
st.sidebar.markdown("1. ")
st.sidebar.markdown("2. ")
st.sidebar.markdown("3. ")
st.sidebar.markdown("4. ")

#Main Page
st.title("Streamlit Question Answering App")
tabs = st.tabs(["Note","Main"])

tab_note = tabs[0]
tab_main = tabs[1]

with tab_main:
  question_input = st.text_input("Question:")

  if question_input:
    keywords = question_input.split()

    wikipedia = WikipediaAPIWrapper()
    context_input = wikipedia.run(' '.join(keywords))

    QA_input = {
        'question': question_input,
        'context': context_input
    }

    res = nlp(QA_input)

    st.text_area("Answer:", res['answer'])
    st.write("Score:", res['score']) 
