from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# .env 파일에서 API 불러올 수 있도록 지정
load_dotenv()

# .env 파일 안에 <OPENAI_API_KEY = "API키로 바꿔서 기입"> 넣어서 저장
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file.")

os.environ["OPENAI_API_KEY"] = api_key

st.set_page_config(
    page_title="yo-plan",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# gpt가 학습할 문서들 기입
pdf_files = [
    r"C:\Users\csh16\Desktop\2024-2\졸업프로젝트\dataset\kt_membership_benefits_updated.csv"
]

# Load documents from all PDF files
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# Process documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 해당 봇의 말투, 사용목적, 당부 사항 기입
system_template = """너의 이름은 요봇이야.
            너는 항상 한글과 존댓말을 하는 챗봇이야.
            입력받은 통신사의 혜택을 읽고 사용자 소비 패턴에 가장 최적화된, 혜택이 많은 통신사를 알려줘.
----------------
{summaries}

You MUST answer in Korean and in Markdown format:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

st.subheader('질문을 적어 주세요')

def generate_response(input_text):
    result = chain(input_text)
    return result['answer']

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "질문을 적어 주세요 무엇을 도와 드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


# 코드 수정이 끝나고 terminal에서 streamlit run <해당 파일 이름> 적어주면 됩니당