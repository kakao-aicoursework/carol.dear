"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
from langchain.text_splitter import CharacterTextSplitter
from pcconfig import config
import openai
import os.path
import pynecone as pc
import chromadb
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
# from chatbot import style
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma

from langchain.prompts.chat import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from pynecone.base import Base

import os

import pandas as pd

os.environ["OPENAI_API_KEY"] = open("../appkey.txt", "r").read()
openai.api_key = open("../appkey.txt", "r").read()

DATA_DIR = os.path.dirname(os.path.abspath('datas'))
INTENT_PROMPT_TEMPLATE = os.path.join("prompts", "parse_intent.txt")

CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


def formalize_data(file_path):
    # 데이터 준비
    # 인덱스
    ids = []
    # 메타데이터
    doc_meta = []
    # 벡터로 변환 저장할 텍스트 데이터로 ChromaDB에 Embedding 데이터가 없으면 자동으로 벡터로 변환해서 저장
    documents = []

    long_string = open(file_path).read()
    sections = long_string.strip().split('\n')

    # 각 섹션을 분석하여 'menu'와 'data' 열에 데이터를 저장할 리스트 생성
    menu = []
    data = []

    current_menu = None
    current_data = []

    # 섹션을 순회하면서 데이터를 분류
    for section in sections:
        if section.startswith("#"):  # 메뉴 정보인 경우
            if current_menu:  # 현재 메뉴가 존재하면 저장
                menu.append(current_menu)
                data.append("\n".join(current_data))  # 데이터 항목을 개행문자로 구분하여 저장
            current_menu = section[1:].strip()  # '#' 제거하고 공백 제거
            current_data = []  # 새로운 데이터 항목 초기화
        else:  # 데이터 항목인 경우
            current_data.append(section.strip())

    # 마지막 메뉴와 데이터 저장
    if current_menu:
        menu.append(current_menu)
        data.append("\n".join(current_data))

    # DataFrame 생성
    df = pd.DataFrame({'menu': menu, 'data': data})

    for idx in range(len(df)):
        item = df.iloc[idx]
        id = item['menu'].lower().replace(' ', '-')
        document = f"{item['menu']}: {item['data']}"

        ids.append(id)
        # doc_meta.append(meta)
        documents.append(document)

    return (ids, documents)


def upload_embedding_from_file(file_path):
    documents = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


client = chromadb.PersistentClient()

collection = client.get_or_create_collection(
    name="kakao_data",
    metadata={"hnsw:space": "cosine"}
)


def create_collection():
    (ids, documents) = formalize_data(os.path.join("datas", "project_data_카카오소셜.txt"))
    collection.add(documents=documents, ids=ids)
    (ids, documents) = formalize_data(os.path.join("datas", "project_data_카카오싱크.txt"))
    collection.add(documents=documents, ids=ids)
    (ids, documents) = formalize_data(os.path.join("datas", "project_data_카카오톡채널.txt"))
    collection.add(documents=documents, ids=ids)


create_collection()


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


llm = ChatOpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo")
chain = create_chain(llm, os.path.join("prompts", "default_response.txt"), "output")

parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)


def query_db(query: str) -> list[str]:
    vector_res = collection.query(
        query_texts=[query],
        n_results=10,
    )

    srchres = []
    for v in vector_res['documents'][0]:
        item = v.split(':')
        srchres.append({
            "menu": item[0].strip(),
            "data": item[1].strip()
        })

    return srchres


INTENT_LIST_TXT = os.path.join(DATA_DIR, "prompts/intent_list.txt")


def generate_answer(user_message) -> dict[str, str]:
    print('generate_answer')

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)

    intent = parse_intent_chain(context)["intent"]
    print("===============intent : " + intent)

    context["related_documents"] = query_db(context["user_message"])
    answer = ""

    for step in [chain]:
        context = step(context)
        answer += context[step.output_key]
        answer += "\n\n"

    return {"answer": answer}


class Message(Base):
    original_text: str
    text: str
    # created_at: str
    # to_lang: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []
    answer = "답변"

    def output(self):
        if not self.text.strip():
            return "Answer will appear here."
        self.answer = generate_answer(
            self.text)['answer']
        # return answer

    def post(self):
        self.output()
        self.messages = self.messages + [
            Message(
                original_text=self.text,
                text=self.answer,
            )
        ]


# Define views.
def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("챗봇 🗺", font_size="2rem"),
        pc.text(
            "Translate things and post them as messages!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        pc.input(
            placeholder="chat bot",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("Post", on_click=State.post, margin_top="1rem"),

        # output(),
        padding="2rem",
        max_width="1600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Translator")
app.compile()
