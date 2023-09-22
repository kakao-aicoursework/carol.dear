"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
from langchain.text_splitter import CharacterTextSplitter
from pcconfig import config
import openai
import os.path
import pynecone as pc
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import LLMMathChain
from langchain.chat_models import ChatOpenAI
# from chatbot import style
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.tools import Tool

from langchain.prompts.chat import ChatPromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.document_loaders import TextLoader
from pynecone.base import Base

import os

os.environ["OPENAI_API_KEY"] = open("../appkey.txt", "r").read()
openai.api_key = open("../appkey.txt", "r").read()


DATA_DIR = os.path.dirname(os.path.abspath('datas'))

CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"

def upload_embedding_from_file(file_path):
    # loader = TextLoader.get(file_path)
    # if loader is None:
    #     raise ValueError("Not supported file type")
    documents = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


upload_embedding_from_file(os.path.join("datas", "project_data_카카오소셜.txt"))
upload_embedding_from_file(os.path.join("datas", "project_data_카카오싱크.txt"))
upload_embedding_from_file(os.path.join("datas", "project_data_카카오톡채널.txt"))


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


llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

chain = create_chain(llm, os.path.join("prompts", "default_response.txt"), "output")

# parse_intent_chain = create_chain(
#     llm=llm,
#     template_path=os.path.join("prompts", "default_response.txt"),
#     output_key="intent",
# )

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)

_retriever = _db.as_retriever()


def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

INTENT_LIST_TXT = os.path.join(DATA_DIR, "prompts/intent_list.txt")
def generate_answer(user_message) -> dict[str, str]:

    print('generate_answer')

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)

    # intent = parse_intent_chain(context)["intent"]
    # intent = parse_intent_chain.run(context)

    context["related_documents"] = query_db(context["user_message"])
    answer = ""

    for step in [chain]:
        context = step(context)
        answer += context[step.output_key]
        answer += "\n\n"

    print(answer)

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

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        answer = generate_answer(
            self.text)
        return answer

    def post(self):
        # answer = generate_answer(
        #             self.text)
        #
        # self.output()
        self.messages = self.messages + [
            Message(
                original_text=self.text,
                text=self.output,
            )
        ]
        #

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
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Translator")
app.compile()
