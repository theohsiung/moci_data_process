from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from typing import Tuple, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from yfiles_jupyter_graphs import GraphWidget
from neo4j import GraphDatabase
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import warnings
import logging
# 設置只顯示 ERROR，忽略來自 Neo4j 驅動的警告
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    print("s_d", structured_data)
    print("u_d", unstructured_data)
    final_data = f"""整理後的資料如下:
{structured_data}
原始的資料如下:
{"#Document ". join(unstructured_data)}
    """
    return final_data

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        # 查詢該節點的所有出邊關係和鄰居
        outgoing_relationships_response = graph.query(
            """
            MATCH (n {名稱: $entity})-[r]->(neighbor)
            RETURN type(r) AS relationship, neighbor
            """,
            {"entity": entity}
        )
        
        # 查詢該節點的所有入邊關係和鄰居
        incoming_relationships_response = graph.query(
            """
            MATCH (n {名稱: $entity})<-[r]-(neighbor)
            RETURN type(r) AS relationship, neighbor
            """,
            {"entity": entity}
        )

        # 將出邊關係和鄰居資訊追加到 result
        if outgoing_relationships_response:
            # result += f"\n節點 '{entity}' 的出邊關係:\n"
            for record in outgoing_relationships_response:
                relationship = record["relationship"]
                neighbor = record["neighbor"]
                result += f"{entity} - 關係:{relationship} -> {neighbor}\n"
        else:
            result += f"\n節點 '{entity}' 沒有找到出邊關係\n"

        # 將入邊關係和鄰居資訊追加到 result
        if incoming_relationships_response:
            # result += f"\n節點 '{entity}' 的入邊關係:"
            for record in incoming_relationships_response:
                relationship = record["relationship"]
                neighbor = record["neighbor"]
                result += f"{neighbor} -> 關係:{relationship} - {entity}"
        else:
            result += f"\n節點 '{entity}' 沒有找到入邊關係"

        # 新增查詢來檢索該節點的所有屬性
        node_properties_response = graph.query(
            """
            MATCH (n {名稱: $entity})
            RETURN properties(n) AS props
            """,
            {"entity": entity}
        )
        
        # 將節點的屬性資訊追加到 result
        if node_properties_response:
            node_properties = node_properties_response[0]["props"]
            result += f"\n'{entity}' 的屬性: {node_properties}"
        print("結果！！！！",result)
    return result

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All proper nouns, modules, functions, logic, or buttons that appear in the text.",
    )

NEO4J_URI="neo4j+s://758078e8.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD=""

os.environ['OPENAI_API_KEY']=''
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
graph = Neo4jGraph()

# -------------------------------------------------------------------------------------
# # 指定txt文件的路徑
# file_path = 'extract.txt'

# # 打開並讀取文件內容，將其存儲到變數 raw_document 中
# with open(file_path, 'r', encoding='utf-8') as file:
#     raw_documents = file.read()

# # 現在文件的內容已經被存儲到 raw_document 變數中

# len(raw_documents)

# raw_documents[:3]
# raw_documents

# class Document:
#     def __init__(self, metadata, page_content):
#         self.metadata = metadata
#         self.page_content = page_content

# Doc = [Document(metadata={"title":"公版規格書"}, page_content=raw_documents)]

# from langchain.text_splitter import TokenTextSplitter
# text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
# documents = text_splitter.split_documents(Doc)

# -------------------------------------------------------------------------------------
# llm_transformer = LLMGraphTransformer(llm=llm)

# graph_documents = llm_transformer.convert_to_graph_documents(documents)

# graph_documents

# graph.add_graph_documents(
#     graph_documents,
#     baseEntityLabel=True,
#     include_source=True
# )
# -------------------------------------------------------------------------------------

# directly show the graph resulting from the given Cypher query
llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
# llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
default_cypher = "MATCH (s)-[r:]->(t) RETURN s,r,t LIMIT 15"

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="*",
    text_node_properties=["*"],
    embedding_node_property="embedding"
)

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "您正在從文本中提取組織和功能實體.",
        ),
        (
            "human",
            "請使用給定的格式從以下內容中提取信息 "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

# entity_chain.invoke({"question": "WMS公版系統規格書"}).names

_template = """根據以下對話和後續問題，將後續問題重新表述為一個獨立問題，並保留其原始文本。
對話記錄:
{chat_history}
後續問題輸入: {question}
獨立問題:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

template = """只能根據以下提供的內容回覆問題，不能有除此之外的內容！:
{context}

問題: {question}
用口語化的語言，並且詳細回答.
回答:"""

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

prompt = ChatPromptTemplate.from_template(template)
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke(
    {
        "question": "物流中心主檔維護的頁籤的欄位有哪些",
        "chat_history": [("", "")],
    }
))

#TODO 設定chat history 流程
# import json

# # 假設我們的對話記錄格式如下
# chat_history = [("User: Hi", "AI: Hello!"), ("User: How are you?", "AI: I'm fine, thanks!")]

# # 存儲對話記錄
# def save_chat_history(chat_history, filename="chat_history.json"):
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(chat_history, f, ensure_ascii=False, indent=4)

# # 從檔案中載入對話記錄
# def load_chat_history(filename="chat_history.json"):
#     try:
#         with open(filename, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return []

# # 存儲對話記錄
# save_chat_history(chat_history)

# # 載入對話記錄
# loaded_chat_history = load_chat_history()
# print(loaded_chat_history)
