from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda , RunnableParallel
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "you are a great prodouct reviewer ."),
        ( "user" , "gave me the features about this prodouct {prodouct}")
    ]
)

def get_positive(feature):
    pos_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "you are a great prodouct reviewer ."),
        ( "user" , f"gave me the positive features about this prodouct {feature} . in a csv manner")
    ]
    )           
    return pos_template.format_prompt(feature = feature)

def get_negitive(feature):
    neg_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "you are a great prodouct reviewer ."),
        ( "user" , f"gave me the negitive features about this prodouct {feature} . in a csv manner")
    ]
    )           
    return neg_template.format_prompt(feature = feature)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

pos_branch_chain = (
    RunnableLambda(lambda x : get_positive(x) )| llm | StrOutputParser()
    )

neg_branch_chain = (
    RunnableLambda(lambda x : get_negitive(x)) | llm | StrOutputParser()
    )

chain = (
    prompt_template | llm | StrOutputParser() |
    RunnableParallel(branches = {"pros" : pos_branch_chain , "cons" : neg_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"] , x["branches"]["cons"] ))
    | StrOutputParser()
)

print(chain.invoke( {"prodouct" : "apple macbook pro m2"}))