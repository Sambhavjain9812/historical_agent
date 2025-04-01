import os
import random
import smtplib
import redis
import re
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
smtp_email = os.getenv("SMTP_EMAIL")
smtp_password = os.getenv("SMTP_PASSWORD")
redis_url = os.getenv("REDIS_URL")

if not all([gemini_api_key, smtp_email, smtp_password, redis_url]):
    st.error("Please set all required environment variables.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
redis_client = redis.from_url(redis_url, decode_responses=True)

def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

class AgentState(BaseModel):
    messages: list[HumanMessage | AIMessage] = []
    email: str | None = None
    otp_verified: bool = False
    otp_sent: bool = False
    user_exiting: bool = False
    otp_attempts: int = 0  

    def model_copy(self, update: dict):
        state_dict = self.model_dump()
        state_dict.update(update)
        return AgentState(**state_dict)

def send_otp_email(email: str, otp: str):
    st.write(f"Sending OTP to {email}...")
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_email
        msg["To"] = email
        msg["Subject"] = "Your OTP Code for Historical Agent Bot"
        body = f"Your  requested OTP is: {otp}. It is valid for 5 minutes."
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, email, msg.as_string())
    except Exception as e:
        st.error(f"[ERROR] Email send failed: {e}")
def send_summary_email(email: str, summary: str):
    print("Starting send_summary_email function...")
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_email
        msg["To"] = email
        msg["Subject"] = "Your Historical Chat Summary"
        body = f"Here is the summary of your conversation:\n\n{summary}"
        msg.attach(MIMEText(body, "plain"))
        print("Email message created successfully.")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            print("Connected to SMTP server. Starting TLS...")
            server.starttls()
            print("TLS started. Logging in...")
            server.login(smtp_email, smtp_password)
            print("Logged in successfully. Sending email...")
            server.sendmail(smtp_email, email, msg.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print("Error in send_summary_email:", e)
        raise Exception(f"Failed to send summary email: {e}")

def detect_exit(state: AgentState) -> AgentState:
    if not state.messages:
        return state
    user_message = state.messages[-1].content.lower()
    if user_message in ["exit", "summary", "end"]:
        return state.model_copy(update={"user_exiting": True})
    system_prompt = (
        "Classify the user's intent based on their message.\n"
        "Respond with 'EXIT' if the user wants to exit (e.g., goodbye, quit, thanks, stop), "
        "otherwise respond with 'CONTINUE'.\n"
        f"User: {user_message}\n"
        "Response:"
    )
    response = model.generate_content(system_prompt).text.strip()
    return state.model_copy(update={"user_exiting": response == "EXIT"})

def generate_response(state: AgentState) -> AgentState:
    if state.user_exiting:
        return state.model_copy(update={
            "messages": state.messages + [AIMessage(content="It seems you want to exit. Would you like to share you email with us? if not just type end")]
        })
    system_prompt = "You are a knowledgeable AI assistant specialized in historical monuments worldwide. Engage in a friendly and informative conversation, providing historically relevant recommendations. Keep responses concise yet helpful. If the user mentions travel, suggest historical places near their location.If the user expresses interest in a monument, share key details (e.g., location, significance, travel tips). When appropriate, politely ask for their email to send additional information but respect their decision if they decline. Ensure the conversation remains relevant to historical topics and avoid unrelated discussions.Example interaction: User: I am visiting Noida next month. Any recommendations? Bot: Noida itself has modern attractions, but if you're interested in history, you must visit the Taj Mahal in Agra. It's about 200km away, and you can take a cab easily.User: Sounds great!Bot: Would you like me to send more details to your email? It’ll include travel tips and historical insights.User: No, thanks. I’m in a hurry.Bot: No problem! Let me know if you need more suggestions. Have a great trip"
    conversation_history = system_prompt + "\n" + "\n".join(
        [f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in state.messages]
    )
    response = model.generate_content(conversation_history)
    return state.model_copy(update={"messages": state.messages + [AIMessage(content=response.text)]})

def store_email(state: AgentState) -> AgentState:
    email = next((msg.content.strip() for msg in reversed(state.messages) if "@" in msg.content), None)
    if email:
        return state.model_copy(update={"email": email})
    return state.model_copy(update={
        "messages": state.messages + [AIMessage(content="Please provide your email to receive more details.")]
    })

def generate_otp(state: AgentState) -> AgentState:
    if not state.email or state.otp_sent:
        return state
    otp = "".join([str(random.randint(0, 9)) for _ in range(6)])
    redis_client.setex(f"otp:{state.email}", 300, otp)
    send_otp_email(state.email, otp)
    return state.model_copy(update={
        "otp_sent": True,
        "otp_attempts": 0,
        "messages": state.messages + [AIMessage(content="A 6-digit OTP has been sent to your email. Please verify.")]
    })

def verify_otp(state: AgentState) -> AgentState:
    user_otp = state.messages[-1].content.strip() if state.messages and isinstance(state.messages[-1], HumanMessage) else None
    stored_otp = redis_client.get(f"otp:{state.email}") if state.email else None
    if user_otp and stored_otp and user_otp == stored_otp:
        redis_client.delete(f"otp:{state.email}")
        return state.model_copy(update={
            "otp_verified": True,
            "messages": state.messages + [AIMessage(content="OTP verified successfully!")]
        })
    new_attempts = state.otp_attempts + 1
    if new_attempts >= 3:
        return state.model_copy(update={
            "otp_attempts": 0,
            "otp_sent": False,
            "messages": state.messages + [AIMessage(content="Too many incorrect attempts. Sending a new OTP...")]
        })
    return state.model_copy(update={
        "otp_attempts": new_attempts,
        "messages": state.messages + [AIMessage(content=f"Invalid OTP. {3 - new_attempts} attempts left. Please try again.")]
    })
    
def handle_exit(state: AgentState) -> AgentState:
    user_message = state.messages[-1].content.lower().strip() if state.messages else ""
    print(f"[DEBUG] In handle_exit, user_message: '{user_message}'")
    if user_message == "summary":
        conversation_text = "\n".join([msg.content for msg in state.messages if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)])
        system_prompt = "Summarize the following historical discussion into a concise and informative paragraph:"
        summary_prompt = f"{system_prompt}\n\n{conversation_text}\n\nSummary:"
        summary = model.generate_content(summary_prompt).text.strip()
        print(f"[DEBUG] AI-generated Summary: {summary[:100]}...")
        if state.email:
            try:
                send_summary_email(state.email, summary)
                return state.model_copy(update={"messages": state.messages + [AIMessage(content="A concise summary has been sent to your email. Let me know if you need anything else!")]})
            except Exception as e:
                print(f"[DEBUG] Exception when sending summary email: {e}")
                return state.model_copy(update={"messages": state.messages + [AIMessage(content=f"Failed to send summary. Error: {e}")]})
        else:
            return state.model_copy(update={"messages": state.messages + [AIMessage(content="You haven't provided an email. Please enter your email to receive the summary.")]})
    elif user_message == "end":
        return state.model_copy(update={"messages": state.messages + [AIMessage(content="Okay, ending the chat. Goodbye!")]} )
    
    else:
        return state.model_copy(update={"messages": state.messages + [AIMessage(content="It seems you want to exit. Would you like a historical summary of our conversation or simply end the chat? Please type 'summary' for a summary or 'end' to exit. Also, if you haven’t shared your email yet, please provide it now for future updates on historical monuments.")]})

def create_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("detect_exit", detect_exit)
    graph.add_node("generate_response", generate_response)
    graph.add_node("store_email", store_email)
    graph.add_node("generate_otp", generate_otp)
    graph.add_node("verify_otp", verify_otp)
    graph.add_node("handle_exit", handle_exit)
    graph.set_entry_point("detect_exit")
    graph.add_conditional_edges(
        "detect_exit",
        lambda state: (
            "handle_exit" if state.user_exiting
            else "verify_otp" if state.otp_sent and not state.otp_verified
            else "store_email" if "@" in (state.messages[-1].content if state.messages else "")
            else "generate_response"
        ),
        {"handle_exit": "handle_exit", "verify_otp": "verify_otp", "store_email": "store_email", "generate_response": "generate_response"}
    )
    graph.add_conditional_edges(
        "store_email",
        lambda state: "generate_otp" if state.email else "generate_response",
        {"generate_otp": "generate_otp", "generate_response": "generate_response"}
    )
    graph.add_conditional_edges(
        "verify_otp",
        lambda state: (
            END if state.otp_verified 
            else "store_email"
        ),
        {"store_email": "store_email", END: END}
    )
    return graph

def main():
    st.title("🕌 Historical Agent Chat Bot")
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = AgentState(messages=[])
    if 'graph' not in st.session_state:
        st.session_state.graph = create_graph().compile()
    for msg in st.session_state.agent_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)
    if user_input := st.chat_input("Ask me about historical monuments..."):
        new_state = st.session_state.agent_state.model_copy(update={
            "messages": st.session_state.agent_state.messages + [HumanMessage(content=user_input)]
        })
        result = st.session_state.graph.invoke(new_state)
        if isinstance(result, dict):
            st.session_state.agent_state = AgentState(**result)
        else:
            st.session_state.agent_state = result
        with st.chat_message("user"):
            st.markdown(user_input)
        bot_response = st.session_state.agent_state.messages[-1].content
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    main()
