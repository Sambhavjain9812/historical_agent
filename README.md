# Historical_agent

# 🕌 Historical Agent Chat Bot

A conversational AI bot providing information about historical monuments. The bot verifies user emails via OTP before sending detailed information.

## 🚀 Features
- Conversational AI with historical monument knowledge
- Email verification using OTP
- Supports natural language interactions
- Streamlit-based chat interface
- Uses Redis for OTP storage

## 🛠️ Tech Stack
- **LLM**: Google Gemini API
- **Framework**: Streamlit
- **Database**: Redis
- **Email**: SMTP
- **Orchestration**: LangGraph

## 📌 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/historical-agent-bot.git
cd historical-agent-bot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file and add:
```
GOOGLE_API_KEY=your_gemini_api_key
SMTP_EMAIL=your_smtp_email
SMTP_PASSWORD=your_smtp_password
REDIS_URL=your_redis_url
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

## 🌐 Hosted Version
Access the bot here: [Live Demo](https://historicalagent-pnvdqd7wmp4rqyovhreecr.streamlit.app/)


