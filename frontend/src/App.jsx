import { useState, useRef, useEffect } from "react";
import "./App.css";

// In production, set VITE_API_URL in your hosting platform's env vars
// to your deployed backend's URL (e.g. https://your-app.onrender.com).
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (overrideText) => {
    const text = overrideText ?? input;
    if (!text.trim()) return;
    const userMessage = { role: "user", text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await response.json();
      setMessages((prev) => [...prev, { role: "bot", text: data.reply }]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "bot", text: "Sorry, I couldn't reach the server." }]);
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>

      <div className="chat-container">
        <div className="header">
          <div className="sparkle sparkle-1">✨</div>
          <div className="sparkle sparkle-2">🎉</div>
          <div className="sparkle sparkle-3">🍿</div>
          <h1>🎬 BoredBuster 🗺️</h1>
          <p>Feeling bored? Tell me your mood — I'll recommend a movie or a place to go!</p>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">👋</div>
              <p>Hey! Try saying:</p>
              <div className="suggestions">
                <span onClick={() => sendMessage("I want a funny movie")}>🎬 "I want a funny movie"</span>
                <span onClick={() => sendMessage("I'm bored, where can I go out?")}>🗺️ "Where can I go out?"</span>
                <span onClick={() => sendMessage("something scary to watch")}>👻 "Something scary to watch"</span>
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              {msg.role === "bot" && <div className="avatar bot-avatar">🎬</div>}
              <div className="bubble">{msg.text}</div>
              {msg.role === "user" && <div className="avatar user-avatar">🙂</div>}
            </div>
          ))}
          {loading && (
            <div className="message bot">
              <div className="avatar bot-avatar">🎬</div>
              <div className="bubble typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-bar">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Tell me your mood..."
          />
          <button onClick={() => sendMessage()} disabled={loading}>
            <span>Send</span> 🚀
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
