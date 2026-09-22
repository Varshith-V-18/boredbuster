import { useState, useRef, useEffect } from "react";
import "./App.css";

// The bot's replies come back as plain text but use a bit of markdown —
// **bold**, [label](url) links, and bare https:// URLs (e.g. the BookMyShow
// / Google Maps links from the backend tools). The bubble previously just
// dumped msg.text as a raw string, so none of that ever became clickable —
// links just sat there as text a user had to manually select and copy.
// This turns those specific patterns into real <a>/<strong> elements while
// leaving all other text (and existing newlines, which CSS white-space:
// pre-wrap already renders as line breaks) untouched.
function renderMessageText(text) {
  const pattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|\*\*([^*]+)\*\*|(https?:\/\/[^\s)]+)/g;
  const nodes = [];
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const [, linkLabel, linkUrl, boldText, bareUrl] = match;
    if (linkUrl !== undefined) {
      nodes.push(
        <a key={key++} href={linkUrl} target="_blank" rel="noopener noreferrer">
          {linkLabel}
        </a>
      );
    } else if (boldText !== undefined) {
      nodes.push(<strong key={key++}>{boldText}</strong>);
    } else if (bareUrl !== undefined) {
      nodes.push(
        <a key={key++} href={bareUrl} target="_blank" rel="noopener noreferrer">
          {bareUrl}
        </a>
      );
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

// In production, set VITE_API_URL in your hosting platform's env vars
// to your deployed backend's URL (e.g. https://your-app.onrender.com).
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [location, setLocation] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Ask the browser for the user's location once, on load, so recommended
  // places can be real nearby spots instead of a generic list. If the user
  // denies permission (or their browser doesn't support it), we just quietly
  // proceed without it — nothing else breaks.
  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        });
      },
      () => {
        setLocation(null);
      },
      { timeout: 8000 }
    );
  }, []);

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
        body: JSON.stringify({
          message: text,
          history: messages.map(({ role, text }) => ({ role, text })),
          ...(location
            ? { latitude: location.latitude, longitude: location.longitude }
            : {}),
        }),
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
              <div className="bubble">{msg.role === "bot" ? renderMessageText(msg.text) : msg.text}</div>
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
