import { useState, useRef, useEffect } from 'react';
import { Send, Settings, LogOut, Mail, Loader2, Sparkles } from 'lucide-react';

export default function ChatDashboard({ userProfile, onLogout }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: `Hi ${userProfile?.given_name || 'there'}! I'm your AI Gmail Assistant. You can ask me anything about your synced emails.`,
    },
  ]);
  const [input, setInput] = useState('');
  const [isSyncing, setIsSyncing] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuestion = input;
    const newMessage = { id: Date.now(), role: 'user', content: userQuestion };
    setMessages((prev) => [...prev, newMessage]);
    setInput('');

    // Add a temporary loading message
    const loadingId = Date.now() + 1;
    setMessages((prev) => [
      ...prev,
      { id: loadingId, role: 'assistant', content: 'Thinking...', isLoading: true }
    ]);

    try {
      const response = await fetch('http://localhost:8000/api/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          question: userQuestion,
          user_email: userProfile?.email || 'test@example.com' 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch answer');
      }

      const data = await response.json();
      
      setMessages((prev) => 
        prev.map(msg => 
          msg.id === loadingId 
            ? { id: loadingId, role: 'assistant', content: data.answer, sources: data.sources } 
            : msg
        )
      );
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => 
        prev.map(msg => 
          msg.id === loadingId 
            ? { id: loadingId, role: 'assistant', content: "Sorry, I encountered an error connecting to the backend. Is the server running?" } 
            : msg
        )
      );
    }
  };

  const handleSync = async () => {
    setIsSyncing(true);
    try {
      const response = await fetch(`http://localhost:8000/api/sync/initial?user_email=${encodeURIComponent(userProfile?.email || 'test@example.com')}&days_back=5`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Sync failed');
      }
      alert('Sync complete! Your emails have been indexed.');
    } catch (error) {
      console.error('Sync error:', error);
      alert('Failed to sync emails. Is the backend running?');
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 flex-shrink-0 glass-dark border-r border-white/5 flex flex-col p-6 z-10">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-pink-500 flex items-center justify-center shadow-lg shadow-violet-500/20 p-[2px]">
            <div className="w-full h-full bg-slate-900 rounded-full flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-violet-400" />
            </div>
          </div>
          <h1 className="text-xl font-bold text-gradient tracking-tight">AI Inbox</h1>
        </div>

        <div className="bg-slate-900/50 rounded-2xl p-4 mb-6 border border-white/5">
          <div className="flex items-center gap-3 mb-4">
            {userProfile?.picture ? (
              <img src={userProfile.picture} alt="Profile" className="w-12 h-12 rounded-full border-2 border-slate-800" />
            ) : (
              <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center">
                <Mail className="w-5 h-5 text-slate-400" />
              </div>
            )}
            <div className="overflow-hidden">
              <p className="font-medium text-slate-200 truncate">{userProfile?.name || 'User'}</p>
              <p className="text-xs text-slate-500 truncate">{userProfile?.email || 'Logged in via Google'}</p>
            </div>
          </div>
          
          <button
            onClick={handleSync}
            disabled={isSyncing}
            className="w-full py-2.5 px-4 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm rounded-xl font-medium transition-colors border border-white/5 flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isSyncing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-violet-400" />
                <span>Syncing Emails...</span>
              </>
            ) : (
              <>
                <Settings className="w-4 h-4 text-slate-400 group-hover:text-slate-200 transition-colors" />
                <span>Run Initial Sync</span>
              </>
            )}
          </button>
        </div>

        <div className="mt-auto">
          <button 
            onClick={onLogout}
            className="w-full py-3 px-4 text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 rounded-xl text-sm font-medium transition-colors flex items-center gap-2"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative z-0">
        {/* Subtle background glow */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-violet-900/10 via-slate-950 to-slate-950 -z-10" />
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-5 py-3.5 shadow-sm ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-violet-600 to-indigo-600 text-white shadow-violet-900/20'
                      : 'bg-slate-800/80 border border-white/5 text-slate-200'
                  }`}
                >
                  <p className="leading-relaxed text-sm">
                    {msg.isLoading ? (
                      <span className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                      </span>
                    ) : (
                      msg.content
                    )}
                  </p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/10">
                      <p className="text-xs text-slate-400 font-medium mb-2">Sources:</p>
                      <ul className="space-y-1">
                        {msg.sources.map((src, idx) => (
                          <li key={idx} className="text-xs text-slate-500 truncate">
                            • {src.subject} ({src.sender})
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 bg-transparent">
          <div className="max-w-3xl mx-auto">
            <form
              onSubmit={handleSend}
              className="relative flex items-center glass-dark rounded-2xl shadow-xl shadow-black/20 focus-within:ring-2 focus-within:ring-violet-500/50 transition-all p-2"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your emails..."
                className="flex-1 bg-transparent text-slate-200 px-4 py-3 outline-none placeholder:text-slate-500 text-sm"
              />
              <button
                type="submit"
                disabled={!input.trim()}
                className="bg-violet-600 hover:bg-violet-500 text-white p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mr-1 shadow-md shadow-violet-900/20"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <p className="text-center text-xs text-slate-500 mt-4">
              AI Gmail Assistant can make mistakes. Consider verifying important information.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
