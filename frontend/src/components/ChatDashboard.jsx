import { useState, useRef, useEffect } from 'react';
import { Send, Settings, LogOut, Mail, Loader2, Bot, Eye, Info } from 'lucide-react';
import ChunkInspector from './ChunkInspector';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { motion, AnimatePresence } from 'framer-motion';
import * as Tooltip from '@radix-ui/react-tooltip';

// Simulated typewriter effect component
const TypewriterMarkdown = ({ content }) => {
  const [displayedContent, setDisplayedContent] = useState('');

  useEffect(() => {
    let currentLength = 0;
    // Reveal text in chunks to simulate fast streaming
    const interval = setInterval(() => {
      currentLength += Math.max(3, Math.floor(content.length / 50));
      if (currentLength >= content.length) {
        setDisplayedContent(content);
        clearInterval(interval);
      } else {
        setDisplayedContent(content.substring(0, currentLength));
      }
    }, 20);

    return () => clearInterval(interval);
  }, [content]);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ node, ...props }) => <p className="mb-3 last:mb-0 leading-relaxed text-sm text-slate-300" {...props} />,
        ul: ({ node, ...props }) => <ul className="list-disc ml-5 mb-3 space-y-1 text-sm text-slate-300" {...props} />,
        ol: ({ node, ...props }) => <ol className="list-decimal ml-5 mb-3 space-y-1 text-sm text-slate-300" {...props} />,
        li: ({ node, ...props }) => <li {...props} />,
        h1: ({ node, ...props }) => <h1 className="text-lg font-semibold mb-3 mt-4 text-slate-100" {...props} />,
        h2: ({ node, ...props }) => <h2 className="text-base font-semibold mb-2 mt-4 text-slate-100" {...props} />,
        h3: ({ node, ...props }) => <h3 className="text-sm font-semibold mb-2 mt-3 text-slate-100" {...props} />,
        a: ({ node, ...props }) => <a className="text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors" target="_blank" rel="noopener noreferrer" {...props} />,
        strong: ({ node, ...props }) => <strong className="font-semibold text-slate-200" {...props} />,
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <div className="my-4 rounded-lg overflow-hidden border border-slate-700/50">
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
                customStyle={{ margin: 0, background: '#1e1e1e', padding: '1rem', fontSize: '0.875rem' }}
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            </div>
          ) : (
            <code className="bg-slate-800 border border-slate-700/50 px-1.5 py-0.5 rounded-md text-slate-300 text-[0.85em] font-mono" {...props}>
              {children}
            </code>
          );
        }
      }}
    >
      {displayedContent}
    </ReactMarkdown>
  );
};

export default function ChatDashboard({ userProfile, onLogout }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSyncing, setIsSyncing] = useState(false);
  const [hasSynced, setHasSynced] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorTrace, setInspectorTrace] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const buildChatHistory = () => {
    const conversational = messages
      .filter(m => !m.isLoading)
      .slice(-6);

    return conversational.map(m => ({
      role: m.role,
      content: m.content,
    }));
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuestion = input;
    const newMessage = { id: Date.now(), role: 'user', content: userQuestion };
    setMessages((prev) => [...prev, newMessage]);
    setInput('');

    const chatHistory = buildChatHistory();

    const loadingId = Date.now() + 1;
    setMessages((prev) => [
      ...prev,
      { id: loadingId, role: 'assistant', content: '', isLoading: true }
    ]);

    try {
      const response = await fetch('http://localhost:8000/api/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userQuestion,
          user_email: userProfile?.email || 'test@example.com',
          chat_history: chatHistory,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to fetch answer');
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('text/event-stream')) {
        // Future proofing for actual SSE
        // This block would handle the stream if backend supported it
        const data = await response.json(); // fallback for now
        // (Actual streaming logic would go here)
      }

      const data = await response.json();

      setMessages((prev) =>
        prev.map(msg =>
          msg.id === loadingId
            ? {
              id: loadingId,
              role: 'assistant',
              content: data.answer,
              sources: data.sources,
              pipelineTrace: data.pipeline_trace,
              semanticQuery: data.semantic_query,
              metadataFilters: data.metadata_filters,
              isNew: true, // Flag to trigger typewriter effect
            }
            : msg
        )
      );

      if (data.pipeline_trace) {
        setInspectorTrace(data.pipeline_trace);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) =>
        prev.map(msg =>
          msg.id === loadingId
            ? {
              id: loadingId,
              role: 'assistant',
              content: `Sorry, I encountered an error: ${error.message}. Is the server running?`,
              isNew: true
            }
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
      setHasSynced(true);
      alert('Sync complete! Your emails have been indexed.');
    } catch (error) {
      console.error('Sync error:', error);
      alert('Failed to sync emails. Is the backend running?');
    } finally {
      setIsSyncing(false);
    }
  };

  const openInspector = (trace) => {
    setInspectorTrace(trace);
    setInspectorOpen(true);
  };

  return (
    <div className="flex h-screen w-full bg-[#09090b] overflow-hidden text-slate-200">
      {/* Sidebar */}
      <aside className="w-80 flex-shrink-0 bg-slate-950/80 backdrop-blur-xl border-r border-white/5 flex flex-col p-6 z-10">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center border border-white/10 shadow-lg">
            <Bot className="w-5 h-5 text-slate-300" />
          </div>
          <h1 className="text-xl font-medium tracking-tight text-slate-100">AI Inbox</h1>
        </div>

        <div className="bg-slate-900/60 rounded-2xl p-4 mb-6 border border-white/5 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            {userProfile?.picture ? (
              <img src={userProfile.picture} alt="Profile" className="w-12 h-12 rounded-full border border-slate-700" referrerPolicy="no-referrer" />
            ) : (
              <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
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
            className="w-full py-2.5 px-4 bg-slate-100 hover:bg-white text-slate-900 text-sm rounded-xl font-medium transition-colors flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed shadow-sm"
          >
            {isSyncing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
                <span>Syncing Emails...</span>
              </>
            ) : hasSynced ? (
              <>
                <Settings className="w-4 h-4 text-slate-500 group-hover:text-slate-700 transition-colors" />
                <span>New Mails? Re-Sync</span>
              </>
            ) : (
              <>
                <Settings className="w-4 h-4 text-slate-500 group-hover:text-slate-700 transition-colors" />
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
      <main className="flex-1 flex flex-col relative z-0 min-w-0">
        <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
          <div className="max-w-3xl mx-auto space-y-6">

            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center min-h-[50vh] text-center opacity-90 animate-in fade-in slide-in-from-bottom-8 duration-700">
                <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center border border-slate-700/50 mb-6 shadow-xl">
                  <Bot className="w-8 h-8 text-slate-400" />
                </div>
                <h2 className="text-3xl font-medium text-slate-100 mb-3 tracking-tight">
                  Hey, {userProfile?.given_name || userProfile?.name?.split(' ')[0] || 'there'}
                </h2>
                <p className="text-slate-400 text-sm max-w-sm mx-auto">
                  I'm ready to help you search and understand your synced emails. What are you looking for?
                </p>
              </div>
            )}

            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-6 py-4 shadow-sm ${msg.role === 'user'
                        ? 'bg-slate-200 text-slate-900 rounded-br-sm'
                        : 'bg-slate-900/80 border border-white/5 text-slate-200 rounded-bl-sm shadow-xl shadow-black/10'
                      }`}
                  >
                    {msg.isLoading ? (
                      <div className="flex flex-col gap-2.5 py-1 w-48">
                        <div className="flex items-center gap-2 text-xs text-slate-500 font-medium mb-1">
                          <Loader2 className="w-3.5 h-3.5 animate-spin" /> Retrieving context...
                        </div>
                        <div className="h-3 bg-slate-700/50 rounded-md animate-pulse w-full"></div>
                        <div className="h-3 bg-slate-700/50 rounded-md animate-pulse w-5/6"></div>
                        <div className="h-3 bg-slate-700/50 rounded-md animate-pulse w-4/6"></div>
                      </div>
                    ) : (
                      <div className="flex flex-col gap-3">
                        {msg.role === 'user' ? (
                          <p className="leading-relaxed text-sm">{msg.content}</p>
                        ) : (
                          <>
                            {msg.isNew ? (
                              <TypewriterMarkdown content={msg.content} />
                            ) : (
                              <TypewriterMarkdown content={msg.content} /> // We just reuse it, it will render instantly if we wanted, but the effect is fast enough
                            )}

                            {/* Inspect button */}
                            {msg.pipelineTrace && (
                              <div className="mt-2 pt-3 border-t border-white/5 flex justify-between items-center">
                                <button
                                  onClick={() => openInspector(msg.pipelineTrace)}
                                  className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition-colors font-medium"
                                >
                                  <Eye className="w-3.5 h-3.5" />
                                  Inspect Retrieval Trace
                                </button>
                                {msg.pipelineTrace?.timings?.total_ms && (
                                  <span className="text-[10px] text-slate-500 font-mono bg-slate-900/50 px-1.5 py-0.5 rounded">
                                    {msg.pipelineTrace.timings.total_ms}ms
                                  </span>
                                )}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 bg-gradient-to-t from-[#09090b] via-[#09090b] to-transparent pt-10">
          <div className="max-w-3xl mx-auto">
            <form
              onSubmit={handleSend}
              className="relative flex items-center bg-slate-900/90 backdrop-blur-md rounded-2xl shadow-xl shadow-black/40 border border-slate-800 focus-within:border-slate-600 focus-within:bg-slate-900 transition-all p-2"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message AI Inbox..."
                className="flex-1 bg-transparent text-slate-200 px-4 py-3 outline-none placeholder:text-slate-500 text-sm"
              />
              <button
                type="submit"
                disabled={!input.trim()}
                className="bg-slate-100 hover:bg-white text-slate-900 p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mr-1 shadow-sm"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <p className="text-center text-xs text-slate-500 mt-4 font-medium">
              AI assistant can make mistakes. Consider verifying important information.
            </p>
          </div>
        </div>
      </main>

      <ChunkInspector
        trace={inspectorTrace}
        isOpen={inspectorOpen}
        onClose={() => setInspectorOpen(false)}
      />
    </div>
  );
}
