import { useState } from 'react';
import { useGoogleLogin } from '@react-oauth/google';
import { Sparkles, Mail, Shield, Zap, Loader2 } from 'lucide-react';
import ChatDashboard from './components/ChatDashboard';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userProfile, setUserProfile] = useState(null);
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  const login = useGoogleLogin({
    flow: 'auth_code',
    scope: 'https://www.googleapis.com/auth/gmail.readonly',
    onSuccess: async (codeResponse) => {
      console.log('Auth Code Received:', codeResponse.code);
      setIsLoggingIn(true);
      
      try {
        // Send the auth code to our FastAPI backend to exchange for tokens
        const response = await fetch('http://localhost:8000/api/auth/google', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ code: codeResponse.code }),
        });

        if (!response.ok) {
          throw new Error('Failed to exchange auth code');
        }

        const data = await response.json();
        console.log('Backend Auth Success:', data);
        
        setUserProfile({
          name: data.profile.name || 'Google User',
          email: data.profile.email,
          picture: data.profile.picture,
        });
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Error during backend auth:', error);
        alert('Authentication failed on the backend. Make sure the backend is running and client_secret is configured.');
      } finally {
        setIsLoggingIn(false);
      }
    },
    onError: (errorResponse) => {
      console.error('Login Failed', errorResponse);
      setIsLoggingIn(false);
    },
  });

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserProfile(null);
  };

  if (isAuthenticated) {
    return <ChatDashboard userProfile={userProfile} onLogout={handleLogout} />;
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 relative overflow-hidden z-0">
      {/* Background glow effects */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-slate-600/20 blur-[120px] -z-10"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-slate-600/20 blur-[120px] -z-10"></div>

      <div className="w-full max-w-md">
        <div className="glass-dark rounded-3xl p-10 shadow-2xl shadow-black/50 text-center border border-white/10 relative overflow-hidden">
          
          {/* Subtle top highlight */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>

          <div className="w-16 h-16 mx-auto bg-slate-800 rounded-2xl flex items-center justify-center mb-8 shadow-lg shadow-black/30 border border-white/10">
            <Sparkles className="w-8 h-8 text-slate-300" />
          </div>
          
          <h1 className="text-3xl font-extrabold text-white mb-3 tracking-tight">
            Talk to your <span className="text-gradient">Inbox</span>
          </h1>
          <p className="text-slate-400 mb-10 text-sm leading-relaxed">
            AI-powered semantic search for your Gmail. Find what you need, instantly.
          </p>

          <div className="space-y-4 text-left mb-10">
            <div className="flex items-center gap-3 text-sm text-slate-300">
              <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                <Zap className="w-4 h-4 text-slate-400" />
              </div>
              <p>Instant answers from thousands of emails</p>
            </div>
            <div className="flex items-center gap-3 text-sm text-slate-300">
              <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                <Shield className="w-4 h-4 text-slate-400" />
              </div>
              <p>Private & secure multi-tenant architecture</p>
            </div>
            <div className="flex items-center gap-3 text-sm text-slate-300">
              <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                <Mail className="w-4 h-4 text-slate-400" />
              </div>
              <p>Continuous background synchronization</p>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={() => login()}
              disabled={isLoggingIn}
              className="px-6 py-3 bg-white text-slate-900 rounded-full font-medium hover:bg-slate-100 transition-colors flex items-center gap-3 disabled:opacity-70"
            >
              {isLoggingIn ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Connecting...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                    <path d="M1 1h22v22H1z" fill="none"/>
                  </svg>
                  Continue with Google
                </>
              )}
            </button>
          </div>
          
        </div>
        
        <p className="text-center mt-8 text-xs text-slate-600">
          By continuing, you agree to our Terms of Service and Privacy Policy.
        </p>
      </div>
    </div>
  );
}

export default App;
