"use client";

import { GoogleAuthProvider, createUserWithEmailAndPassword, signInWithEmailAndPassword, signInWithPopup, sendPasswordResetEmail } from "firebase/auth";
import { useState } from "react";
import { auth } from "@/lib/firebase";

// Turn Firebase error codes into something a user (or whoever sets up the project) can act on.
function authErrorMessage(e: unknown): string {
  const code = (e as { code?: string }).code ?? "";
  switch (code) {
    case "auth/unauthorized-domain":
      return `This domain (${window.location.hostname}) is not authorised for sign-in. Add it in Firebase Console → Authentication → Settings → Authorized domains.`;
    case "auth/operation-not-allowed":
      return "This sign-in method is disabled. Enable it in Firebase Console → Authentication → Sign-in method.";
    case "auth/popup-blocked":
      return "The browser blocked the sign-in popup. Allow popups for this site and try again.";
    case "auth/popup-closed-by-user":
    case "auth/cancelled-popup-request":
      return "Sign-in was cancelled.";
    case "auth/invalid-credential":
    case "auth/wrong-password":
    case "auth/user-not-found":
      return "Incorrect email or password.";
    case "auth/email-already-in-use":
      return "An account with this email already exists. Try signing in instead.";
    case "auth/weak-password":
      return "Password should be at least 6 characters.";
    default:
      return (e as Error).message;
  }
}

export function AuthGate() {
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [mode,     setMode]     = useState<"login"|"signup">("login");
  const [error,    setError]    = useState<string|null>(null);
  const [notice,   setNotice]   = useState<string|null>(null);
  const [loading,  setLoading]  = useState(false);

  const submit = async () => {
    setError(null); setNotice(null); setLoading(true);
    try {
      if (mode === "login") await signInWithEmailAndPassword(auth, email, password);
      else await createUserWithEmailAndPassword(auth, email, password);
    } catch (e: unknown) { setError(authErrorMessage(e)); }
    finally { setLoading(false); }
  };

  const googleSignIn = async () => {
    setError(null); setNotice(null); setLoading(true);
    try { await signInWithPopup(auth, new GoogleAuthProvider()); }
    catch (e: unknown) { setError(authErrorMessage(e)); }
    finally { setLoading(false); }
  };

  const resetPassword = async () => {
    setError(null); setNotice(null);
    if (!email.trim()) { setError("Enter your email address first, then click “Forgot password?”"); return; }
    setLoading(true);
    try {
      await sendPasswordResetEmail(auth, email.trim());
      setNotice(`Password reset link sent to ${email.trim()}. Check your inbox (and spam).`);
    } catch (e: unknown) { setError(authErrorMessage(e)); }
    finally { setLoading(false); }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#030712] px-4">
      {/* Background gradient orbs */}
      <div className="pointer-events-none absolute left-1/4 top-1/4 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-green-500/5 blur-3xl" />
      <div className="pointer-events-none absolute right-1/4 bottom-1/4 h-96 w-96 translate-x-1/2 translate-y-1/2 rounded-full bg-cyan-500/5 blur-3xl" />

      <div className="relative w-full max-w-sm">
        {/* Logo */}
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-green-400 to-cyan-500 shadow-[0_0_30px_rgba(34,197,94,0.4)]">
            <svg className="h-7 w-7 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
            </svg>
          </div>
          <h1 className="text-2xl font-black tracking-tight text-white">MACHANI</h1>
          <p className="mt-1 text-sm text-gray-500">Retail Intelligence Platform</p>
        </div>

        <div className="glass rounded-2xl p-6 shadow-2xl">
          {/* Tab toggle */}
          <div className="mb-5 flex rounded-lg bg-gray-900 p-1">
            {(["login","signup"] as const).map(m => (
              <button key={m} onClick={() => { setMode(m); setError(null); setNotice(null); }}
                className={`flex-1 rounded-md py-1.5 text-xs font-semibold tracking-wide transition-all ${
                  mode === m ? "bg-gray-700 text-white shadow-sm" : "text-gray-500 hover:text-gray-300"
                }`}>
                {m === "login" ? "Sign In" : "Sign Up"}
              </button>
            ))}
          </div>

          <div className="space-y-3">
            <input type="email" placeholder="Email address" value={email}
              onChange={e => setEmail(e.target.value)} onKeyDown={e => e.key === "Enter" && submit()}
              className="w-full rounded-lg bg-gray-900 px-4 py-2.5 text-sm text-white placeholder-gray-600 outline-none ring-1 ring-white/10 focus:ring-green-500/50 transition-shadow" />
            <input type="password" placeholder="Password" value={password}
              onChange={e => setPassword(e.target.value)} onKeyDown={e => e.key === "Enter" && submit()}
              className="w-full rounded-lg bg-gray-900 px-4 py-2.5 text-sm text-white placeholder-gray-600 outline-none ring-1 ring-white/10 focus:ring-green-500/50 transition-shadow" />
          </div>

          {mode === "login" && (
            <div className="mt-2 text-right">
              <button onClick={resetPassword} disabled={loading}
                className="text-[11px] font-medium text-gray-500 hover:text-green-400 transition-colors disabled:opacity-50">
                Forgot password?
              </button>
            </div>
          )}

          {error && (
            <div className="mt-3 rounded-lg bg-red-950/50 border border-red-500/20 px-3 py-2 text-xs text-red-400">{error}</div>
          )}
          {notice && (
            <div className="mt-3 rounded-lg bg-green-950/50 border border-green-500/20 px-3 py-2 text-xs text-green-400">{notice}</div>
          )}

          <button onClick={submit} disabled={loading}
            className="mt-4 w-full rounded-lg bg-gradient-to-r from-green-500 to-cyan-500 py-2.5 text-sm font-bold text-black shadow-[0_0_20px_rgba(34,197,94,0.3)] hover:shadow-[0_0_30px_rgba(34,197,94,0.5)] transition-all disabled:opacity-50 active:scale-[0.98]">
            {loading ? "…" : mode === "login" ? "Sign In" : "Create Account"}
          </button>

          <div className="my-4 flex items-center gap-3">
            <div className="flex-1 h-px bg-white/5" />
            <span className="text-[10px] text-gray-600 uppercase tracking-widest">or</span>
            <div className="flex-1 h-px bg-white/5" />
          </div>

          <button onClick={googleSignIn} disabled={loading}
            className="flex w-full items-center justify-center gap-2.5 rounded-lg border border-white/10 py-2.5 text-xs font-medium text-gray-300 hover:bg-white/5 hover:border-white/20 transition-all disabled:opacity-50">
            <svg className="h-4 w-4" viewBox="0 0 24 24">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </button>
        </div>

        <p className="mt-4 text-center text-[10px] text-gray-700">
          Real-time retail behaviour AI powered by YOLOv8 + MediaPipe
        </p>
      </div>
    </div>
  );
}
