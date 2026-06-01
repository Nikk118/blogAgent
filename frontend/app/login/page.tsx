"use client";

import { useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import {
  GoogleAuthProvider,
  signInWithPopup,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { Sparkles } from "lucide-react";
import { auth } from "@/lib/firebase";

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
      <path d="M3.964 10.707A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.707V4.961H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.039l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.961L3.964 6.293C4.672 4.166 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  );
}

function getFirebaseError(code: string, isSignup: boolean): string {
  switch (code) {
    case "auth/user-not-found":
      return "No account found with this email. Please sign up first.";
    case "auth/wrong-password":
      return "Invalid credentials. Please check your email and password.";
    case "auth/invalid-credential":
      return "Invalid credentials. Please check your email and password.";
    case "auth/email-already-in-use":
      return "An account with this email already exists. Try logging in.";
    case "auth/weak-password":
      return "Password must be at least 6 characters.";
    case "auth/invalid-email":
      return "Please enter a valid email address.";
    case "auth/too-many-requests":
      return "Too many failed attempts. Please try again later.";
    case "auth/popup-closed-by-user":
      return "Google sign-in was cancelled.";
    default:
      return "Something went wrong. Please try again.";
  }
}

export default function LoginPage() {
  const [isSignup, setIsSignup] = useState(false);
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fetchCurrentUser = async (token: string) => {
    const response = await axios.get("http://localhost:8000/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  };

  const syncUser = async (token: string) => {
    const response = await axios.post(
      "http://localhost:8000/sync-user",
      {},
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  };

  const handleGoogleLogin = async () => {
    try {
      setLoading(true);
      setError("");
      setSuccess("");
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      const token = await result.user.getIdToken(true);
      await fetchCurrentUser(token);
      await syncUser(token);
      setSuccess("Successfully logged in with Google");
      router.push("/");
    } catch (error: any) {
      setError(getFirebaseError(error?.code, isSignup));
    } finally {
      setLoading(false);
    }
  };

  const handleEmailAuth = async () => {
    try {
      setLoading(true);
      setError("");
      setSuccess("");
      let result;
      if (isSignup) {
        result = await createUserWithEmailAndPassword(auth, email, password);
      } else {
        result = await signInWithEmailAndPassword(auth, email, password);
      }
      const token = await result.user.getIdToken(true);
      await fetchCurrentUser(token);
      await syncUser(token);
      setSuccess(isSignup ? "Account created successfully" : "Login successful");
      router.push("/");
    } catch (error: any) {
      setError(getFirebaseError(error?.code, isSignup));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center px-6 relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,#134e4a,transparent_60%)]" />

      <div className="relative z-10 w-full max-w-md rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl p-8 shadow-2xl">
        {/* Logo */}
        <div className="flex items-center gap-3 mb-8">
          <div className="bg-teal-300 text-black p-3 rounded-2xl">
            <Sparkles size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Blog Agent OS</h1>
            <p className="text-sm text-zinc-400">Autonomous AI Workspace</p>
          </div>
        </div>

        {/* Heading */}
        <div className="mb-8">
          <h2 className="text-3xl font-semibold">
            {isSignup ? "Create account" : "Welcome back"}
          </h2>
          <p className="text-zinc-400 mt-2">
            {isSignup
              ? "Create your account to start generating blogs."
              : "Sign in to continue building AI-powered blogs."}
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Success */}
        {success && (
          <div className="mb-4 rounded-2xl border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-300">
            {success}
          </div>
        )}

        {/* Form */}
        <div className="space-y-4">
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-2xl border border-white/10 bg-black/40 px-4 py-3 outline-none focus:border-teal-300"
          />

          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-2xl border border-white/10 bg-black/40 px-4 py-3 outline-none focus:border-teal-300"
          />

          <button
            onClick={handleEmailAuth}
            disabled={loading}
            className="w-full rounded-2xl bg-teal-300 text-black font-semibold py-3 transition hover:scale-[1.02] hover:bg-teal-200 disabled:opacity-50"
          >
            {loading ? "Loading..." : isSignup ? "Create Account" : "Login"}
          </button>

          <div className="flex items-center gap-4 py-2">
            <div className="h-px flex-1 bg-white/10" />
            <span className="text-sm text-zinc-500">OR</span>
            <div className="h-px flex-1 bg-white/10" />
          </div>

          <button
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full rounded-2xl bg-white text-black font-medium py-3 transition hover:scale-[1.02] hover:bg-zinc-200 disabled:opacity-50 flex items-center justify-center gap-3"
          >
            <GoogleIcon />
            Continue with Google
          </button>

          <div className="text-center pt-4">
            <button
              onClick={() => { setIsSignup(!isSignup); setError(""); setSuccess(""); }}
              className="text-teal-300 hover:underline"
            >
              {isSignup
                ? "Already have an account? Login"
                : "Don't have an account? Sign up"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}