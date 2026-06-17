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
    const response = await axios.get("${process.env.NEXT_PUBLIC_API_URL}/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  };

  const syncUser = async (token: string) => {
    const response = await axios.post(
      "${process.env.NEXT_PUBLIC_API_URL}/sync-user",
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
<main className="min-h-screen bg-[#f5f0e8] text-black flex items-center justify-center px-6 relative overflow-hidden">
  
  {/* Background hatch texture */}
  <div
    className="pointer-events-none absolute inset-0 opacity-[0.04]"
    style={{
      backgroundImage: "repeating-linear-gradient(45deg, #000 0, #000 1px, transparent 0, transparent 50%)",
      backgroundSize: "12px 12px",
    }}
  />

  <div className="relative z-10 w-full max-w-md border-[3px] border-black bg-white shadow-[10px_10px_0px_#000] p-8">

    {/* Logo */}
    <div className="flex items-center gap-3 mb-8">
      <div className="flex size-12 items-center justify-center border-[2px] border-black bg-black shadow-[3px_3px_0px_#000]">
        <Sparkles className="size-6 text-[#c8f135] stroke-[2.5px]" />
      </div>
      <div>
        <h1
          className="uppercase tracking-[0.08em] text-black leading-none"
          style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "26px" }}
        >
          Blog Agent OS
        </h1>
        <p className="font-mono text-[10px] font-black uppercase tracking-widest text-gray-500 mt-0.5">
          Autonomous AI Workspace
        </p>
      </div>
    </div>

    {/* Heading */}
    <div className="mb-8 border-l-[4px] border-black pl-4">
      <h2
        className="uppercase tracking-tight text-black leading-none"
        style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "36px" }}
      >
        {isSignup ? "Create Account" : "Welcome Back"}
      </h2>
      <p className="font-mono text-xs font-bold text-gray-500 mt-1.5">
        {isSignup
          ? "Create your account to start generating blogs."
          : "Sign in to continue building AI-powered blogs."}
      </p>
    </div>

    {/* Error */}
    {error && (
      <div className="mb-4 flex items-center gap-2 border-[2px] border-black bg-[#ff2d78] px-4 py-3 font-mono text-xs font-black uppercase tracking-wider text-white shadow-[3px_3px_0px_#000]">
        {error}
      </div>
    )}

    {/* Success */}
    {success && (
      <div className="mb-4 flex items-center gap-2 border-[2px] border-black bg-[#c8f135] px-4 py-3 font-mono text-xs font-black uppercase tracking-wider text-black shadow-[3px_3px_0px_#000]">
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
        className="w-full border-[2px] border-black bg-[#f5f0e8] px-4 py-3 font-mono text-sm font-bold text-black placeholder:text-gray-400 outline-none focus:shadow-[3px_3px_0px_#000] transition-all"
      />

      <input
        type="password"
        placeholder="Enter your password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="w-full border-[2px] border-black bg-[#f5f0e8] px-4 py-3 font-mono text-sm font-bold text-black placeholder:text-gray-400 outline-none focus:shadow-[3px_3px_0px_#000] transition-all"
      />

      <button
        onClick={handleEmailAuth}
        disabled={loading}
        className="w-full border-[2px] border-black bg-[#ff2d78] py-3 font-mono text-sm font-black uppercase tracking-wider text-white shadow-[4px_4px_0px_#000] transition-all hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000] disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "Loading..." : isSignup ? "Create Account" : "Login"}
      </button>

      {/* Divider */}
      <div className="flex items-center gap-4 py-1">
        <div className="h-[2px] flex-1 bg-black" />
        <span className="font-mono text-[10px] font-black uppercase tracking-widest text-black">or</span>
        <div className="h-[2px] flex-1 bg-black" />
      </div>

      {/* Google */}
      <button
        onClick={handleGoogleLogin}
        disabled={loading}
        className="w-full flex items-center justify-center gap-3 border-[2px] border-black bg-white py-3 font-mono text-sm font-black uppercase tracking-wider text-black shadow-[4px_4px_0px_#000] transition-all hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000] hover:bg-[#fce135] disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <GoogleIcon />
        Continue with Google
      </button>

      {/* Toggle */}
      <div className="text-center pt-2">
        <button
          onClick={() => { setIsSignup(!isSignup); setError(""); setSuccess(""); }}
          className="font-mono text-xs font-black uppercase tracking-wider text-black underline underline-offset-4 hover:text-[#ff2d78] transition-colors"
        >
          {isSignup ? "Already have an account? Login" : "Don't have an account? Sign up"}
        </button>
      </div>
    </div>
  </div>
</main>
  );
}