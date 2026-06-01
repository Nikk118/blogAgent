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

export default function LoginPage() {
  const [isSignup, setIsSignup] = useState(false);
    const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fetchCurrentUser = async (token: string) => {
    const response = await axios.get(
      "http://localhost:8000/auth/me",
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    );

    return response.data;
  };

  const syncUser = async (token: string) => {
  const response = await axios.post(
    "http://localhost:8000/sync-user",
    {},
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  )

  return response.data
}

  const handleGoogleLogin = async () => {
    try {
      setLoading(true);
      setError("");
      setSuccess("");

      const provider = new GoogleAuthProvider();

      const result = await signInWithPopup(auth, provider);

      const token = await result.user.getIdToken(true);

      const userData = await fetchCurrentUser(token)

console.log("GOOGLE USER:", userData)

const syncedUser = await syncUser(token)

console.log("SYNCED USER:", syncedUser)

      setSuccess("Successfully logged in with Google");

      // redirect later
      router.push("/");

    } catch (error: any) {
      console.error(error);

      setError(
        error?.message || "Something went wrong"
      );

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
        result = await createUserWithEmailAndPassword(
          auth,
          email,
          password
        );
      } else {
        result = await signInWithEmailAndPassword(
          auth,
          email,
          password
        );
      }

      const token = await result.user.getIdToken(true);

      const userData = await fetchCurrentUser(token)

console.log("EMAIL USER:", userData)

const syncedUser = await syncUser(token)

console.log("SYNCED USER:", syncedUser)

      setSuccess(
        isSignup
          ? "Account created successfully"
          : "Login successful"
      );

      // redirect later
      router.push("/");

    } catch (error: any) {
      console.error(error);

      setError(
        error?.message || "Something went wrong"
      );

    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center px-6 relative overflow-hidden">

      {/* Background Glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,#134e4a,transparent_40%)]" />

      {/* Card */}
      <div className="relative z-10 w-full max-w-md rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl p-8 shadow-2xl">

        {/* Logo */}
        <div className="flex items-center gap-3 mb-8">
          <div className="bg-teal-300 text-black p-3 rounded-2xl">
            <Sparkles size={24} />
          </div>

          <div>
            <h1 className="text-2xl font-bold">
              Blog Agent OS
            </h1>

            <p className="text-sm text-zinc-400">
              Autonomous AI Workspace
            </p>
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

          {/* Email */}
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-2xl border border-white/10 bg-black/40 px-4 py-3 outline-none focus:border-teal-300"
          />

          {/* Password */}
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-2xl border border-white/10 bg-black/40 px-4 py-3 outline-none focus:border-teal-300"
          />

          {/* Email Auth Button */}
          <button
            onClick={handleEmailAuth}
            disabled={loading}
            className="w-full rounded-2xl bg-teal-300 text-black font-semibold py-3 transition hover:scale-[1.02] hover:bg-teal-200 disabled:opacity-50"
          >
            {loading
              ? "Loading..."
              : isSignup
              ? "Create Account"
              : "Login"}
          </button>

          {/* Divider */}
          <div className="flex items-center gap-4 py-2">
            <div className="h-px flex-1 bg-white/10" />

            <span className="text-sm text-zinc-500">
              OR
            </span>

            <div className="h-px flex-1 bg-white/10" />
          </div>

          {/* Google Login */}
          <button
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full rounded-2xl bg-white text-black font-medium py-3 transition hover:scale-[1.02] hover:bg-zinc-200 disabled:opacity-50"
          >
            Continue with Google
          </button>

          {/* Toggle */}
          <div className="text-center pt-4">
            <button
              onClick={() => setIsSignup(!isSignup)}
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