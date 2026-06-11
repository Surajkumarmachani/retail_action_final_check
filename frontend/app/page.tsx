// app/page.tsx – entry point: Firebase auth gate → webcam predictor
"use client";

import { useEffect, useState } from "react";
import { onAuthStateChanged, User } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { AuthGate } from "@/components/AuthGate";
import { WebcamPredictor } from "@/components/WebcamPredictor";

export default function Home() {
  // undefined = loading, null = signed-out, User = signed-in
  const [user, setUser] = useState<User | null | undefined>(undefined);

  useEffect(() => {
    return onAuthStateChanged(auth, (u) => setUser(u));
  }, []);

  if (user === undefined) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-950">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-700 border-t-blue-400" />
      </main>
    );
  }

  if (!user) return <AuthGate />;

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <WebcamPredictor />
    </main>
  );
}
