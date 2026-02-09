"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { signInWithPopup } from "firebase/auth";
import { auth, googleProvider } from "@/lib/firebase";
import { api } from "@/lib/api";

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  status: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  isAdmin: boolean;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signInWithGoogle: () => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
  registerOTP: (email: string, recoveryEmail: string, password: string) => Promise<{ error: Error | null; devOtp?: string }>;
  verifyOTP: (recoveryEmail: string, otp: string) => Promise<{ error: Error | null; userId?: string }>;
  completeRegistration: (recoveryEmail: string, username: string) => Promise<{ error: Error | null }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);

  const fetchCurrentUser = async () => {
    try {
      const { data } = await api.get("/auth/me");
      setUser(data);
      setIsAdmin(data.role === "admin");
    } catch {
      setUser(null);
      setIsAdmin(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCurrentUser();
  }, []);

  const signIn = async (email: string, password: string) => {
    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);

      await api.post("/auth/login", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });

      await fetchCurrentUser();
      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const signInWithGoogle = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const idToken = await result.user.getIdToken();

      await api.post("/auth/google", { id_token: idToken });

      await fetchCurrentUser();
      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const registerOTP = async (email: string, recoveryEmail: string, password: string) => {
    try {
      const { data } = await api.post("/auth/register/otp", {
        email,
        recovery_email: recoveryEmail,
        password,
      });
      return { error: null, devOtp: data.dev_otp };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const verifyOTP = async (recoveryEmail: string, otp: string) => {
    try {
      const { data } = await api.post("/auth/register/verify_otp", {
        recovery_email: recoveryEmail,
        otp,
      });
      return { error: null, userId: data.user?.id };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const completeRegistration = async (recoveryEmail: string, username: string) => {
    try {
      await api.post("/auth/register/complete", {
        recovery_email: recoveryEmail,
        username,
      });
      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const signOut = async () => {
    try {
      await api.post("/auth/logout");
      setUser(null);
      setIsAdmin(false);
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  return (
    <AuthContext.Provider value={{
      user,
      loading,
      isAdmin,
      signIn,
      signInWithGoogle,
      signOut,
      registerOTP,
      verifyOTP,
      completeRegistration,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
