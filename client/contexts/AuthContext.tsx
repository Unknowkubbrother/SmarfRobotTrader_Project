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
  avatar_url?: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  isAdmin: boolean;
  signIn: (email: string, password: string, cfToken?: string) => Promise<{ error: Error | null; requireOtp?: boolean; recoveryEmailHint?: string }>;
  loginVerify: (email: string, otp: string) => Promise<{ error: Error | null }>;
  signInWithGoogle: () => Promise<{ error: Error | null; requireOtp?: boolean; requireRegister?: boolean; recoveryEmailHint?: string; googleInfo?: any; email?: string }>;
  signOut: () => Promise<void>;

  registerOTP: (email: string, recoveryEmail: string, password: string, cfToken?: string) => Promise<{ error: Error | null; devOtp?: string }>;
  verifyOTP: (recoveryEmail: string, otp: string) => Promise<{ error: Error | null; userId?: string }>;
  completeRegistration: (recoveryEmail: string, username: string) => Promise<{ error: Error | null }>;

  googleRegisterOTP: (idToken: string, recoveryEmail: string, cfToken?: string) => Promise<{ error: Error | null; devOtp?: string }>;
  googleRegisterVerify: (email: string, otp: string) => Promise<{ error: Error | null; verified?: boolean }>;
  googleRegisterComplete: (email: string, username: string) => Promise<{ error: Error | null }>;

  checkUser: (email: string, cfToken?: string) => Promise<{ exists: boolean; hasPassword?: boolean; isGoogle?: boolean; recoveryEmailHint?: string; otpSent?: boolean; devOtp?: string; error?: Error }>;
  loginOtpInit: (email: string, cfToken?: string) => Promise<{ error: Error | null; devOtp?: string }>;
  setPassword: (password: string) => Promise<{ error: Error | null }>;
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

  const signIn = async (email: string, password: string, cfToken?: string) => {
    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);
      if (cfToken) formData.append("cf_token", cfToken);

      const { data } = await api.post("/auth/login", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });

      if (data.require_otp) {
        return {
          error: null,
          requireOtp: true,
          recoveryEmailHint: data.recovery_email_hint
        };
      }

      await fetchCurrentUser();
      return { error: null };
    } catch (error: any) {
      return { error: error };
    }
  };

  const loginVerify = async (email: string, otp: string) => {
    try {
      await api.post("/auth/login/verify", { email, otp });
      await fetchCurrentUser();
      return { error: null };
    } catch (error: any) {
      return { error: error };
    }
  };

  const signInWithGoogle = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const email = result.user.email || undefined;
      const idToken = await result.user.getIdToken();

      const { data } = await api.post("/auth/google", { id_token: idToken });

      if (data.require_otp) {
        return {
          error: null,
          requireOtp: true,
          recoveryEmailHint: data.recovery_email_hint,
          email
        };
      }

      if (data.require_register) {
        return {
          error: null,
          requireRegister: true,
          googleInfo: { ...data.google_info, idToken: idToken },
          email
        };
      }

      await fetchCurrentUser();
      return { error: null, email };
    } catch (error: any) {
      return { error: error };
    }
  };

  const registerOTP = async (email: string, recoveryEmail: string, password: string, cfToken?: string) => {
    try {
      const { data } = await api.post("/auth/register/otp", {
        email,
        recovery_email: recoveryEmail,
        password,
        cf_token: cfToken,
      });
      return { error: null, devOtp: data.dev_otp };
    } catch (error: any) {
      return { error: error };
    }
  };

  const verifyOTP = async (recoveryEmail: string, otp: string) => {
    try {
      const { data } = await api.post("/auth/register/verify_otp", {
        recovery_email: recoveryEmail,
        otp,
      });
      return { error: null, userId: data.user?.id };
    } catch (error: any) {
      return { error: error };
    }
  };

  const completeRegistration = async (recoveryEmail: string, username: string) => {
    try {
      await api.post("/auth/register/complete", {
        recovery_email: recoveryEmail,
        username,
      });
      return { error: null };
    } catch (error: any) {
      return { error: error };
    }
  };

  const googleRegisterOTP = async (idToken: string, recoveryEmail: string, cfToken?: string) => {
    try {
      const { data } = await api.post("/auth/google/register/otp", {
        id_token: idToken,
        recovery_email: recoveryEmail,
        cf_token: cfToken
      });
      return { error: null, devOtp: data.dev_otp };
    } catch (error: any) {
      return { error: error };
    }
  };

  const googleRegisterVerify = async (email: string, otp: string) => {
    try {
      const { data } = await api.post("/auth/google/register/verify", { email, otp });
      return { error: null, verified: data.verified };
    } catch (error: any) {
      return { error: error };
    }
  };

  const googleRegisterComplete = async (email: string, username: string) => {
    try {
      await api.post("/auth/google/register/complete", { email, username });
      await fetchCurrentUser();
      return { error: null };
    } catch (error: any) {
      return { error: error };
    }
  };

  const checkUser = async (email: string, cfToken?: string) => {
    try {
      const { data } = await api.post("/auth/check-user", { email, cf_token: cfToken });
      return {
        exists: data.exists,
        hasPassword: data.has_password,
        isGoogle: data.is_google,
        recoveryEmailHint: data.recovery_email_hint,
        otpSent: data.otp_sent,
        devOtp: data.dev_otp
      };
    } catch (error: any) {
      return { exists: false, error: error };
    }
  };

  const loginOtpInit = async (email: string, cfToken?: string) => {
    try {
      const { data } = await api.post("/auth/login/otp-init", { email, cf_token: cfToken });
      return { error: null, devOtp: data.dev_otp };
    } catch (error: any) {
      return { error: error };
    }
  };

  const setPassword = async (password: string) => {
    try {
      await api.post("/auth/set-password", { new_password: password });
      return { error: null };
    } catch (error: any) {
      return { error: error };
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
      loginVerify,
      signInWithGoogle,
      signOut,
      registerOTP,
      verifyOTP,
      completeRegistration,
      googleRegisterOTP,
      googleRegisterVerify,
      googleRegisterComplete,
      checkUser,
      loginOtpInit,
      setPassword
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
