"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { z } from "zod";
// import { useForm } from "react-hook-form";
// import { zodResolver } from "@hookform/resolvers/zod";
import { Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
// import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import TurnstileWidget from "@/components/ui/TurnstileWidget";

const loginSchema = z.object({
    email: z.string().trim().email({ message: "Invalid email address" }),
    password: z.string().min(6, { message: "Password must be at least 6 characters" }),
});

export default function Login() {
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [otp, setOtp] = useState("");
    const [recoveryHint, setRecoveryHint] = useState("");
    const [isGoogleUser, setIsGoogleUser] = useState(false);
    const [hasPassword, setHasPassword] = useState(false);
    const [countdown, setCountdown] = useState(0);
    const [devOtp, setDevOtp] = useState<string | null>(null);
    const [cfToken, setCfToken] = useState<string>("");

    const router = useRouter();
    const { user, signIn, signInWithGoogle, loginVerify, checkUser, loginOtpInit, setPassword: setAuthPassword } = useAuth();

    useEffect(() => {
        if (user) router.push("/");
    }, [user, router]);

    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [countdown]);

    const handleGoogleLogin = async () => {
        setIsLoading(true);
        try {
            const result = await signInWithGoogle();
            if (result.error) {
                toast.error(result.error.message);
                return;
            }

            if (result.requireOtp) {
                if (result.email) setEmail(result.email);
                setRecoveryHint(result.recoveryEmailHint || "");
                setIsGoogleUser(true); // Treat as google user flow
                setStep(3); // Go to OTP
                toast.success("OTP sent to your recovery email");
            } else if (result.requireRegister) {
                const params = new URLSearchParams();
                params.set("mode", "google");
                toast.info("New account. Please complete registration.", { duration: 4000 });
                router.push(`/auth/register?${params.toString()}`);
            } else {
                toast.success("Welcome!");
                router.push("/");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleEmailSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!email) return;

        // Basic email validation
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            toast.error("Invalid email address");
            return;
        }

        if (!cfToken) {
            toast.error("Please complete the security check");
            return;
        }

        setIsLoading(true);
        try {
            const result = await checkUser(email, cfToken);
            if (result.error) {
                toast.error(result.error.message);
                return;
            }

            if (!result.exists) {
                toast.info("Account not found. Redirecting to registration.");
                router.push("/auth/register");
                return;
            }

            setHasPassword(!!result.hasPassword);
            setIsGoogleUser(!!result.isGoogle);

            if (result.isGoogle && !result.hasPassword) {
                setRecoveryHint(result.recoveryEmailHint || "");

                if (result.otpSent) {
                    if (result.devOtp) setDevOtp(result.devOtp);
                    setCountdown(60);
                    setStep(3);
                    toast.success("OTP sent to your recovery email");
                } else {
                    setStep(3);
                    toast.info("Please request a verification code.");
                }
            } else {

                setStep(2);
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handlePasswordLogin = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!cfToken) {
            toast.error("Please complete the security check");
            return;
        }

        setIsLoading(true);
        try {
            const result = await signIn(email, password, cfToken);
            if (result.error) {
                toast.error(result.error.message);
                return;
            }

            if (result.requireOtp) {
                setRecoveryHint(result.recoveryEmailHint || "");
                setStep(3);
                toast.success("OTP sent to your recovery email");
            } else {
                toast.success("Welcome back!");
                setCfToken(""); // Reset token
                router.push("/");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleVerifyOtp = async () => {
        if (otp.length !== 6) {
            toast.error("Please enter 6-digit OTP");
            return;
        }
        setIsLoading(true);
        try {
            const { error } = await loginVerify(email, otp);
            if (error) {
                toast.error(error.message);
                return;
            }

            // Check if we need to set password
            if (isGoogleUser && !hasPassword) {
                setStep(4);
                toast.success("Login verified! Please set a password.");
            } else {
                toast.success("Login verified!");
                router.push("/");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleSetPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        if (password.length < 6) {
            toast.error("Password must be at least 6 characters");
            return;
        }

        setIsLoading(true);
        try {
            const { error } = await setAuthPassword(password);
            if (error) {
                toast.error(error.message);
                return;
            }
            toast.success("Password set successfully! Welcome.");
            router.push("/");
        } finally {
            setIsLoading(false);
        }
    };

    const resendOTP = async (token?: string) => {
        if (countdown > 0) return;

        // If we require token and none provided (unless we want to allow one-click retry if backend allows? No, backend enforces it)
        if (!token) {
            // In this design, we use Turnstile onVerify to trigger this, so token should differ.
            // But if we call it from button click?
            // We won't have a button, we have the widget.
            return;
        }

        setIsLoading(true);
        try {
            const result = await loginOtpInit(email, token);
            if (result.error) {
                toast.error(result.error.message);
                return;
            }
            if (result.devOtp) setDevOtp(result.devOtp);
            setCountdown(60);
            toast.success("OTP resent!");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background flex">
            <div className="hidden lg:flex lg:w-1/2 bg-muted items-center justify-center p-12">
                <div className="max-w-md">
                    <p className="text-sm text-muted-foreground mb-2">Welcome To</p>
                    <h1 className="text-4xl font-bold text-foreground mb-1">
                        <span>Smarf</span>
                        <span className="text-primary">Robot</span>
                        <span>Trade</span>
                    </h1>
                    <p className="text-muted-foreground">Developed by robotTeam</p>
                </div>
            </div>

            <div className="flex-1 flex items-center justify-center p-8">
                <div className="w-full max-w-md">
                    <div className="lg:hidden mb-8">
                        <p className="text-sm text-muted-foreground mb-1">Welcome To</p>
                        <h1 className="text-2xl font-bold">
                            <span>Smarf</span>
                            <span className="text-primary">Robot</span>
                            <span>Trade</span>
                        </h1>
                    </div>

                    <div className="bg-card rounded-2xl shadow-lg p-8 border border-border">

                        {step === 1 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Sign In</h2>
                                <p className="text-sm text-muted-foreground text-center mb-6">To SmarfRobotTrade</p>

                                <Button
                                    variant="outline"
                                    className="w-full h-11 rounded-full mb-4 gap-2"
                                    onClick={handleGoogleLogin}
                                    disabled={isLoading}
                                >
                                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                                    </svg>
                                    Sign in with Google
                                </Button>

                                <div className="relative mb-6">
                                    <div className="absolute inset-0 flex items-center"><div className="w-full border-t" /></div>
                                    <div className="relative flex justify-center text-xs">
                                        <span className="px-2 bg-card text-muted-foreground">Or sign in with email</span>
                                    </div>
                                </div>

                                <form onSubmit={handleEmailSubmit} className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Email</label>
                                        <Input
                                            type="email"
                                            placeholder="name@example.com"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            className="h-11 rounded-lg bg-muted border-0"
                                            required
                                        />
                                    </div>
                                    <TurnstileWidget onVerify={setCfToken} />
                                    <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                        {isLoading ? "Checking..." : "Next"}
                                    </Button>
                                </form>

                                <p className="mt-6 text-center text-sm text-muted-foreground">
                                    Don't have an account?{" "}
                                    <Link href="/auth/register" className="text-foreground font-medium hover:underline">Sign Up</Link>
                                </p>
                            </>
                        )}

                        {step === 2 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Enter Password</h2>
                                <p className="text-sm text-muted-foreground text-center mb-6">for {email}</p>

                                <form onSubmit={handlePasswordLogin} className="space-y-4">
                                    <div className="space-y-2">
                                        <div className="flex justify-between">
                                            <label className="text-sm font-medium leading-none">Password</label>
                                            <Link href="/auth/forgot-password" className="text-xs text-muted-foreground hover:text-foreground">
                                                Forgot Password?
                                            </Link>
                                        </div>
                                        <div className="relative">
                                            <Input
                                                type={showPassword ? "text" : "password"}
                                                value={password}
                                                onChange={(e) => setPassword(e.target.value)}
                                                className="h-11 rounded-lg bg-muted border-0 pr-10"
                                                required
                                            />
                                            <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                                                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        </div>
                                    </div>
                                    <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                        {isLoading ? "Signing In..." : "Sign In"}
                                    </Button>
                                    <button type="button" onClick={() => setStep(1)} className="w-full text-sm text-muted-foreground hover:underline">Back</button>
                                </form>
                            </>
                        )}

                        {step === 3 && (
                            <div className="flex flex-col items-center space-y-4">
                                <h2 className="text-2xl font-semibold text-center mb-1">Verify Login</h2>
                                <p className="text-sm text-muted-foreground text-center">
                                    Enter OTP sent to <span className="text-primary">{recoveryHint}</span>
                                </p>

                                {devOtp && (
                                    <p className="text-xs text-amber-600 bg-amber-50 px-3 py-1 rounded">[DEV] OTP: {devOtp}</p>
                                )}

                                <InputOTP maxLength={6} value={otp} onChange={setOtp}>
                                    <InputOTPGroup>
                                        <InputOTPSlot index={0} />
                                        <InputOTPSlot index={1} />
                                        <InputOTPSlot index={2} />
                                        <InputOTPSlot index={3} />
                                        <InputOTPSlot index={4} />
                                        <InputOTPSlot index={5} />
                                    </InputOTPGroup>
                                </InputOTP>

                                <Button onClick={handleVerifyOtp} className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading || otp.length !== 6}>
                                    {isLoading ? "Verifying..." : "Verify Login"}
                                </Button>

                                {countdown > 0 && (
                                    <p className="text-sm text-muted-foreground text-center">Resend OTP in {countdown}s</p>
                                )}

                                <button onClick={() => setStep(1)} className="text-sm text-muted-foreground hover:underline">Back to Login</button>
                            </div>
                        )}

                        {step === 4 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Set Password</h2>
                                <p className="text-sm text-muted-foreground text-center mb-6">Create a password for your account</p>

                                <form onSubmit={handleSetPassword} className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium leading-none">New Password</label>
                                        <div className="relative">
                                            <Input
                                                type={showPassword ? "text" : "password"}
                                                value={password} // Reuse password state
                                                onChange={(e) => setPassword(e.target.value)}
                                                className="h-11 rounded-lg bg-muted border-0 pr-10"
                                                required
                                                minLength={6}
                                            />
                                            <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                                                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        </div>
                                    </div>
                                    <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                        {isLoading ? "Setting Password..." : "Set Password & Continue"}
                                    </Button>
                                    <div className="text-center">
                                        <Link href="/" className="text-sm text-muted-foreground hover:underline">Skip (Not Recommended)</Link>
                                    </div>
                                </form>
                            </>
                        )}

                    </div>
                </div>
            </div>
        </div>
    );
}
