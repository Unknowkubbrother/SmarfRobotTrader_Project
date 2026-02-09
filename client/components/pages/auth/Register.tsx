"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Eye, EyeOff, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import TurnstileWidget from "@/components/ui/TurnstileWidget";

type RegisterStep = 1 | 2 | 3 | 4;

const step1Schema = z.object({
    email: z.string().trim().email({ message: "Invalid email address" }),
    password: z.string().min(6, { message: "Password must be at least 6 characters" }),
    confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
});

const step2Schema = z.object({
    recoveryEmail: z.string().trim().email({ message: "Invalid email address" }),
});

const step4Schema = z.object({
    username: z.string().min(3, { message: "Username must be at least 3 characters" }).max(20),
});

export default function Register() {
    const [step, setStep] = useState<RegisterStep>(1);
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [otp, setOtp] = useState("");
    const [countdown, setCountdown] = useState(0);
    const [devOtp, setDevOtp] = useState<string | null>(null);
    const [cfToken, setCfToken] = useState<string>("");

    const searchParams = useSearchParams();
    const mode = searchParams.get("mode");

    const [regData, setRegData] = useState({
        email: "",
        password: "",
        recoveryEmail: "",
        isGoogle: false,
        googleIdToken: ""
    });

    const router = useRouter();
    const { user, signInWithGoogle, registerOTP, verifyOTP, completeRegistration, googleRegisterOTP, googleRegisterVerify, googleRegisterComplete } = useAuth();

    const step1Form = useForm<z.infer<typeof step1Schema>>({
        resolver: zodResolver(step1Schema),
        defaultValues: { email: "", password: "", confirmPassword: "" },
    });

    const step2Form = useForm<z.infer<typeof step2Schema>>({
        resolver: zodResolver(step2Schema),
        defaultValues: { recoveryEmail: "" },
    });

    const step4Form = useForm<z.infer<typeof step4Schema>>({
        resolver: zodResolver(step4Schema),
        defaultValues: { username: "" },
    });

    useEffect(() => {
        if (user) router.push("/");
    }, [user, router]);

    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [countdown]);

    const handleGoogleSignUp = async () => {
        setIsLoading(true);
        try {
            const result = await signInWithGoogle();
            if (result.error) {
                toast.error(result.error.message);
                return;
            }

            if (result.requireRegister) {
                setRegData(prev => ({
                    ...prev,
                    email: result.email || "",
                    isGoogle: true,
                    googleIdToken: result.googleInfo?.idToken
                }));

                setStep(2);
                toast.info("Please set a recovery email to complete registration.");
            } else if (result.requireOtp) {
                toast.info("Account already exists. Please login.");
                router.push("/auth/login");
            } else {
                toast.success("Welcome!");
                router.push("/");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleStep1 = async (data: z.infer<typeof step1Schema>) => {
        setRegData(prev => ({ ...prev, email: data.email, password: data.password, isGoogle: false }));
        setStep(2);
    };

    const handleStep2 = async (data: z.infer<typeof step2Schema>) => {
        if (data.recoveryEmail === regData.email) {
            toast.error(regData.isGoogle ? "Recovery email must be different from your Google email" : "Recovery email must be different from account email");
            return;
        }

        if (!cfToken) {
            toast.error("Please complete the security check");
            return;
        }

        setIsLoading(true);
        try {
            let result;
            if (regData.isGoogle) {
                result = await googleRegisterOTP(regData.googleIdToken, data.recoveryEmail, cfToken);
            } else {
                result = await registerOTP(regData.email, data.recoveryEmail, regData.password, cfToken);
            }

            setCfToken(""); // Reset token

            if (result.error) {
                toast.error(result.error.message);
                return;
            }

            setRegData(prev => ({ ...prev, recoveryEmail: data.recoveryEmail }));
            if (result.devOtp) setDevOtp(result.devOtp);
            setCountdown(60);
            setStep(3);
            toast.success("OTP sent to your recovery email");
        } finally {
            setIsLoading(false);
        }
    };

    const handleStep3 = async () => {
        if (otp.length !== 6) {
            toast.error("Please enter 6-digit OTP");
            return;
        }

        setIsLoading(true);
        try {
            let result;
            if (regData.isGoogle) {
                result = await googleRegisterVerify(regData.email, otp);
            } else {
                result = await verifyOTP(regData.recoveryEmail, otp);
            }

            if (result.error) {
                toast.error(result.error.message);
                return;
            }
            setStep(4);
            toast.success("Email verified!");
        } finally {
            setIsLoading(false);
        }
    };

    const handleStep4 = async (data: z.infer<typeof step4Schema>) => {
        setIsLoading(true);
        try {
            let result;
            if (regData.isGoogle) {
                result = await googleRegisterComplete(regData.email, data.username);
            } else {
                result = await completeRegistration(regData.recoveryEmail, data.username);
            }

            if (result.error) {
                toast.error(result.error.message);
                return;
            }

            if (regData.isGoogle) {
                toast.success("Registration complete!");
                router.push("/");
            } else {
                toast.success("Registration complete! Please login.");
                router.push("/auth/login");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const resendOTP = async () => {
        if (countdown > 0) return;
        setIsLoading(true);
        try {
            let result;
            if (regData.isGoogle) {
                result = await googleRegisterOTP(regData.googleIdToken, regData.recoveryEmail);
            } else {
                result = await registerOTP(regData.email, regData.recoveryEmail, regData.password);
            }

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

    const maskEmail = (email: string) => {
        if (!email) return "";
        const [name, domain] = email.split("@");
        if (!domain) return email;
        return `${name.slice(0, 4)}****@${domain}`;
    };

    const renderStepIndicator = () => (
        <p className="text-sm text-muted-foreground text-center mb-6">step {step} of 4</p>
    );

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
                                <h2 className="text-2xl font-semibold text-center mb-1">Create Account</h2>
                                {renderStepIndicator()}

                                <Button variant="outline" className="w-full h-11 rounded-full mb-4 gap-2" onClick={handleGoogleSignUp} disabled={isLoading}>
                                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                                    </svg>
                                    Sign up with Google
                                </Button>

                                <div className="relative mb-6">
                                    <div className="absolute inset-0 flex items-center"><div className="w-full border-t" /></div>
                                    <div className="relative flex justify-center text-xs"><span className="px-2 bg-card text-muted-foreground">Or</span></div>
                                </div>

                                <Form {...step1Form}>
                                    <form onSubmit={step1Form.handleSubmit(handleStep1)} className="space-y-4">
                                        <FormField control={step1Form.control} name="email" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Email</FormLabel>
                                                <FormControl><Input type="email" className="h-11 rounded-lg bg-muted border-0" {...field} /></FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )} />
                                        <FormField control={step1Form.control} name="password" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Password</FormLabel>
                                                <FormControl>
                                                    <div className="relative">
                                                        <Input type={showPassword ? "text" : "password"} className="h-11 rounded-lg bg-muted border-0 pr-10" {...field} />
                                                        <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                                                            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                                        </button>
                                                    </div>
                                                </FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )} />
                                        <FormField control={step1Form.control} name="confirmPassword" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Confirm password</FormLabel>
                                                <FormControl><Input type={showPassword ? "text" : "password"} className="h-11 rounded-lg bg-muted border-0" {...field} /></FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )} />
                                        <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]">NEXT</Button>
                                    </form>
                                </Form>
                            </>
                        )}

                        {step === 2 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">
                                    {regData.isGoogle ? "Complete Registration" : "Security Verification Email"}
                                </h2>
                                {renderStepIndicator()}

                                {regData.isGoogle && (
                                    <p className="text-sm text-center mb-4 text-muted-foreground">
                                        Signing up as <span className="text-foreground font-medium">{regData.email}</span>
                                    </p>
                                )}

                                <Form {...step2Form}>
                                    <form onSubmit={step2Form.handleSubmit(handleStep2)} className="space-y-4">
                                        <FormField control={step2Form.control} name="recoveryEmail" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Recovery Email {regData.isGoogle && "(OTP)"}</FormLabel>
                                                <FormControl><Input type="email" placeholder="Different email for security" className="h-11 rounded-lg bg-muted border-0" {...field} /></FormControl>
                                                <p className="text-xs text-muted-foreground">This email will be used for OTP verification.</p>
                                                <FormMessage />
                                            </FormItem>
                                        )} />

                                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-2 items-start">
                                            <Info className="w-4 h-4 text-amber-600 mt-0.5" />
                                            <p className="text-sm text-amber-800">
                                                {regData.isGoogle
                                                    ? "Please provide a recovery email different from your Google account email."
                                                    : "Note: OTP email must be different from your account email."}
                                            </p>
                                        </div>

                                        <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                            {isLoading ? "Sending..." : "NEXT"}
                                        </Button>
                                        <TurnstileWidget onVerify={setCfToken} />
                                    </form>
                                </Form>
                            </>
                        )}

                        {step === 3 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Verify Your Email</h2>
                                {renderStepIndicator()}

                                <div className="flex flex-col items-center space-y-4">
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

                                    <p className="text-sm text-muted-foreground">
                                        Verification code sent to <span className="text-primary">{maskEmail(regData.recoveryEmail)}</span>
                                    </p>

                                    {devOtp && (
                                        <p className="text-xs text-amber-600 bg-amber-50 px-3 py-1 rounded">[DEV] OTP: {devOtp}</p>
                                    )}

                                    <Button onClick={handleStep3} className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading || otp.length !== 6}>
                                        {isLoading ? "Verifying..." : "Verify"}
                                    </Button>

                                    <button onClick={resendOTP} disabled={countdown > 0} className="text-sm text-muted-foreground hover:text-foreground disabled:opacity-50">
                                        {countdown > 0 ? `Resend OTP in ${countdown}s` : "Resend OTP"}
                                    </button>
                                </div>
                            </>
                        )}

                        {step === 4 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Choose Your Username</h2>
                                {renderStepIndicator()}

                                <Form {...step4Form}>
                                    <form onSubmit={step4Form.handleSubmit(handleStep4)} className="space-y-4">
                                        <FormField control={step4Form.control} name="username" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Username</FormLabel>
                                                <FormControl><Input placeholder="Create a username" className="h-11 rounded-lg bg-muted border-0" {...field} /></FormControl>
                                                <p className="text-xs text-muted-foreground">Your public username.</p>
                                                <FormMessage />
                                            </FormItem>
                                        )} />

                                        <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                            {isLoading ? "Finishing..." : "Finish Sign Up"}
                                        </Button>
                                    </form>
                                </Form>
                            </>
                        )}

                        <p className="mt-6 text-center text-sm text-muted-foreground">
                            Already have an account?{" "}
                            <Link href="/auth/login" className="text-foreground font-medium hover:underline">Sign In</Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
