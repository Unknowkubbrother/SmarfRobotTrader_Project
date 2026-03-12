"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Eye, EyeOff, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import { toast } from "sonner";
import { api } from "@/lib/api";

type Step = 1 | 2 | 3;

const emailSchema = z.object({
    email: z.string().trim().email({ message: "Invalid email address" }),
});

const passwordSchema = z.object({
    password: z.string().min(6, { message: "Password must be at least 6 characters" }),
    confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
});

export default function ForgotPassword() {
    const [step, setStep] = useState<Step>(1);
    const [isLoading, setIsLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [otp, setOtp] = useState("");
    const [countdown, setCountdown] = useState(0);
    const [devOtp, setDevOtp] = useState<string | null>(null);
    const [email, setEmail] = useState("");
    const [recoveryEmailHint, setRecoveryEmailHint] = useState("");
    const router = useRouter();
    const searchParams = useSearchParams();

    const emailForm = useForm<z.infer<typeof emailSchema>>({
        resolver: zodResolver(emailSchema),
        defaultValues: { email: "" },
    });

    const passwordForm = useForm<z.infer<typeof passwordSchema>>({
        resolver: zodResolver(passwordSchema),
        defaultValues: { password: "", confirmPassword: "" },
    });

    const watchedEmail = emailForm.watch("email");
    const loginParams = new URLSearchParams();
    const preservedEmail = watchedEmail || email;
    if (preservedEmail) loginParams.set("email", preservedEmail);
    const loginHref = loginParams.toString() ? `/auth/login?${loginParams.toString()}` : "/auth/login";

    useEffect(() => {
        const prefilledEmail = searchParams.get("email")?.trim();
        if (!prefilledEmail) return;
        emailForm.reset({ email: prefilledEmail });
        setEmail(prefilledEmail);
    }, [emailForm, searchParams]);

    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [countdown]);

    const handleStep1 = async (data: z.infer<typeof emailSchema>) => {
        setIsLoading(true);
        try {
            const response = await api.post("/auth/forgot-password/request", { email: data.email });
            setEmail(data.email);
            setRecoveryEmailHint(response.data.recovery_email_hint || "");
            if (response.data.dev_otp) setDevOtp(response.data.dev_otp);
            setCountdown(60);
            setStep(2);
            toast.success("OTP sent to your recovery email");
        } catch (error: any) {
            toast.error(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStep2 = async () => {
        if (otp.length !== 6) {
            toast.error("Please enter 6-digit OTP");
            return;
        }

        setIsLoading(true);
        try {
            await api.post("/auth/forgot-password/verify", { email, otp });
            setStep(3);
            toast.success("OTP verified!");
        } catch (error: any) {
            toast.error(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStep3 = async (data: z.infer<typeof passwordSchema>) => {
        setIsLoading(true);
        try {
            await api.post("/auth/forgot-password/reset", {
                email,
                otp,
                new_password: data.password,
            });
            toast.success("Password reset successfully!");
            router.replace(`/auth/login?${new URLSearchParams({ email, reset: "success" }).toString()}`);
        } catch (error: any) {
            toast.error(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const resendOTP = async () => {
        if (countdown > 0) return;
        setIsLoading(true);
        try {
            const response = await api.post("/auth/forgot-password/request", { email });
            if (response.data.dev_otp) setDevOtp(response.data.dev_otp);
            setCountdown(60);
            toast.success("OTP resent!");
        } catch (error: any) {
            toast.error(error.message);
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
                        <Link href={loginHref} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-4">
                            <ArrowLeft className="w-4 h-4" />
                            Back to login
                        </Link>

                        {step === 1 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Forgot Password</h2>
                                <p className="text-sm text-muted-foreground text-center mb-6">
                                    Enter your email to receive OTP
                                </p>

                                <Form {...emailForm}>
                                    <form onSubmit={emailForm.handleSubmit(handleStep1)} className="space-y-4">
                                        <FormField control={emailForm.control} name="email" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Email</FormLabel>
                                                <FormControl><Input type="email" placeholder="your@email.com" className="h-11 rounded-lg bg-muted border-0" {...field} /></FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )} />

                                        <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                            {isLoading ? "Sending..." : "Send OTP"}
                                        </Button>
                                    </form>
                                </Form>
                            </>
                        )}

                        {step === 2 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Verify OTP</h2>
                                <p className="text-sm text-muted-foreground text-center mb-6">step 2 of 3</p>

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
                                        OTP sent to <span className="text-primary">{recoveryEmailHint}</span>
                                    </p>

                                    {devOtp && (
                                        <p className="text-xs text-amber-600 bg-amber-50 px-3 py-1 rounded">[DEV] OTP: {devOtp}</p>
                                    )}

                                    <Button onClick={handleStep2} className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading || otp.length !== 6}>
                                        {isLoading ? "Verifying..." : "Verify"}
                                    </Button>

                                    <button onClick={resendOTP} disabled={countdown > 0} className="text-sm text-muted-foreground hover:text-foreground disabled:opacity-50">
                                        {countdown > 0 ? `Resend OTP in ${countdown}s` : "Resend OTP"}
                                    </button>
                                </div>
                            </>
                        )}

                        {step === 3 && (
                            <>
                                <h2 className="text-2xl font-semibold text-center mb-1">Reset Password</h2>
                                <p className="text-sm text-muted-foreground text-center mb-6">step 3 of 3</p>

                                <Form {...passwordForm}>
                                    <form onSubmit={passwordForm.handleSubmit(handleStep3)} className="space-y-4">
                                        <FormField control={passwordForm.control} name="password" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>New Password</FormLabel>
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

                                        <FormField control={passwordForm.control} name="confirmPassword" render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Confirm New Password</FormLabel>
                                                <FormControl>
                                                    <Input type={showPassword ? "text" : "password"} className="h-11 rounded-lg bg-muted border-0" {...field} />
                                                </FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )} />

                                        <Button type="submit" className="w-full h-11 rounded-full bg-gradient-to-r from-[#1e3a5f] to-[#3b82f6]" disabled={isLoading}>
                                            {isLoading ? "Resetting..." : "Reset Password"}
                                        </Button>
                                    </form>
                                </Form>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
