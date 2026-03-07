import type { Metadata } from "next";
import { cookies } from "next/headers";
import Link from "next/link";
import {
    ArrowRight,
    BarChart3,
    Bot,
    Brain,
    Building2,
    CheckCircle2,
    LineChart,
    Shield,
    Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";

export const metadata: Metadata = {
    title: "SmarfRobotTrader | AI Trading Control Center",
    description:
        "Automated AI trading platform. Connect your MT5 account, deploy AI bots, and track performance — all from a single dashboard.",
};

const capabilities = [
    {
        title: "AI-Driven Analysis",
        description: "The bot reads charts and decides when to buy or sell automatically — no manual watching needed",
        icon: Brain,
    },
    {
        title: "Instant MT5 Connection",
        description: "Link your broker account and start automated trading right away. Over 60 servers supported",
        icon: Building2,
    },
    {
        title: "Real-time Bot Control",
        description: "Start, stop, and monitor your bots at any time from a single control panel",
        icon: Bot,
    },
    {
        title: "Complete Dashboard",
        description: "View your balance, profit/loss, trade statistics, and open orders — all updated live",
        icon: BarChart3,
    },
];

const highlights = [
    {
        stat: "AI-Powered",
        label: "Smart Trading",
        description: "Bots learn from real charts, not fixed formulas",
    },
    {
        stat: "60+",
        label: "Broker/Server",
        description: "Supports popular MT5 brokers worldwide, ready to connect",
    },
    {
        stat: "Real-time",
        label: "Monitoring",
        description: "Track your portfolio and control bots instantly with live updates",
    },
];

const supportedBrokers = [
    "IC Markets",
    "Pepperstone",
    "OANDA",
    "FBS",
    "FXTM",
    "RoboForex",
    "Tickmill",
    "FOREX.com",
    "FP Markets",
    "Admirals",
];

const startSteps = [
    {
        step: "Create an Account",
        detail: "Sign up to get started with the platform",
    },
    {
        step: "Connect Your MT5",
        detail: "Enter your broker's server name, login ID, and password",
    },
    {
        step: "Deploy a Bot and Start Trading",
        detail: "Choose an AI model and hit start — the system trades for you automatically",
    },
];

export default function HomePage() {
    const hasAccessToken = Boolean(cookies().get("access_token")?.value);

    return (
        <main className="min-h-screen bg-background text-foreground">
            <div className="mx-auto max-w-7xl px-6 py-6 lg:py-8">
                {/* Header */}
                <header className="bg-white border border-border rounded-2xl flex items-center justify-between px-6 py-4 shadow-sm">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground">
                            SR
                        </div>
                        <div>
                            <p className="text-base font-semibold">SmarfRobotTrader</p>
                            <p className="text-sm text-muted-foreground">AI Trading Control Center</p>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        {hasAccessToken ? (
                            <Button asChild size="lg" className="rounded-full px-6">
                                <Link href="/dashboard">
                                    Go to Dashboard
                                    <ArrowRight className="h-4 w-4" />
                                </Link>
                            </Button>
                        ) : (
                            <>
                                <Link
                                    href="/auth/login"
                                    className="hidden rounded-full border border-border px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground sm:inline-flex"
                                >
                                    Sign In
                                </Link>
                                <Button asChild size="lg" className="rounded-full px-6">
                                    <Link href="/auth/register">
                                        Get Started
                                        <ArrowRight className="h-4 w-4" />
                                    </Link>
                                </Button>
                            </>
                        )}
                    </div>
                </header>

                {/* Hero Section */}
                <section className="grid gap-8 py-10 lg:grid-cols-[1.1fr_0.9fr] lg:py-14">
                    <div className="space-y-6">
                        <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 text-sm font-medium text-primary">
                            <Zap className="h-3.5 w-3.5" />
                            Automated AI Trading Platform
                        </div>

                        <div className="space-y-4">
                            <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-foreground lg:text-5xl">
                                Let AI Trade for You
                                <br />
                                <span className="text-primary">You Just Watch the Results</span>
                            </h1>
                            <p className="max-w-2xl text-lg leading-8 text-muted-foreground">
                                SmarfRobotTrader lets you connect your MT5 account and deploy AI bots
                                that analyze charts and place orders automatically
                                — just sit back and monitor your performance from the dashboard.
                            </p>
                        </div>

                        <div className="flex flex-col gap-4 sm:flex-row">
                            <Button asChild size="xl" className="rounded-full px-8">
                                <Link href="/auth/register">
                                    Start for Free
                                    <ArrowRight className="h-4 w-4" />
                                </Link>
                            </Button>
                            <Button asChild size="xl" variant="outline" className="rounded-full px-8">
                                <Link href="/dashboard">View Dashboard</Link>
                            </Button>
                        </div>

                        {/* Highlight Stats */}
                        <div className="grid gap-4 sm:grid-cols-3">
                            {highlights.map(({ stat, label, description }) => (
                                <div key={label} className="bg-white border border-border rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow">
                                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</p>
                                    <p className="mt-2 text-2xl font-semibold text-primary">{stat}</p>
                                    <p className="mt-2 text-sm text-muted-foreground">{description}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Capabilities Card */}
                    <div className="bg-white border border-border rounded-2xl p-6 shadow-sm lg:p-7">
                        <div className="flex items-center gap-3">
                            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                                <LineChart className="h-5 w-5" />
                            </div>
                            <div>
                                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Features</p>
                                <h2 className="text-xl font-semibold">What You Can Do</h2>
                            </div>
                        </div>

                        <div className="mt-6 grid gap-3">
                            {capabilities.map(({ title, description, icon: Icon }) => (
                                <div key={title} className="rounded-xl border border-border bg-background/50 p-4 hover:bg-background transition-colors">
                                    <div className="flex items-start gap-4">
                                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                                            <Icon className="h-5 w-5" />
                                        </div>
                                        <div>
                                            <h3 className="text-sm font-semibold">{title}</h3>
                                            <p className="mt-1 text-sm leading-6 text-muted-foreground">{description}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* How It Works */}
                <section className="py-8">
                    <div className="text-center mb-8">
                        <p className="text-xs font-medium uppercase tracking-[0.2em] text-primary">How It Works</p>
                        <h2 className="mt-3 text-3xl font-semibold">How the AI Trades for You</h2>
                        <p className="mt-2 max-w-2xl mx-auto text-muted-foreground">
                            Our bots don't rely on fixed rules — they learn from real charts and adapt to market conditions.
                        </p>
                    </div>

                    <div className="grid gap-4 md:grid-cols-3">
                        <div className="bg-white border border-border rounded-2xl p-6 shadow-sm text-center hover:shadow-md transition-shadow">
                            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary mb-4">
                                <Brain className="h-7 w-7" />
                            </div>
                            <h3 className="text-base font-semibold">Learns from Experience</h3>
                            <p className="mt-2 text-sm text-muted-foreground">
                                The AI learns from thousands of past trades to decide the best time to buy, sell, or wait.
                            </p>
                        </div>
                        <div className="bg-white border border-border rounded-2xl p-6 shadow-sm text-center hover:shadow-md transition-shadow">
                            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary mb-4">
                                <LineChart className="h-7 w-7" />
                            </div>
                            <h3 className="text-base font-semibold">Reads Charts Like a Pro</h3>
                            <p className="mt-2 text-sm text-muted-foreground">
                                The AI looks at price charts and identifies patterns just like an experienced trader would.
                            </p>
                        </div>
                        <div className="bg-white border border-border rounded-2xl p-6 shadow-sm text-center hover:shadow-md transition-shadow">
                            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary mb-4">
                                <Shield className="h-7 w-7" />
                            </div>
                            <h3 className="text-base font-semibold">Built-in Risk Protection</h3>
                            <p className="mt-2 text-sm text-muted-foreground">
                                Automatic risk management adjusts trade sizes based on your account balance to protect your capital.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Supported Brokers */}
                <section className="space-y-6 py-8">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <p className="text-xs font-medium uppercase tracking-[0.2em] text-primary">Supported Brokers</p>
                            <h2 className="mt-3 text-3xl font-semibold">Works with Popular MT5 Brokers</h2>
                            <p className="mt-2 max-w-2xl text-muted-foreground">
                                Pick your existing broker, enter your login details, and start AI-powered trading right away.
                            </p>
                        </div>
                        <div className="rounded-full bg-primary/10 px-4 py-2 text-sm font-medium text-primary">
                            Over 60 broker/servers
                        </div>
                    </div>

                    <div className="bg-white border border-border rounded-2xl p-6 shadow-sm lg:p-7">
                        <div className="flex flex-wrap gap-3">
                            {supportedBrokers.map((broker) => (
                                <div
                                    key={broker}
                                    className="rounded-full border border-border bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-primary/5 hover:border-primary/30"
                                >
                                    {broker}
                                </div>
                            ))}
                        </div>
                        <p className="mt-5 text-sm text-muted-foreground">
                            Plus many more brokers and servers available in the system — search when connecting your account.
                        </p>
                    </div>
                </section>

                {/* Getting Started + Prerequisites */}
                <section className="grid gap-6 py-8 lg:grid-cols-[1fr_0.95fr]">
                    <div className="rounded-2xl bg-primary p-8 text-primary-foreground shadow-lg shadow-primary/20">
                        <p className="text-xs font-medium uppercase tracking-[0.2em] text-primary-foreground/80">Getting Started</p>
                        <h2 className="mt-3 text-3xl font-semibold">Up and Running in 3 Steps</h2>

                        <div className="mt-6 grid gap-3">
                            {startSteps.map(({ step, detail }, index) => (
                                <div key={step} className="flex items-start gap-4 rounded-xl border border-white/20 bg-white/10 px-5 py-4">
                                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/15 text-xs font-semibold">
                                        0{index + 1}
                                    </div>
                                    <div>
                                        <p className="text-sm font-semibold">{step}</p>
                                        <p className="mt-1 text-xs text-primary-foreground/70">{detail}</p>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="mt-8 flex flex-col gap-4 sm:flex-row">
                            <Button asChild size="xl" variant="secondary" className="rounded-full px-8">
                                <Link href="/auth/register">Create Account</Link>
                            </Button>
                            <Button asChild size="xl" variant="outline" className="rounded-full border-white/30 bg-transparent px-8 text-white hover:bg-white/10 hover:text-white">
                                <Link href="/auth/login">Sign In</Link>
                            </Button>
                        </div>
                    </div>

                    <div className="bg-white border border-border rounded-2xl p-8 shadow-sm">
                        <h3 className="text-xl font-semibold">What You Need</h3>
                        <p className="mt-2 text-sm text-muted-foreground">Just an MT5 account and you are good to go.</p>
                        <div className="mt-6 grid gap-3">
                            <div className="flex items-center gap-3 rounded-xl bg-background px-4 py-3">
                                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                <span className="text-sm text-muted-foreground">An email address to sign up</span>
                            </div>
                            <div className="flex items-center gap-3 rounded-xl bg-background px-4 py-3">
                                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                <span className="text-sm text-muted-foreground">An MT5 account from your broker</span>
                            </div>
                            <div className="flex items-center gap-3 rounded-xl bg-background px-4 py-3">
                                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                <span className="text-sm text-muted-foreground">Your server name, login ID, and MT5 password</span>
                            </div>
                        </div>

                        <div className="mt-6 rounded-xl bg-primary/5 border border-primary/15 px-4 py-3">
                            <div className="flex items-center gap-1.5">
                                <Shield className="h-3.5 w-3.5 text-primary" />
                                <p className="text-xs text-primary font-medium">Note</p>
                            </div>
                            <p className="mt-1 text-xs text-muted-foreground">
                                Your login credentials are encrypted and used only within the system. They are never shared with third parties.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Footer */}
                <footer className="border-t border-border py-6 mt-4">
                    <div className="flex flex-col items-center gap-2 text-center">
                        <p className="text-sm font-medium text-foreground">SmarfRobotTrader</p>
                        <p className="text-xs text-muted-foreground">
                            AI Trading Control Center — Automated Trading, Simplified
                        </p>
                    </div>
                </footer>
            </div>
        </main>
    );
}
