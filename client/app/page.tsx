import type { Metadata } from "next";
import Link from "next/link";
import { Space_Grotesk } from "next/font/google";
import {
    ArrowRight,
    BarChart3,
    Bot,
    Building2,
    CheckCircle2,
    Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const headingFont = Space_Grotesk({
    subsets: ["latin"],
    weight: ["500", "700"],
});

export const metadata: Metadata = {
    title: "SmarfRobotTrader | AI Trading Control Center",
    description:
        "Modern landing page for SmarfRobotTrader with a concise overview of features, supported brokers, and the first-use flow.",
};

const featureCards = [
    {
        title: "เชื่อมบัญชี MT5",
        description: "เลือก broker, server และเพิ่มบัญชีเทรดได้เป็นระบบ",
        icon: Building2,
    },
    {
        title: "คุมบอทแบบ Live",
        description: "สั่งงานและติดตามสถานะบอทจากหน้าใช้งานเดียว",
        icon: Bot,
    },
    {
        title: "ดูผลลัพธ์ชัดเจน",
        description: "เช็ก performance, calendar และภาพรวมพอร์ตได้ทันที",
        icon: BarChart3,
    },
];

const supportedBrokers = [
    "IC Markets",
    "Pepperstone",
    "OANDA",
    "FBS",
    "FOREX.com",
    "FXTM",
    "FP Markets",
    "Admirals",
    "RoboForex",
    "Tickmill",
];

const quickSteps = [
    "สมัครสมาชิกหรือเข้าสู่ระบบ",
    "เชื่อมบัญชี MT5 ของคุณ",
    "เริ่มติดตามและควบคุมบอท",
];

export default function HomePage() {
    return (
        <main className="relative min-h-screen overflow-hidden bg-[linear-gradient(180deg,#f8fbff_0%,#eff6ff_52%,#f9fbff_100%)] text-slate-900">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.18),transparent_24%),radial-gradient(circle_at_top_right,rgba(37,99,235,0.16),transparent_30%),linear-gradient(rgba(96,165,250,0.10)_1px,transparent_1px),linear-gradient(90deg,rgba(96,165,250,0.10)_1px,transparent_1px)] bg-[size:auto,auto,82px_82px,82px_82px]" />
            <div className="pointer-events-none absolute left-[-8rem] top-8 h-72 w-72 rounded-full bg-cyan-300/30 blur-3xl" />
            <div className="pointer-events-none absolute right-[-8rem] top-20 h-80 w-80 rounded-full bg-blue-300/25 blur-3xl" />

            <div className="relative">
                <header className="px-6 pt-6">
                    <nav className="mx-auto flex max-w-7xl items-center justify-between rounded-full border border-sky-100 bg-white/82 px-5 py-4 shadow-[0_16px_40px_rgba(37,99,235,0.08)] backdrop-blur">
                        <div className="flex items-center gap-3">
                            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-sky-400 to-blue-600 text-sm font-semibold text-white">
                                SR
                            </div>
                            <div>
                                <p className="text-base font-semibold text-slate-950">SmarfRobotTrader</p>
                                <p className="text-sm text-slate-500">AI Trading</p>
                            </div>
                        </div>

                        <div className="hidden items-center gap-8 text-sm text-slate-600 lg:flex">
                            <a href="#brokers" className="transition hover:text-blue-700">โบรกเกอร์</a>
                            <a href="#start" className="transition hover:text-blue-700">เริ่มใช้งาน</a>
                        </div>

                        <div className="flex items-center gap-3">
                            <Link
                                href="/auth/login"
                                className="hidden rounded-full border border-slate-200 px-4 py-2 text-sm text-slate-700 transition hover:border-sky-200 hover:text-blue-700 sm:inline-flex"
                            >
                                เข้าสู่ระบบ
                            </Link>
                            <Button
                                asChild
                                size="lg"
                                className="rounded-full bg-gradient-to-r from-sky-400 to-blue-600 px-6 text-white shadow-[0_12px_30px_rgba(37,99,235,0.20)] hover:from-sky-500 hover:to-blue-700"
                            >
                                <Link href="/auth/register">
                                    สมัครสมาชิก
                                    <ArrowRight className="h-4 w-4" />
                                </Link>
                            </Button>
                        </div>
                    </nav>
                </header>

                <section className="mx-auto grid max-w-7xl gap-10 px-6 pb-16 pt-12 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:pt-20">
                    <div>
                        <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white/80 px-4 py-2 text-sm text-sky-900 shadow-sm">
                            <Sparkles className="h-4 w-4 text-sky-500" />
                            ระบบ AI เทรดสำหรับ Forex
                        </div>

                        <h1 className={`${headingFont.className} mt-8 max-w-4xl text-5xl font-medium leading-[1.02] text-slate-950 sm:text-6xl lg:text-7xl`}>
                            <span className="bg-gradient-to-r from-sky-500 to-blue-700 bg-clip-text text-transparent"> SmarfRobotTrade </span>
                            บอทเทรด
                        </h1>


                        <div className="mt-8 flex flex-col gap-4 sm:flex-row">
                            <Button
                                asChild
                                size="xl"
                                className="rounded-full bg-gradient-to-r from-sky-400 to-blue-600 px-8 text-white shadow-[0_16px_36px_rgba(37,99,235,0.20)] hover:from-sky-500 hover:to-blue-700"
                            >
                                <Link href="/auth/register">
                                    เริ่มต้นใช้งาน
                                    <ArrowRight className="h-4 w-4" />
                                </Link>
                            </Button>

                            <Button
                                asChild
                                size="xl"
                                variant="outline"
                                className="rounded-full border-sky-200 bg-white/80 px-8 text-slate-800 shadow-sm hover:bg-sky-50 hover:text-blue-700"
                            >
                                <Link href="/dashboard">ดูหน้า dashboard</Link>
                            </Button>
                        </div>

                        <div className="mt-8 flex flex-wrap gap-3">
                            <span className="rounded-full border border-sky-100 bg-white/85 px-4 py-2 text-sm text-slate-700">MT5 Broker Support</span>
                            <span className="rounded-full border border-sky-100 bg-white/85 px-4 py-2 text-sm text-slate-700">Live Bot Control</span>
                            <span className="rounded-full border border-sky-100 bg-white/85 px-4 py-2 text-sm text-slate-700">Performance Tracking</span>
                        </div>
                    </div>

                    <div className="rounded-[34px] border border-sky-100 bg-white/80 p-6 shadow-[0_24px_60px_rgba(37,99,235,0.12)] backdrop-blur">
                        <p className="text-sm uppercase tracking-[0.28em] text-sky-700/80">Quick View</p>

                        <div className="mt-8 grid gap-4">
                            {featureCards.map(({ title, description, icon: Icon }) => (
                                <div
                                    key={title}
                                    className="flex items-start gap-4 rounded-[26px] border border-sky-100 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(239,246,255,0.92))] p-5"
                                >
                                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-100 to-blue-100 text-blue-700">
                                        <Icon className="h-5 w-5" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-950">{title}</h3>
                                        <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                <section id="brokers" className="mx-auto max-w-7xl px-6 py-20">
                    <div className="rounded-[34px] border border-sky-100 bg-white/84 p-8 shadow-[0_20px_48px_rgba(37,99,235,0.10)]">
                        <p className="text-sm uppercase tracking-[0.28em] text-sky-700/80">Supported Brokers</p>
                        <div className="mt-4 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                            <div className="max-w-2xl">
                                <h2 className={`${headingFont.className} text-4xl font-medium text-slate-950`}>
                                    รองรับ broker MT5 ยอดนิยม
                                </h2>
                                <p className="mt-4 text-lg leading-8 text-slate-600">
                                    ตัวอย่างเช่นรายชื่อด้านล่าง และในระบบมี broker/server มากกว่า 60 รายการสำหรับเลือกใช้งาน
                                </p>
                            </div>
                            <div className="rounded-full border border-sky-100 bg-sky-50 px-4 py-2 text-sm font-medium text-sky-900">
                                มากกว่า 60 broker/server
                            </div>
                        </div>

                        <div className="mt-8 flex flex-wrap gap-3">
                            {supportedBrokers.map((broker) => (
                                <span
                                    key={broker}
                                    className="rounded-full border border-sky-100 bg-sky-50 px-4 py-2 text-sm font-medium text-sky-900"
                                >
                                    {broker}
                                </span>
                            ))}
                        </div>
                    </div>
                </section>

                <section id="start" className="mx-auto max-w-7xl px-6 pb-24">
                    <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
                        <div className="rounded-[34px] border border-blue-100 bg-gradient-to-r from-sky-500 to-blue-700 p-8 text-white shadow-[0_24px_60px_rgba(37,99,235,0.18)]">
                            <p className="text-sm uppercase tracking-[0.28em] text-sky-100/80">Start Fast</p>
                            <h2 className={`${headingFont.className} mt-4 text-4xl font-medium`}>
                                เริ่มใช้งานใน 3 ขั้นตอน
                            </h2>

                            <div className="mt-6 grid gap-3">
                                {quickSteps.map((step, index) => (
                                    <div
                                        key={step}
                                        className="flex items-center gap-3 rounded-2xl border border-white/20 bg-white/10 px-4 py-3 text-sm text-white"
                                    >
                                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/20 text-xs font-semibold">
                                            0{index + 1}
                                        </div>
                                        <span>{step}</span>
                                    </div>
                                ))}
                            </div>

                            <div className="mt-8 flex flex-col gap-4 sm:flex-row">
                                <Button
                                    asChild
                                    size="xl"
                                    className="rounded-full bg-white px-8 text-blue-700 shadow-none hover:bg-sky-50"
                                >
                                    <Link href="/auth/register">สร้างบัญชีใหม่</Link>
                                </Button>
                                <Button
                                    asChild
                                    size="xl"
                                    variant="outline"
                                    className="rounded-full border-white/35 bg-transparent px-8 text-white hover:bg-white/10 hover:text-white"
                                >
                                    <Link href="/auth/login">เข้าสู่ระบบ</Link>
                                </Button>
                            </div>
                        </div>

                        <div className="rounded-[34px] border border-sky-100 bg-white/84 p-8 shadow-[0_18px_42px_rgba(37,99,235,0.10)]">
                            <p className="text-sm uppercase tracking-[0.22em] text-sky-700/80">Before You Start</p>
                            <h3 className="mt-4 text-2xl font-semibold text-slate-950">สิ่งที่ควรเตรียม</h3>

                            <div className="mt-6 grid gap-3">
                                <div className="flex items-center gap-3 rounded-2xl border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-slate-700">
                                    <CheckCircle2 className="h-4 w-4 shrink-0 text-blue-700" />
                                    <span>อีเมลสำหรับสมัครสมาชิก</span>
                                </div>
                                <div className="flex items-center gap-3 rounded-2xl border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-slate-700">
                                    <CheckCircle2 className="h-4 w-4 shrink-0 text-blue-700" />
                                    <span>บัญชี MT5 ของ broker ที่ใช้งาน</span>
                                </div>
                                <div className="flex items-center gap-3 rounded-2xl border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-slate-700">
                                    <CheckCircle2 className="h-4 w-4 shrink-0 text-blue-700" />
                                    <span>Server name, Account MT5</span>
                                </div>
                            </div>

                        </div>
                    </div>
                </section>
            </div>
        </main>
    );
}
