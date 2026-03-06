import type { Metadata } from "next";
import { cookies } from "next/headers";
import Link from "next/link";
import {
    ArrowRight,
    BarChart3,
    Bot,
    Building2,
    CheckCircle2,
    LineChart,
} from "lucide-react";
import { Button } from "@/components/ui/button";

export const metadata: Metadata = {
    title: "SmarfRobotTrader | AI Trading Control Center",
    description:
        "Landing page for SmarfRobotTrader with a concise overview of broker support, bot control, and portfolio tracking.",
};

const featureCards = [
    {
        title: "เชื่อมบัญชี MT5",
        description: "เพิ่ม broker, server และบัญชีเทรดของคุณใน flow เดียว",
        icon: Building2,
    },
    {
        title: "คุมบอทจากหน้าเดียว",
        description: "เริ่ม หยุด และติดตามสถานะบอทได้แบบ realtime",
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
    "FXTM",
    "RoboForex",
    "Tickmill",
    "FOREX.com",
    "FP Markets",
    "Admirals",
];

const startSteps = [
    "สมัครสมาชิกหรือเข้าสู่ระบบ",
    "เชื่อมบัญชี MT5 ของคุณ",
    "เริ่มดูผลและควบคุมบอท",
];

export default function HomePage() {
    const hasAccessToken = Boolean(cookies().get("access_token")?.value);

    return (
        <main className="min-h-screen bg-background text-foreground">
            <div className="mx-auto max-w-7xl px-6 py-6 lg:py-8">
                <header className="glass-card rounded-[32px] flex items-center justify-between px-6 py-4">
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
                                    ไปหน้า dashboard
                                    <ArrowRight className="h-4 w-4" />
                                </Link>
                            </Button>
                        ) : (
                            <>
                                <Link
                                    href="/auth/login"
                                    className="hidden rounded-full border border-border px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground sm:inline-flex"
                                >
                                    เข้าสู่ระบบ
                                </Link>
                                <Button asChild size="lg" className="rounded-full px-6">
                                    <Link href="/auth/register">
                                        สมัครสมาชิก
                                        <ArrowRight className="h-4 w-4" />
                                    </Link>
                                </Button>
                            </>
                        )}
                    </div>
                </header>

                <section className="grid gap-6 py-8 lg:grid-cols-[1.1fr_0.9fr] lg:py-12">
                    <div className="space-y-6">
                        <div className="inline-flex rounded-full bg-primary/10 px-4 py-2 text-sm font-medium text-primary">
                            สำหรับผู้ใช้ที่ต้องการเริ่มก่อนเข้าใช้งานจริง
                        </div>

                        <div className="space-y-4">
                            <h1 className="max-w-3xl text-5xl font-semibold tracking-tight text-foreground lg:text-6xl">
                                เชื่อม broker คุมบอท และดูผลลัพธ์ในที่เดียว
                            </h1>
                            <p className="max-w-2xl text-lg leading-8 text-muted-foreground">
                                SmarfRobotTrader ช่วยให้คุณเริ่มจากการเชื่อมบัญชี MT5 แล้วไปต่อที่การควบคุมบอท
                                และติดตาม performance ได้ใน workspace เดียว
                            </p>
                        </div>

                        <div className="flex flex-col gap-4 sm:flex-row">
                            <Button asChild size="xl" className="rounded-full px-8">
                                <Link href="/auth/register">
                                    เริ่มต้นใช้งาน
                                    <ArrowRight className="h-4 w-4" />
                                </Link>
                            </Button>
                            <Button asChild size="xl" variant="outline" className="rounded-full px-8">
                                <Link href="/dashboard">ดูหน้า dashboard</Link>
                            </Button>
                        </div>

                        <div className="grid gap-4 sm:grid-cols-3">
                            <div className="glass-card rounded-[32px] p-5">
                                <p className="text-sm text-muted-foreground">Broker Catalog</p>
                                <p className="mt-2 text-3xl font-semibold">60+</p>
                                <p className="mt-2 text-sm text-muted-foreground">รายการ broker/server สำหรับเลือกใช้งาน</p>
                            </div>
                            <div className="glass-card rounded-[32px] p-5">
                                <p className="text-sm text-muted-foreground">Main Flow</p>
                                <p className="mt-2 text-3xl font-semibold">MT5</p>
                                <p className="mt-2 text-sm text-muted-foreground">เชื่อมบัญชี คุมบอท และดูผลลัพธ์จากระบบเดียว</p>
                            </div>
                            <div className="glass-card rounded-[32px] p-5">
                                <p className="text-sm text-muted-foreground">Start Fast</p>
                                <p className="mt-2 text-3xl font-semibold">3</p>
                                <p className="mt-2 text-sm text-muted-foreground">ขั้นตอนหลักก่อนเริ่มใช้งานจริง</p>
                            </div>
                        </div>
                    </div>

                    <div className="glass-card rounded-[36px] p-6 lg:p-7">
                        <div className="flex items-center gap-3">
                            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                                <LineChart className="h-5 w-5" />
                            </div>
                            <div>
                                <p className="text-sm text-muted-foreground">Overview</p>
                                <h2 className="text-2xl font-semibold">สิ่งที่คุณจะทำได้ในระบบ</h2>
                            </div>
                        </div>

                        <div className="mt-6 grid gap-4">
                            {featureCards.map(({ title, description, icon: Icon }) => (
                                <div key={title} className="rounded-[28px] border border-border bg-secondary/35 p-4">
                                    <div className="flex items-start gap-4">
                                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                                            <Icon className="h-5 w-5" />
                                        </div>
                                        <div>
                                            <h3 className="text-base font-semibold">{title}</h3>
                                            <p className="mt-1 text-sm leading-6 text-muted-foreground">{description}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                <section className="space-y-6 py-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <p className="text-sm font-medium uppercase tracking-[0.2em] text-primary">Supported Brokers</p>
                            <h2 className="mt-3 text-3xl font-semibold">รองรับ broker MT5 ยอดนิยม</h2>
                            <p className="mt-2 max-w-2xl text-muted-foreground">
                                เลือก broker ที่ใช้งานอยู่แล้ว แล้วเชื่อมเข้าระบบเพื่อเริ่มคุมบอทและดูผลลัพธ์ได้ทันที
                            </p>
                        </div>
                        <div className="rounded-full bg-primary/10 px-4 py-2 text-sm font-medium text-primary">
                            มากกว่า 60 broker/server
                        </div>
                    </div>

                    <div className="glass-card rounded-[36px] p-6 lg:p-7">
                        <div className="flex flex-wrap gap-3">
                            {supportedBrokers.map((broker) => (
                                <div
                                    key={broker}
                                    className="rounded-full border border-border bg-secondary/45 px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-secondary"
                                >
                                    {broker}
                                </div>
                            ))}
                        </div>
                        <p className="mt-5 text-sm text-muted-foreground">
                            และยังมี broker/server ใน catalog ของระบบอีกหลายรายการสำหรับเลือกใช้งาน
                        </p>
                    </div>
                </section>

                <section className="grid gap-6 py-8 lg:grid-cols-[1fr_0.95fr]">
                    <div className="rounded-[36px] bg-primary p-8 text-primary-foreground shadow-lg shadow-primary/20">
                        <p className="text-sm font-medium uppercase tracking-[0.2em] text-primary-foreground/80">Start Fast</p>
                        <h2 className="mt-3 text-3xl font-semibold">เริ่มใช้งานใน 3 ขั้นตอน</h2>

                        <div className="mt-6 grid gap-3">
                            {startSteps.map((step, index) => (
                                <div key={step} className="flex items-center gap-3 rounded-[24px] border border-white/20 bg-white/10 px-4 py-3">
                                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/15 text-xs font-semibold">
                                        0{index + 1}
                                    </div>
                                    <span className="text-sm">{step}</span>
                                </div>
                            ))}
                        </div>

                        <div className="mt-8 flex flex-col gap-4 sm:flex-row">
                            <Button asChild size="xl" variant="secondary" className="rounded-full px-8">
                                <Link href="/auth/register">สร้างบัญชีใหม่</Link>
                            </Button>
                            <Button asChild size="xl" variant="outline" className="rounded-full border-white/30 bg-transparent px-8 text-white hover:bg-white/10 hover:text-white">
                                <Link href="/auth/login">เข้าสู่ระบบ</Link>
                            </Button>
                        </div>
                    </div>

                    <div className="glass-card rounded-[36px] p-8">
                        <h3 className="text-2xl font-semibold">สิ่งที่ควรเตรียม</h3>
                        <div className="mt-6 grid gap-3">
                            <div className="flex items-center gap-3 rounded-[24px] bg-secondary/45 px-4 py-3">
                                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                <span className="text-sm text-muted-foreground">อีเมลสำหรับสมัครสมาชิก</span>
                            </div>
                            <div className="flex items-center gap-3 rounded-[24px] bg-secondary/45 px-4 py-3">
                                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                <span className="text-sm text-muted-foreground">บัญชี MT5 ของ broker ที่คุณใช้งาน</span>
                            </div>
                            <div className="flex items-center gap-3 rounded-[24px] bg-secondary/45 px-4 py-3">
                                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                <span className="text-sm text-muted-foreground">Server name, login ID และรหัสผ่าน MT5</span>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </main>
    );
}
