import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
    title: "AlgoTrade AI - AI-Powered Algorithmic Trading Platform",
    description: "Advanced AI-powered algorithmic trading platform using Reinforcement Learning and LLM for automated, transparent, and secure trading.",
    authors: [{ name: "AlgoTrade AI" }],
    openGraph: {
        title: "AlgoTrade AI - AI Trading Platform",
        description: "Advanced AI-powered algorithmic trading with RL and LLM technology",
        type: "website",
    },
    twitter: {
        card: "summary_large_image",
        site: "@AlgoTradeAI",
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body>
                <Providers>{children}</Providers>
            </body>
        </html>
    );
}
