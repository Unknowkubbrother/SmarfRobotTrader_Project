import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
    title: "SmarfRobotTrade | AI Trading",
    description: "Advanced AI-powered algorithmic trading platform using Reinforcement Learning and LLM for automated, transparent, and secure trading.",
    authors: [{ name: "SmarfRobotTrade" }],
    icons: {
        icon: "/logo.png",
    },
    openGraph: {
        title: "SmarfRobotTrade | AI Trading",
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
