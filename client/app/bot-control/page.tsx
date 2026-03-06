"use client";

import { Suspense } from "react";
import { Layout } from "@/components/layout/Layout";
import BotControl from "@/components/pages/BotControl";

export default function BotControlPage() {
    return (
        <Suspense
            fallback={
                <Layout>
                    <div className="flex h-96 items-center justify-center">
                        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    </div>
                </Layout>
            }
        >
            <Layout>
                <BotControl />
            </Layout>
        </Suspense>
    );
}
