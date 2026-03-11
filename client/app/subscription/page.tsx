"use client";

import { Suspense } from "react";
import { Layout } from "@/components/layout/Layout";
import Subscription from "@/components/pages/Subscription";

export default function SubscriptionPage() {
    return (
        <Layout>
            <Suspense
                fallback={
                    <div className="flex h-96 items-center justify-center">
                        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    </div>
                }
            >
                <Subscription />
            </Suspense>
        </Layout>
    );
}
