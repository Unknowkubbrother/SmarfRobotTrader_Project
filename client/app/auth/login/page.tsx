"use client";

import { Suspense } from "react";
import Login from "@/components/pages/auth/Login";

export default function LoginPage() {
    return (
        <Suspense fallback={<div className="min-h-screen bg-background" />}>
            <Login />
        </Suspense>
    );
}
