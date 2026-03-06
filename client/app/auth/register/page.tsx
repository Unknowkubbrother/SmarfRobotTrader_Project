"use client";

import { Suspense } from "react";
import Register from "@/components/pages/auth/Register";

export default function RegisterPage() {
    return (
        <Suspense fallback={<div className="min-h-screen bg-background" />}>
            <Register />
        </Suspense>
    );
}
