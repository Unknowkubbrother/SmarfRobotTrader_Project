"use client";

import { Suspense } from "react";
import ForgotPassword from "@/components/pages/auth/ForgotPassword";

export default function ForgotPasswordPage() {
    return (
        <Suspense fallback={<div className="min-h-screen bg-background" />}>
            <ForgotPassword />
        </Suspense>
    );
}
