"use client";

import { useEffect, useState } from "react";
import Turnstile, { useTurnstile } from "react-turnstile";

interface TurnstileWidgetProps {
    onVerify: (token: string) => void;
    onError?: (error: any) => void;
    action?: string;
}

export default function TurnstileWidget({ onVerify, onError, action }: TurnstileWidgetProps) {
    const siteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY || "1x00000000000000000000AA"; // Default to testing key

    return (
        <div className="w-full flex justify-center my-4">
            <Turnstile
                sitekey={siteKey}
                onVerify={onVerify}
                onError={onError}
                action={action}
                theme="light"
            />
        </div>
    );
}
