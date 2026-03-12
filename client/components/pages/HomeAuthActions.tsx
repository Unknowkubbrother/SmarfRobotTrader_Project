"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";

interface HomeAuthActionsProps {
    placement: "header" | "hero";
}

export default function HomeAuthActions({ placement }: HomeAuthActionsProps) {
    const { user, loading } = useAuth();

    if (placement === "header") {
        if (loading) {
            return <div className="h-11 w-40 rounded-full bg-secondary/70 animate-pulse" />;
        }

        if (user) {
            return (
                <Button asChild size="lg" className="rounded-full px-6">
                    <Link href="/dashboard">
                        Go to Dashboard
                        <ArrowRight className="h-4 w-4" />
                    </Link>
                </Button>
            );
        }

        return (
            <>
                <Link
                    href="/auth/login"
                    className="hidden rounded-full border border-border px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground sm:inline-flex"
                >
                    Sign In
                </Link>
                <Button asChild size="lg" className="rounded-full px-6">
                    <Link href="/auth/register">
                        Get Started
                        <ArrowRight className="h-4 w-4" />
                    </Link>
                </Button>
            </>
        );
    }

    if (loading) {
        return (
            <>
                <div className="h-14 w-56 rounded-full bg-secondary/70 animate-pulse" />
                <div className="h-14 w-36 rounded-full bg-secondary/40 animate-pulse" />
            </>
        );
    }

    if (user) {
        return (
            <>
                <Button asChild size="xl" className="rounded-full px-8">
                    <Link href="/dashboard">
                        Open Dashboard
                        <ArrowRight className="h-4 w-4" />
                    </Link>
                </Button>
                <Button asChild size="xl" variant="outline" className="rounded-full px-8">
                    <Link href="/bot-control">Manage Bots</Link>
                </Button>
            </>
        );
    }

    return (
        <>
            <Button asChild size="xl" className="rounded-full px-8">
                <Link href="/auth/register">
                    Start for Free
                    <ArrowRight className="h-4 w-4" />
                </Link>
            </Button>
            <Button asChild size="xl" variant="outline" className="rounded-full px-8">
                <Link href="/auth/login">Sign In</Link>
            </Button>
        </>
    );
}
