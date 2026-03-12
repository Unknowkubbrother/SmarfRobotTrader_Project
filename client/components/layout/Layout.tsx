"use client";

import { ReactNode, useEffect } from "react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { LayoutProvider, useLayout } from "@/contexts/LayoutContext";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { usePathname, useRouter } from "next/navigation";

function LayoutContent({ children }: { children: ReactNode }) {
  const { collapsed } = useLayout();
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading || user) {
      return;
    }

    const params = new URLSearchParams();
    const search = typeof window !== "undefined" ? window.location.search : "";
    params.set("next", search ? `${pathname}${search}` : pathname);
    router.replace(`/auth/login?${params.toString()}`);
  }, [loading, pathname, router, user]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <>
      <Sidebar />
      <div className={cn(
        "transition-all duration-300",
        collapsed ? "pl-16" : "pl-56"
      )}>
        <Header />
        <main className="p-6">{children}</main>
      </div>
    </>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <LayoutProvider>
      <LayoutContent>{children}</LayoutContent>
    </LayoutProvider>
  );
}
