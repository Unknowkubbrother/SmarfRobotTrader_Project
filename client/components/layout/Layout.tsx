"use client";

import { ReactNode } from "react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { LayoutProvider, useLayout } from "@/contexts/LayoutContext";
import { cn } from "@/lib/utils";

function LayoutContent({ children }: { children: ReactNode }) {
  const { collapsed } = useLayout();

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
