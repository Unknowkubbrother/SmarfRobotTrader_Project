"use client";

import { createContext, useContext, useState, ReactNode } from "react";

interface LayoutContextType {
    collapsed: boolean;
    setCollapsed: (collapsed: boolean) => void;
}

const LayoutContext = createContext<LayoutContextType | undefined>(undefined);

export function LayoutProvider({ children }: { children: ReactNode }) {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <LayoutContext.Provider value={{ collapsed, setCollapsed }}>
            {children}
        </LayoutContext.Provider>
    );
}

export function useLayout() {
    const context = useContext(LayoutContext);
    if (context === undefined) {
        throw new Error("useLayout must be used within a LayoutProvider");
    }
    return context;
}
