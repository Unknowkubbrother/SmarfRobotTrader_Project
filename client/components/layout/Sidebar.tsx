"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Bot,
  Calendar,
  BookOpen,
  CreditCard,
  Settings,
  HelpCircle,
  ChevronLeft,
  Shield,
  LogOut,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/" },
  { icon: Bot, label: "Bot Control", path: "/bot-control" },
  { icon: Calendar, label: "Profit Calendar", path: "/calendar" },
  { icon: BookOpen, label: "Trading Journal", path: "/journal" },
  { icon: CreditCard, label: "Subscription", path: "/subscription" },
  { icon: Settings, label: "Settings", path: "/settings" },
  { icon: HelpCircle, label: "Support", path: "/support" },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const { user, isAdmin, signOut } = useAuth();

  const handleSignOut = async () => {
    await signOut();
    router.push("/auth");
  };

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen bg-card border-r border-border transition-all duration-300 flex flex-col",
        collapsed ? "w-16" : "w-56"
      )}
    >
      {/* Logo */}
      <div className="flex items-center justify-between h-14 px-4 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <span className="text-lg font-bold text-foreground">RoBot</span>
            {!collapsed && (
              <span className="text-lg font-bold text-primary">Smarf</span>
            )}
          </div>
        </div>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 rounded hover:bg-secondary transition-colors"
        >
          <ChevronLeft
            className={cn(
              "w-4 h-4 text-muted-foreground transition-transform duration-300",
              collapsed && "rotate-180"
            )}
          />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto scrollbar-thin">
        {navItems.map((item) => {
          const isActive = pathname === item.path;
          return (
            <Link
              key={item.path}
              href={item.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200",
                isActive
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
            >
              <item.icon className="w-5 h-5 shrink-0" />
              {!collapsed && (
                <span className="text-sm">{item.label}</span>
              )}
            </Link>
          );
        })}

        {/* Admin Link */}
        {isAdmin && (
          <Link
            href="/admin"
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200",
              pathname === "/admin"
                ? "bg-primary/10 text-primary font-medium"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            )}
          >
            <Shield className="w-5 h-5 shrink-0" />
            {!collapsed && <span className="text-sm">Admin</span>}
          </Link>
        )}
      </nav>

      {/* User & Logout */}
      <div className="p-3 border-t border-border space-y-2">
        {user ? (
          <button
            onClick={handleSignOut}
            className={cn(
              "flex items-center gap-2 w-full p-2 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors",
              collapsed && "justify-center"
            )}
          >
            <LogOut className="w-4 h-4" />
            {!collapsed && <span className="text-sm">Sign Out</span>}
          </button>
        ) : (
          <Link
            href="/auth"
            className={cn(
              "flex items-center gap-2 w-full p-2 rounded-lg bg-primary text-primary-foreground",
              collapsed && "justify-center"
            )}
          >
            <LogOut className="w-4 h-4" />
            {!collapsed && <span className="text-sm">Sign In</span>}
          </Link>
        )}
        <div
          className={cn(
            "flex items-center gap-2 p-2 rounded-lg bg-secondary",
            collapsed && "justify-center"
          )}
        >
          <span className="w-2 h-2 rounded-full bg-success" />
          {!collapsed && (
            <span className="text-sm text-muted-foreground">Online</span>
          )}
        </div>
      </div>
    </aside>
  );
}
