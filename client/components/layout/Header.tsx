"use client";

import { useEffect, useState, useRef } from "react";
import { Bell, Search, User, Settings, LogOut, CreditCard, Bot, LayoutDashboard } from "lucide-react";
import { Button } from "@/components/ui/button";
import TickerTagWidget from "../dashboard/TickerTagWidget";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface SearchResult {
  id: string;
  type: "bot" | "account";
  label: string;
  subLabel?: string;
  link: string;
}

interface Notification {
  id: string;
  title: string;
  message: string;
  isRead: boolean;
  relatedLink?: string;
  createdAt: string;
}

export function Header() {
  const { user, signOut } = useAuth();
  const router = useRouter();

  // Search State
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // Notification State
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);

  // Close search results when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fetch Notifications
  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        const { data } = await api.get<Notification[]>("/notifications");
        setNotifications(data || []);
        setUnreadCount((data || []).filter((n: Notification) => !n.isRead).length);
      } catch (error) {
        console.error("Failed to fetch notifications", error);
      }
    };

    fetchNotifications();
    // Poll every 60s
    const interval = setInterval(fetchNotifications, 60000);
    return () => clearInterval(interval);
  }, []);

  // Search Logic
  useEffect(() => {
    const debounceTimer = setTimeout(async () => {
      if (query.length < 2) {
        setResults([]);
        return;
      }

      setIsSearching(true);
      try {
        const { data } = await api.get<SearchResult[]>("/search", {
          params: { q: query },
        });
        setResults(data || []);
        setShowResults(true);
      } catch (error) {
        console.error("Search failed", error);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(debounceTimer);
  }, [query]);

  const handleSignOut = async () => {
    await signOut();
    router.push("/auth");
  };

  const markAsRead = async (id: string, link?: string) => {
    try {
      await api.patch(`/notifications/${id}/read`);
      setNotifications(prev => prev.map(n => n.id === id ? { ...n, isRead: true } : n));
      setUnreadCount(prev => Math.max(0, prev - 1));
      if (link) router.push(link);
    } catch (error) {
      console.error("Failed to mark read", error);
    }
  };

  const getAvatarSrc = (url: string | null | undefined) => {
    if (!url) return "";
    if (url.startsWith("http") || url.startsWith("data:")) return url;
    if (url.startsWith("/static")) return `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${url}`;
    return url;
  };

  return (
    <div className="sticky top-0 z-30 bg-card border-b border-border shadow-sm">
      {/* Top Row: Search & Actions */}
      <header className="h-16 flex items-center justify-between px-6 gap-6">
        {/* Search */}
        <div ref={searchRef} className="relative flex items-center gap-4 w-96 shrink-0">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search bots, accounts..."
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                if (e.target.value.length >= 2) setShowResults(true);
              }}
              onFocus={() => {
                if (query.length >= 2) setShowResults(true);
              }}
              className="w-full h-10 pl-10 pr-4 rounded-full bg-secondary border-0 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all"
            />
            {isSearching && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              </div>
            )}
          </div>

          {/* Search Dropdown */}
          {showResults && (
            <div className="absolute top-full left-0 right-0 mt-2 z-50 bg-popover border border-border rounded-xl shadow-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200">
              <div className="p-1">
                {results.length > 0 ? (
                  results.map((result) => (
                    <Link
                      key={result.id}
                      href={result.link}
                      onClick={() => setShowResults(false)}
                      className="flex items-center gap-3 p-3 rounded-lg hover:bg-accent transition-colors"
                    >
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                        {result.type === "bot" ? (
                          <Bot className="w-4 h-4 text-primary" />
                        ) : (
                          <LayoutDashboard className="w-4 h-4 text-primary" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium leading-none truncate">{result.label}</p>
                        {result.subLabel && (
                          <p className="text-xs text-muted-foreground mt-1 truncate">{result.subLabel}</p>
                        )}
                      </div>
                    </Link>
                  ))
                ) : (
                  <div className="p-4 text-center text-sm text-muted-foreground">
                    No results found for "{query}"
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Notifications */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="relative rounded-full">
                <Bell className="w-5 h-5" />
                {unreadCount > 0 && (
                  <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-destructive animate-pulse" />
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80">
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">Notifications</p>
                  <p className="text-xs text-muted-foreground">
                    You have {unreadCount} unread messages
                  </p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <ScrollArea className="h-[300px]">
                {notifications.length === 0 ? (
                  <div className="p-4 text-center text-sm text-muted-foreground">
                    No notifications
                  </div>
                ) : (
                  <div className="flex flex-col gap-1 p-1">
                    {notifications.map((notification) => (
                      <DropdownMenuItem
                        key={notification.id}
                        className={cn(
                          "flex flex-col items-start gap-1 p-3 cursor-pointer",
                          !notification.isRead && "bg-accent/50"
                        )}
                        onClick={() => markAsRead(notification.id, notification.relatedLink)}
                      >
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className={cn("text-sm font-medium", !notification.isRead && "text-primary")}>
                            {notification.title}
                          </span>
                          <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                            {new Date(notification.createdAt).toLocaleDateString()}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-2">
                          {notification.message}
                        </p>
                      </DropdownMenuItem>
                    ))}
                  </div>
                )}
              </ScrollArea>
              <DropdownMenuSeparator />
              <div className="p-2">
                <Button variant="ghost" className="w-full text-xs h-8" asChild>
                  <Link href="/notifications">View all notifications</Link>
                </Button>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Profile */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center overflow-hidden">
                  {user?.avatar_url ? (
                    <img src={getAvatarSrc(user?.avatar_url)} alt="Avatar" className="w-full h-full object-cover" />
                  ) : (
                    <User className="w-4 h-4 text-muted-foreground" />
                  )}
                </div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">{user?.username || "User"}</p>
                  <p className="text-xs text-muted-foreground leading-none">{user?.email}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link href="/settings">
                  <User className="w-4 h-4 mr-2" />
                  Profile
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/subscription">
                  <CreditCard className="w-4 h-4 mr-2" />
                  Billing
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleSignOut} className="text-destructive focus:text-destructive">
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Bottom Row: Ticker Tags */}
      <div className="h-14 border-t border-border/50 bg-card/50 backdrop-blur-sm flex items-center overflow-hidden">
        <TickerTagWidget />
      </div>
    </div>
  );
}
