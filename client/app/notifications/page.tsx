"use client";

import { useEffect, useState } from "react";
import { Bell, Check, Trash2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

interface Notification {
    id: string;
    title: string;
    message: string;
    isRead: boolean;
    relatedLink?: string;
    createdAt: string;
}

import { Layout } from "@/components/layout/Layout";
import { TrendingUp, TrendingDown, Info, ShieldCheck, Mail } from "lucide-react";

const getNotificationIcon = (title: string, message: string) => {
    const text = (title + message).toLowerCase();
    if (text.includes("profit") || text.includes("earned")) return <TrendingUp className="w-4 h-4 text-success" />;
    if (text.includes("loss") || text.includes("risk")) return <TrendingDown className="w-4 h-4 text-destructive" />;
    if (text.includes("security") || text.includes("password") || text.includes("login")) return <ShieldCheck className="w-4 h-4 text-primary" />;
    if (text.includes("email") || text.includes("verify")) return <Mail className="w-4 h-4 text-warning" />;
    return <Info className="w-4 h-4 text-muted-foreground" />;
};

export default function NotificationsPage() {
    const router = useRouter();
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchNotifications = async () => {
        setLoading(true);
        try {
            const { data } = await api.get<Notification[]>("/notifications");
            setNotifications(data || []);
        } catch (error) {
            console.error("Failed to fetch notifications", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchNotifications();
    }, []);

    const markAsRead = async (id: string, link?: string) => {
        try {
            await api.patch(`/notifications/${id}/read`);
            setNotifications((prev) =>
                prev.map((n) => (n.id === id ? { ...n, isRead: true } : n))
            );
            if (link) router.push(link);
        } catch (error) {
            console.error("Failed to mark read", error);
        }
    };

    const markAllAsRead = async () => {
        try {
            await api.patch("/notifications/read-all");
            setNotifications((prev) => prev.map((n) => ({ ...n, isRead: true })));
        } catch (error) {
            console.error("Failed to mark all read", error);
        }
    };

    return (
        <Layout>
            <div className="w-full max-w-7xl mx-auto py-8 px-4 space-y-8 animate-fade-in">
                {/* Header Section */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div className="flex items-center gap-6">
                        <Button
                            variant="secondary"
                            size="icon"
                            onClick={() => router.back()}
                            className="rounded-xl shadow-sm hover:scale-105 transition-transform"
                        >
                            <ArrowLeft className="w-5 h-5" />
                        </Button>
                        <div className="space-y-1">
                            <h1 className="text-3xl font-bold tracking-tight">Notifications</h1>
                            <div className="flex items-center gap-2 text-muted-foreground">
                                <Bell className="w-4 h-4" />
                                <span>{notifications.filter(n => !n.isRead).length} unread alerts</span>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button
                            variant="outline"
                            onClick={markAllAsRead}
                            disabled={notifications.every(n => n.isRead)}
                            className="rounded-xl border-dashed hover:border-primary hover:text-primary transition-all"
                        >
                            <Check className="w-4 h-4 mr-2" />
                            Mark all as read
                        </Button>
                    </div>
                </div>

                {/* Notifications Feed */}
                <div className="space-y-4">
                    {loading ? (
                        <div className="flex items-center justify-center py-20">
                            <div className="flex flex-col items-center gap-4">
                                <div className="w-10 h-10 border-4 border-primary/20 border-t-primary rounded-full animate-spin" />
                                <p className="text-sm text-muted-foreground">Fetching your alerts...</p>
                            </div>
                        </div>
                    ) : notifications.length === 0 ? (
                        <div className="glass-card flex flex-col items-center justify-center py-24 text-center">
                            <div className="w-20 h-20 rounded-full bg-secondary/50 flex items-center justify-center mb-6">
                                <Bell className="w-10 h-10 text-muted-foreground/30" />
                            </div>
                            <h3 className="text-xl font-semibold mb-2">No notifications yet</h3>
                            <p className="text-muted-foreground max-w-sm">
                                We&apos;ll notify you here when there are updates about your bots, performance, or security.
                            </p>
                        </div>
                    ) : (
                        <div className="grid gap-4">
                            {notifications.map((notification, index) => (
                                <div
                                    key={notification.id}
                                    onClick={() => markAsRead(notification.id, notification.relatedLink)}
                                    className={cn(
                                        "group relative flex items-start gap-5 p-6 rounded-2xl border transition-all cursor-pointer animate-slide-up",
                                        notification.isRead
                                            ? "bg-card/50 border-border hover:bg-accent/30"
                                            : "bg-white border-primary/20 shadow-md hover:shadow-lg ring-1 ring-primary/5"
                                    )}
                                    style={{ animationDelay: `${index * 50}ms` }}
                                >
                                    {/* Type-based Icon */}
                                    <div className={cn(
                                        "w-12 h-12 rounded-xl flex items-center justify-center shrink-0 shadow-sm",
                                        notification.isRead ? "bg-muted text-muted-foreground" : "bg-primary/10 text-primary"
                                    )}>
                                        {getNotificationIcon(notification.title || "", notification.message || "")}
                                    </div>

                                    <div className="flex-1 space-y-2 min-w-0">
                                        <div className="flex items-center justify-between gap-4">
                                            <div className="flex items-center gap-3 min-w-0">
                                                <h3 className={cn("font-semibold text-lg truncate", !notification.isRead && "text-primary")}>
                                                    {notification.title}
                                                </h3>
                                                {!notification.isRead && (
                                                    <span className="w-2 h-2 rounded-full bg-primary animate-pulse shrink-0" />
                                                )}
                                            </div>
                                            <span className="text-xs font-medium text-muted-foreground whitespace-nowrap bg-secondary/50 px-2 py-1 rounded-md">
                                                {new Date(notification.createdAt).toLocaleString(undefined, {
                                                    month: 'short',
                                                    day: 'numeric',
                                                    hour: '2-digit',
                                                    minute: '2-digit'
                                                })}
                                            </span>
                                        </div>
                                        <p className="text-muted-foreground leading-relaxed text-sm lg:text-base line-clamp-3 group-hover:line-clamp-none transition-all">
                                            {notification.message}
                                        </p>
                                    </div>

                                    {/* Action Hover Indicator */}
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                                            <Check className="w-4 h-4" />
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </Layout>
    );
}
