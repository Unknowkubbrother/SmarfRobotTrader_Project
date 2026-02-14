"use client";

import { useEffect, useState } from "react";
import { Bell, Check, Trash2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface Notification {
    id: string;
    title: string;
    message: string;
    isRead: boolean;
    relatedLink?: string;
    createdAt: string;
}

import { Layout } from "@/components/layout/Layout";

export default function NotificationsPage() {
    const router = useRouter();
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchNotifications = async () => {
        setLoading(true);
        try {
            const res = await fetch("/notifications");
            if (res.ok) {
                setNotifications(await res.json());
            }
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
            await fetch(`/notifications/${id}/read`, { method: "PATCH" });
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
            await fetch("/notifications/read-all", { method: "PATCH" });
            setNotifications((prev) => prev.map((n) => ({ ...n, isRead: true })));
        } catch (error) {
            console.error("Failed to mark all read", error);
        }
    };

    return (
        <Layout>
            <div className="w-full py-8 space-y-6">
                <div className="flex items-center gap-4 mb-6">
                    <Button variant="ghost" size="icon" onClick={() => router.back()}>
                        <ArrowLeft className="w-5 h-5" />
                    </Button>
                    <div>
                        <h1 className="text-2xl font-bold">Notifications</h1>
                        <p className="text-muted-foreground">Manage your alerts and messages</p>
                    </div>
                    <div className="ml-auto">
                        <Button variant="outline" onClick={markAllAsRead} disabled={notifications.every(n => n.isRead)}>
                            <Check className="w-4 h-4 mr-2" />
                            Mark all as read
                        </Button>
                    </div>
                </div>

                <Card>
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <CardTitle className="text-lg flex items-center gap-2">
                                <Bell className="w-5 h-5 text-primary" />
                                Recent Notifications
                            </CardTitle>
                            <span className="text-xs text-muted-foreground">
                                {notifications.filter(n => !n.isRead).length} unread
                            </span>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-[600px] pr-4">
                            {loading ? (
                                <div className="text-center py-12 text-muted-foreground">Loading...</div>
                            ) : notifications.length === 0 ? (
                                <div className="text-center py-12 text-muted-foreground flex flex-col items-center gap-2">
                                    <Bell className="w-12 h-12 opacity-20" />
                                    <p>No notifications yet</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {notifications.map((notification) => (
                                        <div
                                            key={notification.id}
                                            onClick={() => markAsRead(notification.id, notification.relatedLink)}
                                            className={cn(
                                                "flex items-start gap-4 p-4 rounded-lg border transition-all cursor-pointer hover:bg-accent",
                                                notification.isRead ? "bg-card border-border" : "bg-primary/5 border-primary/20 shadow-sm"
                                            )}
                                        >
                                            <div className={cn(
                                                "mt-1 w-2 h-2 rounded-full shrink-0",
                                                notification.isRead ? "bg-muted-foreground/30" : "bg-primary animate-pulse"
                                            )} />
                                            <div className="flex-1 space-y-1">
                                                <div className="flex items-center justify-between">
                                                    <p className={cn("font-medium", !notification.isRead && "text-primary")}>
                                                        {notification.title}
                                                    </p>
                                                    <span className="text-xs text-muted-foreground">
                                                        {new Date(notification.createdAt).toLocaleString()}
                                                    </span>
                                                </div>
                                                <p className="text-sm text-muted-foreground leading-relaxed">
                                                    {notification.message}
                                                </p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </ScrollArea>
                    </CardContent>
                </Card>
            </div>
        </Layout>
    );
}
