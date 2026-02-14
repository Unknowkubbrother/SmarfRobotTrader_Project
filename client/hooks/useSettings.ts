import { useState, useCallback } from "react";
import { api } from "@/lib/api";
import { toast } from "sonner";

export interface NotificationConfig {
    emailNotificationEnable: boolean;
    alertMarginLevelThreshold: number | null;
    alertProfitTarget: number | null;
    alertLossLimit: number | null;
    enableWeeklySummary: boolean;
    enableMonthlySummary: boolean;
    lineNotifyToken: string | null;
    discordWebhookUrl: string | null;
}

export interface UserProfile {
    id: string;
    username: string;
    email: string;
    recoveryEmail: string | null;
    avatarUrl: string | null;
    notificationConfig: NotificationConfig;
}

export interface ActivityLog {
    id: string;
    date: string;
    ip: string | null;
    device: string | null;
    location: string | null;
    topic: string | null;
}

export function useSettings() {
    const [loading, setLoading] = useState(false);
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);

    const fetchProfile = useCallback(async () => {
        try {
            setLoading(true);
            const response = await api.get("/settings/profile");
            setProfile(response.data);
        } catch (error) {
            console.error("Failed to fetch profile:", error);
            toast.error("Failed to load settings");
        } finally {
            setLoading(false);
        }
    }, []);

    const fetchActivityLogs = useCallback(async () => {
        try {
            const response = await api.get("/settings/activity-logs");
            setActivityLogs(response.data);
        } catch (error) {
            console.error("Failed to fetch activity logs:", error);
        }
    }, []);

    const updateProfile = async (data: Partial<UserProfile>) => {
        try {
            setLoading(true);
            const response = await api.patch("/settings/profile", data);
            setProfile(response.data);
            toast.success("Profile updated successfully");
            return true;
        } catch (error: any) {
            console.error("Failed to update profile:", error);
            toast.error(error.response?.data?.detail || "Failed to update profile");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const updatePassword = async (currentPassword: string, newPassword: string) => {
        try {
            setLoading(true);
            await api.patch("/settings/password", { currentPassword, newPassword });
            toast.success("Password updated successfully");
            return true;
        } catch (error: any) {
            console.error("Failed to update password:", error);
            toast.error(error.response?.data?.detail || "Failed to update password");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const updateNotifications = async (data: Partial<NotificationConfig>) => {
        try {
            setLoading(true);
            await api.patch("/settings/notifications", data);

            // Update local state
            if (profile) {
                setProfile({
                    ...profile,
                    notificationConfig: {
                        ...profile.notificationConfig,
                        ...data
                    }
                });
            }

            toast.success("Notification preferences updated");
            return true;
        } catch (error: any) {
            console.error("Failed to update notifications:", error);
            toast.error(error.response?.data?.detail || "Failed to update notifications");
            return false;
        } finally {
            setLoading(false);
        }
    };

    return {
        loading,
        profile,
        activityLogs,
        fetchProfile,
        fetchActivityLogs,
        updateProfile,
        updatePassword,
        updateNotifications
    };
}
