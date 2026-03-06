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
    discordWebhookUrl?: string | null;
    discordWebhookDisplay: string | null;
    hasDiscordWebhook: boolean;
}

export interface UserProfile {
    id: string;
    username: string;
    email: string;
    recoveryEmail: string | null;
    avatarUrl: string | null;
    notificationConfig: NotificationConfig;
    hasPassword: boolean;
}

export interface ActivityLog {
    id: string;
    date: string;
    ip: string | null;
    device: string | null;
    location: string | null;
    topic: string | null;
}

export interface UpdateProfilePayload {
    username?: string;
    email?: string;
    recoveryEmail?: string | null;
    avatarUrl?: string | null;
    otp?: string;
}

export type SecurityOtpPurpose = "password_change" | "profile_change";

export function useSettings() {
    const [loading, setLoading] = useState(false);
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);

    const fetchProfile = useCallback(async () => {
        try {
            setLoading(true);
            const response = await api.get("/settings/profile");
            setProfile(response.data);
        } catch (error: any) {
            console.error("Failed to fetch profile:", error);
            toast.error(error.message || "Failed to load settings");
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

    const updateProfile = async (data: UpdateProfilePayload) => {
        try {
            setLoading(true);
            const response = await api.patch("/settings/profile", data);
            setProfile(response.data);
            toast.success("Profile updated successfully");
            return true;
        } catch (error: any) {
            console.error("Failed to update profile:", error);
            toast.error(error.response?.data?.detail || error.message || "Failed to update profile");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const requestSecurityOtp = async (purpose: SecurityOtpPurpose = "password_change") => {
        try {
            setLoading(true);
            const { data } = await api.post("/settings/security/otp", { purpose });
            toast.success(data.message);
            return data;
        } catch (error: any) {
            console.error("Failed to request OTP:", error);
            toast.error(error.response?.data?.detail || error.message || "Failed to send OTP");
            return null;
        } finally {
            setLoading(false);
        }
    };

    const updatePassword = async (newPassword: string, otp: string) => {
        try {
            setLoading(true);
            await api.patch("/settings/password", { newPassword, otp });
            toast.success("Password updated successfully");

            if (profile) {
                setProfile({ ...profile, hasPassword: true });
            }

            return true;
        } catch (error: any) {
            console.error("Failed to update password:", error);
            toast.error(error.response?.data?.detail || error.message || "Failed to update password");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const updateNotifications = async (data: Partial<NotificationConfig>) => {
        try {
            setLoading(true);
            const response = await api.patch<NotificationConfig>("/settings/notifications", data);

            // Update local state
            if (profile) {
                setProfile({
                    ...profile,
                    notificationConfig: response.data
                });
            }

            toast.success("Notification preferences updated");
            return true;
        } catch (error: any) {
            console.error("Failed to update notifications:", error);
            toast.error(error.response?.data?.detail || error.message || "Failed to update notifications");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const uploadAvatar = async (file: File) => {
        try {
            setLoading(true);
            const formData = new FormData();
            formData.append("file", file);

            const { data } = await api.post("/settings/profile/avatar", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            if (profile) {
                setProfile({ ...profile, avatarUrl: data.avatarUrl });
            }

            toast.success("Profile photo updated");
            return true;
        } catch (error: any) {
            console.error(error);
            toast.error(error.response?.data?.detail || error.message || "Failed to upload profile photo");
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
        requestSecurityOtp,
        updateNotifications,
        uploadAvatar
    };
}
