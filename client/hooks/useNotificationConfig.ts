import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import type { Database } from "@/lib/integrations/supabase/types";

type NotificationConfig = Database["public"]["Tables"]["notification_configs"]["Row"];

export function useNotificationConfig() {
  const { user } = useAuth();
  const [config, setConfig] = useState<NotificationConfig | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchConfig = useCallback(async () => {
    if (!user) {
      setConfig(null);
      setLoading(false);
      return;
    }

    try {
      const { data, error } = await supabase
        .from("notification_configs")
        .select("*")
        .eq("user_id", user.id)
        .maybeSingle();

      if (error) throw error;

      setConfig(data);
    } catch (error) {
      console.error("Error fetching notification config:", error);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const updateConfig = async (updates: Partial<NotificationConfig>) => {
    if (!user) return false;

    try {
      const { error } = await supabase
        .from("notification_configs")
        .update(updates)
        .eq("user_id", user.id);

      if (error) throw error;

      await fetchConfig();
      toast.success("Notification settings updated");
      return true;
    } catch (error) {
      console.error("Error updating notification config:", error);
      toast.error("Failed to update notification settings");
      return false;
    }
  };

  return {
    config,
    loading,
    refetch: fetchConfig,
    updateConfig,
  };
}
