import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { toast } from "sonner";

export interface Mt5BrokerServerEntry {
  broker_name: string;
  server_names: string[];
}

export interface Mt5ServerCatalog {
  brokers: Mt5BrokerServerEntry[];
  all_servers: string[];
}

const EMPTY_CATALOG: Mt5ServerCatalog = {
  brokers: [],
  all_servers: [],
};

export function useMt5ServerCatalog() {
  const [catalog, setCatalog] = useState<Mt5ServerCatalog>(EMPTY_CATALOG);
  const [loading, setLoading] = useState(false);

  const fetchCatalog = useCallback(async () => {
    try {
      setLoading(true);
      const response = await api.get("/trading/mt5_server_catalog");
      const payload = response?.data?.data ?? {};

      const brokers = Array.isArray(payload?.brokers)
        ? payload.brokers
            .map((entry: any) => ({
              broker_name: String(entry?.broker_name ?? "").trim(),
              server_names: Array.isArray(entry?.server_names)
                ? entry.server_names
                    .map((name: any) => String(name ?? "").trim())
                    .filter(Boolean)
                : [],
            }))
            .filter((entry: Mt5BrokerServerEntry) => entry.broker_name && entry.server_names.length > 0)
        : [];

      const all_servers = Array.isArray(payload?.all_servers)
        ? payload.all_servers.map((name: any) => String(name ?? "").trim()).filter(Boolean)
        : [];

      setCatalog({ brokers, all_servers });
    } catch (error) {
      console.error("Failed to fetch MT5 server catalog:", error);
      toast.error("Failed to load MT5 broker/server list");
      setCatalog(EMPTY_CATALOG);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCatalog();
  }, [fetchCatalog]);

  return {
    catalog,
    loading,
    refetch: fetchCatalog,
  };
}
