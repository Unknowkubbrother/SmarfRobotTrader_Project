import { useState, useEffect } from "react";
import { Plus, Upload, Tag, Clock, Edit, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/integrations/supabase/client";
import { toast } from "sonner";

interface BotVersion {
  id: string;
  label: string;
  version_tag: string;
  symbol: string;
  timeframe: string | null;
  release_notes: string | null;
  release_date: string | null;
  is_active: boolean | null;
}

export function AdminBotVersions() {
  const [versions, setVersions] = useState<BotVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [formData, setFormData] = useState({
    label: "",
    version_tag: "",
    symbol: "XAUUSD",
    timeframe: "H1",
    release_notes: "",
  });

  useEffect(() => {
    fetchVersions();
  }, []);

  const fetchVersions = async () => {
    try {
      const { data, error } = await supabase
        .from("bot_versions")
        .select("*")
        .order("release_date", { ascending: false });

      if (error) throw error;
      setVersions(data || []);
    } catch (error) {
      console.error("Error fetching versions:", error);
      toast.error("Failed to load bot versions");
    } finally {
      setLoading(false);
    }
  };

  const handleAddVersion = async () => {
    if (!formData.label || !formData.version_tag) {
      toast.error("Please fill in all required fields");
      return;
    }

    try {
      const { error } = await supabase.from("bot_versions").insert({
        label: formData.label,
        version_tag: formData.version_tag,
        symbol: formData.symbol,
        timeframe: formData.timeframe,
        release_notes: formData.release_notes || null,
      });

      if (error) throw error;
      toast.success("Bot version added successfully");
      setShowAddDialog(false);
      setFormData({ label: "", version_tag: "", symbol: "XAUUSD", timeframe: "H1", release_notes: "" });
      fetchVersions();
    } catch (error) {
      console.error("Error adding version:", error);
      toast.error("Failed to add bot version");
    }
  };

  const toggleVersionStatus = async (id: string, currentStatus: boolean) => {
    try {
      const { error } = await supabase
        .from("bot_versions")
        .update({ is_active: !currentStatus })
        .eq("id", id);

      if (error) throw error;
      toast.success(`Version ${!currentStatus ? "activated" : "deactivated"}`);
      fetchVersions();
    } catch (error) {
      console.error("Error updating version:", error);
      toast.error("Failed to update version");
    }
  };

  const deleteVersion = async (id: string) => {
    try {
      const { error } = await supabase.from("bot_versions").delete().eq("id", id);

      if (error) throw error;
      toast.success("Version deleted");
      fetchVersions();
    } catch (error) {
      console.error("Error deleting version:", error);
      toast.error("Failed to delete version");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Bot Versions</h2>
        <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Add Version
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Bot Version</DialogTitle>
              <DialogDescription>
                Create a new bot version for deployment
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label htmlFor="label">Label *</Label>
                <Input
                  id="label"
                  placeholder="e.g., Gold Scalper v2"
                  value={formData.label}
                  onChange={(e) => setFormData({ ...formData, label: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="version_tag">Version Tag *</Label>
                <Input
                  id="version_tag"
                  placeholder="e.g., v2.0.0"
                  value={formData.version_tag}
                  onChange={(e) => setFormData({ ...formData, version_tag: e.target.value })}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="symbol">Symbol</Label>
                  <Input
                    id="symbol"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                  />
                </div>
                <div>
                  <Label htmlFor="timeframe">Timeframe</Label>
                  <Input
                    id="timeframe"
                    value={formData.timeframe}
                    onChange={(e) => setFormData({ ...formData, timeframe: e.target.value })}
                  />
                </div>
              </div>
              <div>
                <Label htmlFor="release_notes">Release Notes</Label>
                <Textarea
                  id="release_notes"
                  placeholder="Describe changes in this version..."
                  value={formData.release_notes}
                  onChange={(e) => setFormData({ ...formData, release_notes: e.target.value })}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddVersion}>Add Version</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid gap-4">
        {versions.length === 0 ? (
          <div className="glass-card p-8 text-center">
            <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No bot versions yet</p>
            <p className="text-sm text-muted-foreground">Add your first bot version to get started</p>
          </div>
        ) : (
          versions.map((version) => (
            <div key={version.id} className="glass-card p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold">{version.label}</h3>
                    <Badge variant="outline">
                      <Tag className="w-3 h-3 mr-1" />
                      {version.version_tag}
                    </Badge>
                    {version.is_active ? (
                      <Badge className="bg-success/10 text-success">Active</Badge>
                    ) : (
                      <Badge variant="secondary">Inactive</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span className="font-mono">{version.symbol}</span>
                    <span>{version.timeframe}</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {version.release_date ? new Date(version.release_date).toLocaleDateString() : 'N/A'}
                    </span>
                  </div>
                  {version.release_notes && (
                    <p className="text-sm text-muted-foreground mt-2">{version.release_notes}</p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => toggleVersionStatus(version.id, version.is_active ?? false)}
                  >
                    {version.is_active ? "Deactivate" : "Activate"}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteVersion(version.id)}
                    className="text-destructive hover:text-destructive"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
