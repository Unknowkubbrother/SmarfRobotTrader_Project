import { useState, useEffect } from "react";
import { Plus, Upload, Tag, Clock, Trash2, Boxes } from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface BotVersion {
  id: string;
  label: string | null;
  version_tag: string | null;
  symbol: string | null;
  timeframe: string | null;
  docker_image_id: string | null;
  release_notes: string[];
  release_date: string | null;
  usage_count: number;
}

export function AdminBotVersions() {
  const [versions, setVersions] = useState<BotVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    label: "",
    version_tag: "",
    symbol: "XAUUSD",
    timeframe: "H1",
    docker_image_id: "",
    release_notes: "",
  });

  useEffect(() => {
    fetchVersions();
  }, []);

  const fetchVersions = async () => {
    try {
      setLoading(true);
      const { data } = await api.get<BotVersion[]>("/admin/bot-versions");
      setVersions(data || []);
    } catch (error: any) {
      console.error("Error fetching versions:", error);
      toast.error(error?.message || "Failed to load bot versions");
    } finally {
      setLoading(false);
    }
  };

  const handleAddVersion = async () => {
    if (!formData.label.trim() || !formData.version_tag.trim()) {
      toast.error("Please fill in label and version tag");
      return;
    }

    setSubmitting(true);
    try {
      const releaseNotes = formData.release_notes
        .split("\n")
        .map((note) => note.trim())
        .filter(Boolean);

      await api.post("/admin/bot-versions", {
        label: formData.label.trim(),
        version_tag: formData.version_tag.trim(),
        symbol: formData.symbol.trim() || null,
        timeframe: formData.timeframe.trim() || null,
        docker_image_id: formData.docker_image_id.trim() || null,
        release_notes: releaseNotes,
      });

      toast.success("Bot version added successfully");
      setShowAddDialog(false);
      setFormData({
        label: "",
        version_tag: "",
        symbol: "XAUUSD",
        timeframe: "H1",
        docker_image_id: "",
        release_notes: "",
      });
      await fetchVersions();
    } catch (error: any) {
      console.error("Error adding version:", error);
      toast.error(error?.message || "Failed to add bot version");
    } finally {
      setSubmitting(false);
    }
  };

  const deleteVersion = async (id: string) => {
    setDeletingId(id);
    try {
      await api.delete(`/admin/bot-versions/${id}`);
      toast.success("Version deleted");
      await fetchVersions();
    } catch (error: any) {
      console.error("Error deleting version:", error);
      toast.error(error?.message || "Failed to delete version");
    } finally {
      setDeletingId(null);
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
                  onChange={(event) => setFormData({ ...formData, label: event.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="version_tag">Version Tag *</Label>
                <Input
                  id="version_tag"
                  placeholder="e.g., v2.0.0"
                  value={formData.version_tag}
                  onChange={(event) => setFormData({ ...formData, version_tag: event.target.value })}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="symbol">Symbol</Label>
                  <Input
                    id="symbol"
                    value={formData.symbol}
                    onChange={(event) => setFormData({ ...formData, symbol: event.target.value })}
                  />
                </div>
                <div>
                  <Label htmlFor="timeframe">Timeframe</Label>
                  <Input
                    id="timeframe"
                    value={formData.timeframe}
                    onChange={(event) => setFormData({ ...formData, timeframe: event.target.value })}
                  />
                </div>
              </div>
              <div>
                <Label htmlFor="docker_image_id">Docker Image ID</Label>
                <Input
                  id="docker_image_id"
                  placeholder="registry/repo:tag"
                  value={formData.docker_image_id}
                  onChange={(event) => setFormData({ ...formData, docker_image_id: event.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="release_notes">Release Notes (one line each)</Label>
                <Textarea
                  id="release_notes"
                  placeholder="Improved signal filters&#10;Lower drawdown"
                  value={formData.release_notes}
                  onChange={(event) => setFormData({ ...formData, release_notes: event.target.value })}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowAddDialog(false)} disabled={submitting}>
                Cancel
              </Button>
              <Button onClick={handleAddVersion} disabled={submitting}>
                {submitting ? "Adding..." : "Add Version"}
              </Button>
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
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h3 className="font-semibold">{version.label || "Unnamed Version"}</h3>
                    <Badge variant="outline">
                      <Tag className="w-3 h-3 mr-1" />
                      {version.version_tag || "-"}
                    </Badge>
                    <Badge variant="secondary" className="flex items-center gap-1">
                      <Boxes className="w-3 h-3" />
                      In use {version.usage_count}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground flex-wrap">
                    <span className="font-mono">{version.symbol || "-"}</span>
                    <span>{version.timeframe || "-"}</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {version.release_date ? new Date(version.release_date).toLocaleDateString() : "N/A"}
                    </span>
                  </div>
                  {version.docker_image_id && (
                    <p className="text-xs text-muted-foreground mt-2 font-mono">
                      {version.docker_image_id}
                    </p>
                  )}
                  {version.release_notes.length > 0 && (
                    <ul className="text-sm text-muted-foreground mt-2 list-disc list-inside">
                      {version.release_notes.map((note, index) => (
                        <li key={`${version.id}-${index}`}>{note}</li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteVersion(version.id)}
                    disabled={deletingId === version.id}
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

