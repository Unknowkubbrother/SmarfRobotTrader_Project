import { useEffect, useState } from "react";
import { BellRing, Boxes, Clock, PenSquare, Plus, Power, Rocket, Tag, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
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
  is_active: boolean;
  release_notes: string[];
  release_date: string | null;
  usage_count: number;
}

interface VersionForm {
  label: string;
  version_tag: string;
  symbol: string;
  timeframe: string;
  docker_image_id: string;
  release_notes: string;
}

interface PublishUpdateForm {
  docker_image_id: string;
  version_tag: string;
  release_notes: string;
  notify_users: boolean;
}

const EMPTY_FORM: VersionForm = {
  label: "",
  version_tag: "",
  symbol: "XAUUSD",
  timeframe: "H1",
  docker_image_id: "",
  release_notes: "",
};

const EMPTY_PUBLISH_FORM: PublishUpdateForm = {
  docker_image_id: "",
  version_tag: "",
  release_notes: "",
  notify_users: true,
};

export function AdminBotVersions() {
  const [versions, setVersions] = useState<BotVersion[]>([]);
  const [loading, setLoading] = useState(true);

  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showPublishDialog, setShowPublishDialog] = useState(false);

  const [addSubmitting, setAddSubmitting] = useState(false);
  const [editSubmitting, setEditSubmitting] = useState(false);
  const [publishSubmitting, setPublishSubmitting] = useState(false);

  const [actionKey, setActionKey] = useState<string | null>(null);
  const [editingVersion, setEditingVersion] = useState<BotVersion | null>(null);
  const [publishingVersion, setPublishingVersion] = useState<BotVersion | null>(null);

  const [addForm, setAddForm] = useState<VersionForm>(EMPTY_FORM);
  const [editForm, setEditForm] = useState<VersionForm>(EMPTY_FORM);
  const [publishForm, setPublishForm] = useState<PublishUpdateForm>(EMPTY_PUBLISH_FORM);

  useEffect(() => {
    fetchVersions();
  }, []);

  const toReleaseNotes = (value: string) =>
    value
      .split("\n")
      .map((note) => note.trim())
      .filter(Boolean);

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
    if (!addForm.label.trim() || !addForm.version_tag.trim()) {
      toast.error("Please fill in label and version tag");
      return;
    }

    setAddSubmitting(true);
    try {
      await api.post("/admin/bot-versions", {
        label: addForm.label.trim(),
        version_tag: addForm.version_tag.trim(),
        symbol: addForm.symbol.trim() || null,
        timeframe: addForm.timeframe.trim() || null,
        docker_image_id: addForm.docker_image_id.trim() || null,
        is_active: true,
        release_notes: toReleaseNotes(addForm.release_notes),
      });

      toast.success("Bot version added successfully");
      setShowAddDialog(false);
      setAddForm(EMPTY_FORM);
      await fetchVersions();
    } catch (error: any) {
      console.error("Error adding version:", error);
      toast.error(error?.message || "Failed to add bot version");
    } finally {
      setAddSubmitting(false);
    }
  };

  const openEditDialog = (version: BotVersion) => {
    setEditingVersion(version);
    setEditForm({
      label: version.label || "",
      version_tag: version.version_tag || "",
      symbol: version.symbol || "",
      timeframe: version.timeframe || "",
      docker_image_id: version.docker_image_id || "",
      release_notes: (version.release_notes || []).join("\n"),
    });
    setShowEditDialog(true);
  };

  const handleUpdateVersion = async () => {
    if (!editingVersion) return;
    if (!editForm.label.trim() || !editForm.version_tag.trim()) {
      toast.error("Please fill in label and version tag");
      return;
    }

    setEditSubmitting(true);
    try {
      await api.patch(`/admin/bot-versions/${editingVersion.id}`, {
        label: editForm.label.trim(),
        version_tag: editForm.version_tag.trim(),
        symbol: editForm.symbol.trim() || null,
        timeframe: editForm.timeframe.trim() || null,
        docker_image_id: editForm.docker_image_id.trim() || null,
        release_notes: toReleaseNotes(editForm.release_notes),
      });

      toast.success("Bot version updated");
      setShowEditDialog(false);
      setEditingVersion(null);
      await fetchVersions();
    } catch (error: any) {
      console.error("Error updating version:", error);
      toast.error(error?.message || "Failed to update bot version");
    } finally {
      setEditSubmitting(false);
    }
  };

  const openPublishDialog = (version: BotVersion) => {
    setPublishingVersion(version);
    setPublishForm({
      docker_image_id: version.docker_image_id || "",
      version_tag: version.version_tag || "",
      release_notes: (version.release_notes || []).join("\n"),
      notify_users: true,
    });
    setShowPublishDialog(true);
  };

  const handlePublishUpdate = async () => {
    if (!publishingVersion) return;
    if (!publishForm.docker_image_id.trim()) {
      toast.error("Please provide docker image id");
      return;
    }

    setPublishSubmitting(true);
    try {
      const { data } = await api.post(`/admin/bot-versions/${publishingVersion.id}/publish-update`, {
        docker_image_id: publishForm.docker_image_id.trim(),
        version_tag: publishForm.version_tag.trim() || null,
        release_notes: toReleaseNotes(publishForm.release_notes),
        notify_users: publishForm.notify_users,
      });

      const usersNotified = Number(data?.users_notified || 0);
      const emailsSent = Number(data?.emails_sent || 0);
      toast.success(
        `Update published. Notified ${usersNotified} user(s), sent ${emailsSent} email(s).`
      );
      setShowPublishDialog(false);
      setPublishingVersion(null);
      setPublishForm(EMPTY_PUBLISH_FORM);
      await fetchVersions();
    } catch (error: any) {
      console.error("Error publishing bot update:", error);
      toast.error(error?.message || "Failed to publish bot update");
    } finally {
      setPublishSubmitting(false);
    }
  };

  const toggleVersionActive = async (version: BotVersion) => {
    const nextActive = !version.is_active;
    setActionKey(`active-${version.id}`);

    try {
      const { data } = await api.patch(`/admin/bot-versions/${version.id}/active`, {
        is_active: nextActive,
      });

      if (!nextActive) {
        const stopped = Number(data?.stopped_bots || 0);
        toast.success(
          stopped > 0
            ? `Version deactivated and stopped ${stopped} running bot(s)`
            : "Version deactivated"
        );
      } else {
        toast.success("Version activated");
      }

      await fetchVersions();
    } catch (error: any) {
      console.error("Error toggling version active state:", error);
      toast.error(error?.message || "Failed to toggle version status");
    } finally {
      setActionKey(null);
    }
  };

  const rolloutVersion = async (version: BotVersion) => {
    if (!version.is_active) {
      toast.error("Please activate this version before rollout");
      return;
    }

    const confirmed = window.confirm(
      `Rollout ${version.label || version.version_tag || "this version"} to existing bots?`
    );
    if (!confirmed) return;

    setActionKey(`rollout-${version.id}`);
    try {
      const { data } = await api.post(`/admin/bot-versions/${version.id}/rollout`);
      const updatedBots = Number(data?.updated_bots || 0);
      toast.success(`Rollout complete. Updated ${updatedBots} bot(s).`);
      await fetchVersions();
    } catch (error: any) {
      console.error("Error rolling out version:", error);
      toast.error(error?.message || "Failed to rollout bot version");
    } finally {
      setActionKey(null);
    }
  };

  const deleteVersion = async (id: string) => {
    setActionKey(`delete-${id}`);
    try {
      await api.delete(`/admin/bot-versions/${id}`);
      toast.success("Version deleted");
      await fetchVersions();
    } catch (error: any) {
      console.error("Error deleting version:", error);
      toast.error(error?.message || "Failed to delete version");
    } finally {
      setActionKey(null);
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
              <DialogDescription>Create a new bot version for deployment</DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              <div>
                <Label htmlFor="add-label">Label *</Label>
                <Input
                  id="add-label"
                  placeholder="e.g., Gold Scalper v2"
                  value={addForm.label}
                  onChange={(event) => setAddForm({ ...addForm, label: event.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="add-version">Version Tag *</Label>
                <Input
                  id="add-version"
                  placeholder="e.g., v2.0.0"
                  value={addForm.version_tag}
                  onChange={(event) => setAddForm({ ...addForm, version_tag: event.target.value })}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="add-symbol">Symbol</Label>
                  <Input
                    id="add-symbol"
                    value={addForm.symbol}
                    onChange={(event) => setAddForm({ ...addForm, symbol: event.target.value })}
                  />
                </div>
                <div>
                  <Label htmlFor="add-timeframe">Timeframe</Label>
                  <Input
                    id="add-timeframe"
                    value={addForm.timeframe}
                    onChange={(event) => setAddForm({ ...addForm, timeframe: event.target.value })}
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="add-image">Docker Image ID</Label>
                <Input
                  id="add-image"
                  placeholder="registry/repo:tag"
                  value={addForm.docker_image_id}
                  onChange={(event) => setAddForm({ ...addForm, docker_image_id: event.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="add-notes">Release Notes (one line each)</Label>
                <Textarea
                  id="add-notes"
                  placeholder="Improved signal filters&#10;Lower drawdown"
                  value={addForm.release_notes}
                  onChange={(event) => setAddForm({ ...addForm, release_notes: event.target.value })}
                />
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setShowAddDialog(false)} disabled={addSubmitting}>
                Cancel
              </Button>
              <Button onClick={handleAddVersion} disabled={addSubmitting}>
                {addSubmitting ? "Adding..." : "Add Version"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Dialog
        open={showEditDialog}
        onOpenChange={(open) => {
          setShowEditDialog(open);
          if (!open) {
            setEditingVersion(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Bot Version</DialogTitle>
            <DialogDescription>Update metadata and release notes for this version</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="edit-label">Label *</Label>
              <Input
                id="edit-label"
                value={editForm.label}
                onChange={(event) => setEditForm({ ...editForm, label: event.target.value })}
              />
            </div>

            <div>
              <Label htmlFor="edit-version">Version Tag *</Label>
              <Input
                id="edit-version"
                value={editForm.version_tag}
                onChange={(event) => setEditForm({ ...editForm, version_tag: event.target.value })}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="edit-symbol">Symbol</Label>
                <Input
                  id="edit-symbol"
                  value={editForm.symbol}
                  onChange={(event) => setEditForm({ ...editForm, symbol: event.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="edit-timeframe">Timeframe</Label>
                <Input
                  id="edit-timeframe"
                  value={editForm.timeframe}
                  onChange={(event) => setEditForm({ ...editForm, timeframe: event.target.value })}
                />
              </div>
            </div>

            <div>
              <Label htmlFor="edit-image">Docker Image ID</Label>
              <Input
                id="edit-image"
                value={editForm.docker_image_id}
                onChange={(event) => setEditForm({ ...editForm, docker_image_id: event.target.value })}
              />
            </div>

            <div>
              <Label htmlFor="edit-notes">Release Notes</Label>
              <Textarea
                id="edit-notes"
                value={editForm.release_notes}
                onChange={(event) => setEditForm({ ...editForm, release_notes: event.target.value })}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditDialog(false)} disabled={editSubmitting}>
              Cancel
            </Button>
            <Button onClick={handleUpdateVersion} disabled={editSubmitting}>
              {editSubmitting ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={showPublishDialog}
        onOpenChange={(open) => {
          setShowPublishDialog(open);
          if (!open) {
            setPublishingVersion(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Publish Bot Update</DialogTitle>
            <DialogDescription>
              Push a new docker image + release notes and notify users to update from Bot Control.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="publish-image">Docker Image ID *</Label>
              <Input
                id="publish-image"
                value={publishForm.docker_image_id}
                onChange={(event) => setPublishForm({ ...publishForm, docker_image_id: event.target.value })}
                placeholder="registry/repo:tag"
              />
            </div>

            <div>
              <Label htmlFor="publish-version">Version Tag (optional)</Label>
              <Input
                id="publish-version"
                value={publishForm.version_tag}
                onChange={(event) => setPublishForm({ ...publishForm, version_tag: event.target.value })}
                placeholder="e.g., v2.1.4"
              />
            </div>

            <div>
              <Label htmlFor="publish-notes">Release Notes</Label>
              <Textarea
                id="publish-notes"
                value={publishForm.release_notes}
                onChange={(event) => setPublishForm({ ...publishForm, release_notes: event.target.value })}
                placeholder="One change per line"
              />
            </div>

            <label className="flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm">
              <input
                type="checkbox"
                checked={publishForm.notify_users}
                onChange={(event) => setPublishForm({ ...publishForm, notify_users: event.target.checked })}
              />
              Broadcast to users on older versions of this bot (web + email via Resend, if allowed in settings)
            </label>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowPublishDialog(false)}
              disabled={publishSubmitting}
            >
              Cancel
            </Button>
            <Button onClick={handlePublishUpdate} disabled={publishSubmitting}>
              {publishSubmitting ? "Publishing..." : "Publish Update"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

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
                    <Badge
                      className={
                        version.is_active
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-zinc-200 text-zinc-700"
                      }
                    >
                      {version.is_active ? "Active" : "Inactive"}
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
                    <p className="text-xs text-muted-foreground mt-2 font-mono">{version.docker_image_id}</p>
                  )}

                  {version.release_notes.length > 0 && (
                    <ul className="text-sm text-muted-foreground mt-2 list-disc list-inside">
                      {version.release_notes.map((note, index) => (
                        <li key={`${version.id}-${index}`}>{note}</li>
                      ))}
                    </ul>
                  )}
                </div>

                <div className="flex flex-wrap items-center justify-end gap-2">
                  <Button size="sm" variant="outline" onClick={() => openEditDialog(version)}>
                    <PenSquare className="w-4 h-4 mr-1" />
                    Edit
                  </Button>

                  <Button size="sm" onClick={() => openPublishDialog(version)}>
                    <BellRing className="w-4 h-4 mr-1" />
                    Update Bot
                  </Button>

                  <Button
                    size="sm"
                    variant="outline"
                    disabled={actionKey === `rollout-${version.id}` || !version.is_active}
                    onClick={() => rolloutVersion(version)}
                  >
                    <Rocket className="w-4 h-4 mr-1" />
                    {actionKey === `rollout-${version.id}` ? "Rolling out..." : "Rollout Model"}
                  </Button>

                  <Button
                    size="sm"
                    variant={version.is_active ? "secondary" : "outline"}
                    disabled={actionKey === `active-${version.id}`}
                    onClick={() => toggleVersionActive(version)}
                  >
                    <Power className="w-4 h-4 mr-1" />
                    {actionKey === `active-${version.id}`
                      ? "Saving..."
                      : version.is_active
                        ? "Set Inactive"
                        : "Set Active"}
                  </Button>

                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteVersion(version.id)}
                    disabled={actionKey === `delete-${version.id}`}
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
