import { useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  Calendar,
  Filter,
  Loader2,
  MessageSquare,
  PencilLine,
  Plus,
  Search,
  Tag,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { useTradingJournal, type TradingJournalRow } from "@/hooks/useTradingJournal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const DEFAULT_TAGS = [
  "Breakout",
  "Reversal",
  "Scalping",
  "SwingTrade",
  "GoldTrade",
  "AIConfirmed",
  "Mistake",
  "WinnerTrade",
];

function fmtDate(value: string | null): string {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toISOString().slice(0, 10);
}

function fmtNum(value: number, digits = 2): string {
  return Number(value || 0).toFixed(digits);
}

function toCsvList(value: string[]): string {
  return (Array.isArray(value) ? value : []).join(", ");
}

function parseCsvList(raw: string): string[] {
  return String(raw || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

function normalizeTag(value: string): string {
  return String(value || "").replace(/^#/, "").trim().toLowerCase();
}

function displayTag(value: string): string {
  const text = String(value || "").replace(/^#/, "").trim();
  return text ? `#${text}` : "";
}

export default function TradingJournal() {
  const {
    rows,
    loading,
    query,
    setQuery,
    upsertJournal,
    deleteJournal,
  } = useTradingJournal();

  const [searchText, setSearchText] = useState(query);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [calendarOpen, setCalendarOpen] = useState(false);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [sideFilter, setSideFilter] = useState<"all" | "BUY" | "SELL">("all");
  const [pnlFilter, setPnlFilter] = useState<"all" | "win" | "loss">("all");
  const [screenshotFilter, setScreenshotFilter] = useState<"all" | "yes" | "no">("all");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerSearchText, setPickerSearchText] = useState("");
  const [editing, setEditing] = useState<TradingJournalRow | null>(null);
  const [rationale, setRationale] = useState("");
  const [lesson, setLesson] = useState("");
  const [editTagKeys, setEditTagKeys] = useState<string[]>([]);
  const [attachments, setAttachments] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQuery(searchText);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [searchText, setQuery]);

  const availableTags = useMemo(() => {
    const map = new Map<string, string>();
    for (const base of DEFAULT_TAGS) {
      map.set(normalizeTag(base), displayTag(base));
    }
    for (const row of rows) {
      for (const rowTag of row.tags || []) {
        const key = normalizeTag(rowTag);
        if (!key) continue;
        if (!map.has(key)) map.set(key, displayTag(rowTag));
      }
    }
    return Array.from(map.entries()).map(([key, label]) => ({ key, label }));
  }, [rows]);

  const visibleRows = useMemo(() => {
    const journalRows = rows.filter((row) => Boolean(row.journalId));
    const selected = new Set(selectedTags);
    return journalRows.filter((row) => {
      const side = String(row.type || "").toUpperCase();
      const profit = Number(row.profit || 0);
      const hasScreenshot = Array.isArray(row.attachmentUrls) && row.attachmentUrls.length > 0;

      if (sideFilter !== "all" && side !== sideFilter) return false;
      if (pnlFilter === "win" && !(profit > 0)) return false;
      if (pnlFilter === "loss" && !(profit < 0)) return false;
      if (screenshotFilter === "yes" && !hasScreenshot) return false;
      if (screenshotFilter === "no" && hasScreenshot) return false;

      if (dateFrom || dateTo) {
        const rowDate = fmtDate(row.closeTime);
        if (dateFrom && rowDate < dateFrom) return false;
        if (dateTo && rowDate > dateTo) return false;
      }

      if (selected.size === 0) return true;
      const rowTagSet = new Set((row.tags || []).map((t) => normalizeTag(t)));
      for (const t of selected) {
        if (rowTagSet.has(t)) return true;
      }
      return false;
    });
  }, [rows, selectedTags, sideFilter, pnlFilter, screenshotFilter, dateFrom, dateTo]);

  const pickerRows = useMemo(() => {
    const noJournalRows = rows.filter((r) => !r.journalId);
    const baseRows = noJournalRows;
    const q = String(pickerSearchText || "").trim().toLowerCase();
    if (!q) return baseRows;
    return baseRows.filter((r) => {
      const text = [
        String(r.ticketId),
        String(r.symbol || ""),
        String(r.type || ""),
        String(r.status || ""),
        fmtDate(r.closeTime),
      ]
        .join(" ")
        .toLowerCase();
      return text.includes(q);
    });
  }, [pickerSearchText, rows]);

  const toggleTag = (tagKey: string) => {
    setSelectedTags((prev) =>
      prev.includes(tagKey) ? prev.filter((t) => t !== tagKey) : [...prev, tagKey]
    );
  };

  const openEditor = (row: TradingJournalRow) => {
    setEditing(row);
    setRationale(String(row.tradeRationale || ""));
    setLesson(String(row.mistakeLesson || ""));
    setEditTagKeys(Array.from(new Set((row.tags || []).map((t) => normalizeTag(t)).filter(Boolean))));
    setAttachments(toCsvList(row.attachmentUrls || []));
  };

  const closeEditor = () => {
    setEditing(null);
    setRationale("");
    setLesson("");
    setEditTagKeys([]);
    setAttachments("");
    setSaving(false);
  };

  const toggleEditorTag = (tagKey: string) => {
    setEditTagKeys((prev) =>
      prev.includes(tagKey) ? prev.filter((t) => t !== tagKey) : [...prev, tagKey]
    );
  };

  const onCreateEntry = () => {
    if (rows.length === 0) {
      toast.info("No closed trades available yet");
      return;
    }
    setPickerSearchText("");
    setPickerOpen(true);
  };

  const onPickTradeForEntry = (row: TradingJournalRow) => {
    setPickerOpen(false);
    openEditor(row);
  };

  const onSave = async () => {
    if (!editing) return;
    setSaving(true);
    const labelByKey = new Map(availableTags.map((t) => [t.key, t.label]));
    const selectedTagLabels = editTagKeys.map((key) => {
      const label = labelByKey.get(key) || `#${key}`;
      return label.replace(/^#/, "");
    });
    const ok = await upsertJournal({
      ticketId: Number(editing.ticketId),
      tradeRationale: rationale,
      mistakeLesson: lesson,
      tags: selectedTagLabels,
      attachmentUrls: parseCsvList(attachments),
    });
    setSaving(false);
    if (ok) closeEditor();
  };

  const onDelete = async () => {
    if (!editing?.journalId) return;
    setSaving(true);
    const ok = await deleteJournal(editing.journalId);
    setSaving(false);
    if (ok) closeEditor();
  };

  const hasActiveFilters =
    sideFilter !== "all" ||
    pnlFilter !== "all" ||
    screenshotFilter !== "all";
  const hasActiveCalendar = Boolean(dateFrom || dateTo);

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-foreground">Trading Journal</h1>
          <p className="text-sm text-muted-foreground">Document and learn from your trades</p>
        </div>
        <Button className="gap-2" onClick={onCreateEntry}>
          <Plus className="h-4 w-4" />
          New Entry
        </Button>
      </div>

      <div className="rounded-xl border border-border bg-card p-4">
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              placeholder="Search entries..."
              className="pl-9"
            />
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setCalendarOpen(true)}
            className={cn(hasActiveCalendar ? "border-primary text-primary" : "")}
          >
            <Calendar className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setFilterOpen(true)}
            className={cn(hasActiveFilters ? "border-primary text-primary" : "")}
          >
            <Filter className="h-4 w-4" />
          </Button>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {availableTags.map((tag) => {
            const active = selectedTags.includes(tag.key);
            return (
              <button
                key={tag.key}
                type="button"
                onClick={() => toggleTag(tag.key)}
                className={cn(
                  "rounded-full border px-3 py-1 text-xs transition-colors",
                  active
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-muted text-muted-foreground hover:text-foreground"
                )}
              >
                {tag.label}
              </button>
            );
          })}
        </div>
      </div>

      {loading ? (
        <div className="rounded-xl border border-border bg-card p-10">
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading trading journal...
          </div>
        </div>
      ) : visibleRows.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-12 text-center">
          <BookOpen className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">No entries found</p>
        </div>
      ) : (
        <div className="space-y-4">
          {visibleRows.map((row) => {
            const side = String(row.type || "").toUpperCase();
            const profit = Number(row.profit || 0);
            return (
              <div key={row.ticketId} className="rounded-xl border border-border bg-card p-5">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <p className="text-lg font-semibold tracking-tight">{row.symbol || "-"}</p>
                    <Badge
                      className={cn(
                        "font-mono uppercase",
                        side === "BUY" ? "bg-emerald-100 text-emerald-700" : "",
                        side === "SELL" ? "bg-rose-100 text-rose-700" : ""
                      )}
                    >
                      {side || "-"}
                    </Badge>
                    <span className="text-sm text-muted-foreground">{fmtDate(row.closeTime)}</span>
                  </div>
                  <p
                    className={cn(
                      "font-mono text-lg font-semibold",
                      profit >= 0 ? "text-emerald-600" : "text-rose-600"
                    )}
                  >
                    {profit >= 0 ? "+" : ""}${fmtNum(profit)}
                  </p>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-lg bg-muted/60 p-4">
                    <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                      <MessageSquare className="h-4 w-4 text-primary" />
                      Trade Rationale
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {row.tradeRationale || "No rationale yet"}
                    </p>
                  </div>
                  <div className="rounded-lg bg-muted/60 p-4">
                    <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                      <Tag className="h-4 w-4 text-primary" />
                      Lessons Learned
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {row.mistakeLesson || "No lesson yet"}
                    </p>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between gap-3">
                  <div className="flex flex-wrap gap-2">
                    {(row.tags || []).length > 0 ? (
                      (row.tags || []).map((tagValue, idx) => (
                        <span
                          key={`${row.ticketId}-tag-${idx}`}
                          className="rounded-full bg-muted px-2 py-1 text-xs text-muted-foreground"
                        >
                          {displayTag(tagValue)}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-muted-foreground">No tags</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    {Array.isArray(row.attachmentUrls) && row.attachmentUrls.length > 0 ? (
                      <div className="flex max-w-[360px] items-center gap-2 overflow-x-auto pb-1">
                        {(row.attachmentUrls || [])
                          .map((value) => String(value || "").trim())
                          .filter(Boolean)
                          .map((url, idx) => (
                            <a
                              key={`${row.ticketId}-shot-${idx}`}
                              href={url}
                              target="_blank"
                              rel="noreferrer"
                              className="group relative shrink-0"
                              title={`Open screenshot ${idx + 1}`}
                            >
                              <img
                                src={url}
                                alt={`Screenshot ${idx + 1} for ticket ${row.ticketId}`}
                                className="h-12 w-20 rounded-md border border-border object-cover transition-opacity group-hover:opacity-90"
                                loading="lazy"
                              />
                              <span className="absolute bottom-1 right-1 inline-flex items-center rounded bg-black/55 px-1 py-0.5 text-[10px] text-white">
                                {idx + 1}
                              </span>
                            </a>
                          ))}
                      </div>
                    ) : null}
                    <Button variant="ghost" size="sm" onClick={() => openEditor(row)}>
                      <PencilLine className="mr-1 h-4 w-4" />
                      Edit
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <Dialog open={pickerOpen} onOpenChange={setPickerOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Select Trade For New Journal Entry</DialogTitle>
            <DialogDescription>
              Choose from trades that do not have a journal yet.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={pickerSearchText}
                onChange={(e) => setPickerSearchText(e.target.value)}
                placeholder="Search by ticket, symbol, type..."
                className="pl-9"
              />
            </div>

            <div className="max-h-[360px] space-y-2 overflow-y-auto pr-1">
              {pickerRows.length === 0 ? (
                <div className="rounded-md border border-border p-4 text-sm text-muted-foreground">
                  No new trades to add
                </div>
              ) : (
                pickerRows.map((row) => {
                  const side = String(row.type || "").toUpperCase();
                  const profit = Number(row.profit || 0);
                  return (
                    <div
                      key={`pick-${row.ticketId}`}
                      className="flex items-center justify-between rounded-md border border-border bg-card px-3 py-2"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-sm">#{row.ticketId}</span>
                        <span className="font-semibold">{row.symbol || "-"}</span>
                        <Badge
                          variant="outline"
                          className={cn(
                            "font-mono uppercase",
                            side === "BUY" ? "bg-emerald-100 text-emerald-700" : "",
                            side === "SELL" ? "bg-rose-100 text-rose-700" : ""
                          )}
                        >
                          {side || "-"}
                        </Badge>
                        <span className="text-xs text-muted-foreground">{fmtDate(row.closeTime)}</span>
                        {row.journalId ? (
                          <Badge variant="secondary">Has Journal</Badge>
                        ) : (
                          <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-100">New</Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-3">
                        <span
                          className={cn(
                            "font-mono text-sm",
                            profit >= 0 ? "text-emerald-600" : "text-rose-600"
                          )}
                        >
                          {profit >= 0 ? "+" : ""}${fmtNum(profit)}
                        </span>
                        <Button size="sm" variant="outline" onClick={() => onPickTradeForEntry(row)}>
                          {row.journalId ? "Edit" : "Add"}
                        </Button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setPickerOpen(false)}>
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={calendarOpen} onOpenChange={setCalendarOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Date Range</DialogTitle>
            <DialogDescription>Filter journal cards by close date.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1">
              <p className="text-xs text-muted-foreground">From</p>
              <Input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
            </div>
            <div className="grid gap-1">
              <p className="text-xs text-muted-foreground">To</p>
              <Input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
            <Button
              variant="outline"
              onClick={() => {
                setDateFrom("");
                setDateTo("");
              }}
            >
              Clear
            </Button>
            <Button onClick={() => setCalendarOpen(false)}>Apply</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={filterOpen} onOpenChange={setFilterOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Filters</DialogTitle>
            <DialogDescription>Filter by side, PnL result and screenshot.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Side</p>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant={sideFilter === "all" ? "default" : "outline"} onClick={() => setSideFilter("all")}>All</Button>
                <Button size="sm" variant={sideFilter === "BUY" ? "default" : "outline"} onClick={() => setSideFilter("BUY")}>BUY</Button>
                <Button size="sm" variant={sideFilter === "SELL" ? "default" : "outline"} onClick={() => setSideFilter("SELL")}>SELL</Button>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">PnL</p>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant={pnlFilter === "all" ? "default" : "outline"} onClick={() => setPnlFilter("all")}>All</Button>
                <Button size="sm" variant={pnlFilter === "win" ? "default" : "outline"} onClick={() => setPnlFilter("win")}>Wins</Button>
                <Button size="sm" variant={pnlFilter === "loss" ? "default" : "outline"} onClick={() => setPnlFilter("loss")}>Losses</Button>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Screenshot</p>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant={screenshotFilter === "all" ? "default" : "outline"} onClick={() => setScreenshotFilter("all")}>All</Button>
                <Button size="sm" variant={screenshotFilter === "yes" ? "default" : "outline"} onClick={() => setScreenshotFilter("yes")}>Has Screenshot</Button>
                <Button size="sm" variant={screenshotFilter === "no" ? "default" : "outline"} onClick={() => setScreenshotFilter("no")}>No Screenshot</Button>
              </div>
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
            <Button
              variant="outline"
              onClick={() => {
                setSideFilter("all");
                setPnlFilter("all");
                setScreenshotFilter("all");
              }}
            >
              Reset
            </Button>
            <Button onClick={() => setFilterOpen(false)}>Apply</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(editing)} onOpenChange={(open) => (!open ? closeEditor() : undefined)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Journal Entry #{editing?.ticketId}</DialogTitle>
            <DialogDescription>
              {editing?.symbol || "-"} {editing?.type || "-"} | Profit{" "}
              {editing ? `${editing.profit >= 0 ? "+" : ""}${fmtNum(editing.profit)}` : "0.00"}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-3">
            <div className="grid gap-1">
              <p className="text-xs text-muted-foreground">Trade Rationale</p>
              <Textarea
                rows={4}
                value={rationale}
                onChange={(e) => setRationale(e.target.value)}
                placeholder="Why did you take this trade?"
              />
            </div>
            <div className="grid gap-1">
              <p className="text-xs text-muted-foreground">Mistake / Lesson</p>
              <Textarea
                rows={4}
                value={lesson}
                onChange={(e) => setLesson(e.target.value)}
                placeholder="What did you learn from this trade?"
              />
            </div>
            <div className="grid gap-1">
              <p className="text-xs text-muted-foreground">Tags</p>
              <div className="flex flex-wrap gap-2 rounded-md border border-border bg-muted/30 p-2">
                {availableTags.map((tag) => {
                  const active = editTagKeys.includes(tag.key);
                  return (
                    <button
                      key={`edit-tag-${tag.key}`}
                      type="button"
                      onClick={() => toggleEditorTag(tag.key)}
                      className={cn(
                        "rounded-full border px-3 py-1 text-xs transition-colors",
                        active
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border bg-card text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {tag.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="grid gap-1">
              <p className="text-xs text-muted-foreground">Attachment URLs (comma separated)</p>
              <Input
                value={attachments}
                onChange={(e) => setAttachments(e.target.value)}
                placeholder="https://..., https://..."
              />
            </div>
          </div>

          <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
            <div>
              {editing?.journalId ? (
                <Button variant="destructive" onClick={onDelete} disabled={saving}>
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </Button>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" onClick={closeEditor} disabled={saving}>
                Cancel
              </Button>
              <Button onClick={onSave} disabled={saving}>
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Save Journal
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
