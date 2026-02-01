import { useState } from "react";
import { Plus, Search, Filter, Tag, Calendar, MessageSquare, Image, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface JournalEntry {
  id: number;
  date: string;
  symbol: string;
  type: "BUY" | "SELL";
  profit: number;
  rationale: string;
  lessons: string;
  tags: string[];
  hasScreenshot: boolean;
}

const sampleEntries: JournalEntry[] = [
  {
    id: 1,
    date: "2024-01-15",
    symbol: "XAUUSD",
    type: "BUY",
    profit: 245.50,
    rationale: "Strong support at 2030 level with bullish engulfing pattern on H4. AI confirmed momentum shift.",
    lessons: "Waited for confirmation before entry - patience paid off. Consider scaling in next time.",
    tags: ["#Breakout", "#GoldTrade", "#AIConfirmed"],
    hasScreenshot: true,
  },
  {
    id: 2,
    date: "2024-01-14",
    symbol: "EURUSD",
    type: "SELL",
    profit: -82.30,
    rationale: "Bearish divergence on RSI, expected pullback from resistance.",
    lessons: "Entered too early before confirmation. Should have waited for candle close.",
    tags: ["#Reversal", "#Mistake"],
    hasScreenshot: false,
  },
  {
    id: 3,
    date: "2024-01-13",
    symbol: "GBPJPY",
    type: "BUY",
    profit: 312.00,
    rationale: "Clean breakout above consolidation range. High volume confirmation.",
    lessons: "Perfect execution. This is the pattern to look for more often.",
    tags: ["#Breakout", "#Scalping", "#WinnerTrade"],
    hasScreenshot: true,
  },
];

const allTags = ["#Breakout", "#Reversal", "#Scalping", "#SwingTrade", "#GoldTrade", "#AIConfirmed", "#Mistake", "#WinnerTrade"];

export default function TradingJournal() {
  const [entries, setEntries] = useState(sampleEntries);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [showNewEntry, setShowNewEntry] = useState(false);

  const filteredEntries = entries.filter((entry) => {
    const matchesSearch =
      entry.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.rationale.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.lessons.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesTags = selectedTags.length === 0 || selectedTags.some((tag) => entry.tags.includes(tag));

    return matchesSearch && matchesTags;
  });

  const toggleTag = (tag: string) => {
    setSelectedTags(
      selectedTags.includes(tag) ? selectedTags.filter((t) => t !== tag) : [...selectedTags, tag]
    );
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Trading Journal</h1>
          <p className="text-sm text-muted-foreground">Document and learn from your trades</p>
        </div>
        <Button onClick={() => setShowNewEntry(true)}>
          <Plus className="w-4 h-4" />
          New Entry
        </Button>
      </div>

      {/* Search & Filters */}
      <div className="glass-card p-4 animate-slide-up">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search entries..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full h-10 pl-10 pr-4 rounded-lg bg-secondary border border-border text-sm placeholder:text-muted-foreground focus:outline-none focus:border-primary/50"
            />
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="icon">
              <Calendar className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="icon">
              <Filter className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mt-4">
          {allTags.map((tag) => (
            <button
              key={tag}
              onClick={() => toggleTag(tag)}
              className={cn(
                "px-3 py-1 rounded-full text-xs font-medium transition-all",
                selectedTags.includes(tag)
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-muted-foreground hover:text-foreground"
              )}
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      {/* Journal Entries */}
      <div className="space-y-4">
        {filteredEntries.map((entry, index) => (
          <div
            key={entry.id}
            className="glass-card p-6 glow-border animate-slide-up"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-lg">{entry.symbol}</span>
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium",
                      entry.type === "BUY" ? "bg-success/10 text-success" : "bg-loss/10 text-loss"
                    )}
                  >
                    {entry.type === "BUY" ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                    {entry.type}
                  </span>
                </div>
                <span className="text-sm text-muted-foreground">{entry.date}</span>
              </div>
              <span
                className={cn(
                  "text-xl font-bold font-mono",
                  entry.profit >= 0 ? "profit-text" : "loss-text"
                )}
              >
                {entry.profit >= 0 ? "+" : ""}${entry.profit.toFixed(2)}
              </span>
            </div>

            <div className="grid md:grid-cols-2 gap-4 mb-4">
              <div className="p-4 rounded-lg bg-secondary/30">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Trade Rationale</span>
                </div>
                <p className="text-sm text-muted-foreground">{entry.rationale}</p>
              </div>
              <div className="p-4 rounded-lg bg-secondary/30">
                <div className="flex items-center gap-2 mb-2">
                  <Tag className="w-4 h-4 text-accent" />
                  <span className="text-sm font-medium">Lessons Learned</span>
                </div>
                <p className="text-sm text-muted-foreground">{entry.lessons}</p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex flex-wrap gap-2">
                {entry.tags.map((tag) => (
                  <span key={tag} className="px-2 py-0.5 rounded-full bg-secondary text-xs text-muted-foreground">
                    {tag}
                  </span>
                ))}
              </div>
              <div className="flex items-center gap-2">
                {entry.hasScreenshot && (
                  <span className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Image className="w-3 h-3" />
                    Screenshot
                  </span>
                )}
                <Button variant="ghost" size="sm">
                  Edit
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredEntries.length === 0 && (
        <div className="glass-card p-12 text-center">
          <MessageSquare className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
          <h3 className="text-lg font-medium mb-2">No entries found</h3>
          <p className="text-sm text-muted-foreground mb-4">
            {searchQuery || selectedTags.length > 0
              ? "Try adjusting your search or filters"
              : "Start documenting your trades to improve your performance"}
          </p>
          <Button onClick={() => setShowNewEntry(true)}>
            <Plus className="w-4 h-4" />
            Create First Entry
          </Button>
        </div>
      )}
    </div>
  );
}
