import { Bell, Search, User } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Header() {
  return (
    <header className="h-14 border-b border-border bg-card sticky top-0 z-30 flex items-center justify-between px-6">
      {/* Search */}
      <div className="flex items-center gap-4 flex-1 max-w-lg">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search..."
            className="w-full h-10 pl-10 pr-4 rounded-full bg-secondary border-0 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all"
          />
        </div>
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-3">
        {/* Live Ticker */}
        <div className="hidden md:flex items-center gap-3 px-4 py-2 rounded-full bg-secondary text-sm">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">XAUUSD</span>
            <span className="font-mono text-success font-medium">2,031.45</span>
          </div>
          <div className="w-px h-4 bg-border" />
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">EURUSD</span>
            <span className="font-mono text-destructive font-medium">1.0872</span>
          </div>
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative rounded-full">
          <Bell className="w-5 h-5" />
          <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-destructive" />
        </Button>

        {/* Profile */}
        <Button variant="ghost" size="icon" className="rounded-full">
          <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
            <User className="w-4 h-4 text-muted-foreground" />
          </div>
        </Button>
      </div>
    </header>
  );
}
