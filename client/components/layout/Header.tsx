import { Bell, Search, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import TickerTagWidget from "../dashboard/TickerTagWidget";

export function Header() {
  return (
    <div className="sticky top-0 z-30 bg-card border-b border-border shadow-sm">
      {/* Top Row: Search & Actions */}
      <header className="h-16 flex items-center justify-between px-6 gap-6">
        {/* Search */}
        <div className="flex items-center gap-4 w-96 shrink-0">
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
        <div className="flex items-center gap-3 shrink-0">
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

      {/* Bottom Row: Ticker Tags */}
      <div className="h-14 border-t border-border/50 bg-card/50 backdrop-blur-sm flex items-center overflow-hidden">
        <TickerTagWidget />
      </div>
    </div>
  );
}
