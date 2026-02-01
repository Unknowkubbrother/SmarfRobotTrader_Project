import { useState } from "react";
import { User, Shield, Bell, Monitor, Globe, Key } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

const loginHistory = [
  { id: 1, date: "Jan 15, 2024 14:32", ip: "192.168.1.1", device: "Chrome on MacOS", location: "Bangkok, Thailand" },
  { id: 2, date: "Jan 14, 2024 09:15", ip: "192.168.1.1", device: "Safari on iPhone", location: "Bangkok, Thailand" },
  { id: 3, date: "Jan 13, 2024 18:45", ip: "203.150.52.1", device: "Chrome on Windows", location: "Singapore" },
];

export default function Settings() {
  const [activeTab, setActiveTab] = useState<"profile" | "security" | "notifications">("profile");

  const handleSave = () => {
    toast.success("Settings saved successfully");
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground">Manage your account and preferences</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        {[
          { id: "profile", label: "Profile", icon: User },
          { id: "security", label: "Security", icon: Shield },
          { id: "notifications", label: "Notifications", icon: Bell },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={cn(
              "flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px",
              activeTab === tab.id
                ? "text-primary border-primary"
                : "text-muted-foreground border-transparent hover:text-foreground"
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Profile Tab */}
      {activeTab === "profile" && (
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="glass-card p-6 animate-slide-up">
              <h3 className="text-lg font-semibold mb-6">Account Information</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Full Name</label>
                  <input
                    type="text"
                    defaultValue="John Trader"
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Email</label>
                  <input
                    type="email"
                    defaultValue="john@trader.com"
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Timezone</label>
                  <select className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50">
                    <option>Asia/Bangkok (UTC+7)</option>
                    <option>America/New_York (UTC-5)</option>
                    <option>Europe/London (UTC+0)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Language</label>
                  <select className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50">
                    <option>English</option>
                    <option>Thai</option>
                    <option>Japanese</option>
                  </select>
                </div>
              </div>
              <Button onClick={handleSave} className="mt-6">
                Save Changes
              </Button>
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <h3 className="text-lg font-semibold mb-6">Profile Photo</h3>
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center mb-4">
                <User className="w-10 h-10 text-primary-foreground" />
              </div>
              <Button variant="outline" size="sm">
                Upload Photo
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Security Tab */}
      {activeTab === "security" && (
        <div className="space-y-6">
          <div className="glass-card p-6 animate-slide-up">
            <div className="flex items-center gap-2 mb-6">
              <Key className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Password & Authentication</h3>
            </div>
            <div className="space-y-4 max-w-md">
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Current Password</label>
                <input
                  type="password"
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                />
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">New Password</label>
                <input
                  type="password"
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                />
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Confirm New Password</label>
                <input
                  type="password"
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                />
              </div>
              <Button onClick={handleSave}>Update Password</Button>
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <div className="flex items-center gap-2 mb-6">
              <Monitor className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Login History</h3>
            </div>
            <div className="space-y-3">
              {loginHistory.map((login) => (
                <div key={login.id} className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                  <div>
                    <p className="font-medium">{login.device}</p>
                    <p className="text-sm text-muted-foreground">
                      {login.location} • {login.ip}
                    </p>
                  </div>
                  <span className="text-sm text-muted-foreground">{login.date}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Notifications Tab */}
      {activeTab === "notifications" && (
        <div className="glass-card p-6 animate-slide-up">
          <h3 className="text-lg font-semibold mb-6">Notification Preferences</h3>
          <div className="space-y-6">
            {[
              { id: "email", label: "Email Notifications", description: "Receive updates via email" },
              { id: "profit", label: "Profit Alerts", description: "Get notified when daily profit exceeds threshold" },
              { id: "loss", label: "Loss Alerts", description: "Get notified when losses exceed threshold" },
              { id: "bot", label: "Bot Status", description: "Alerts for bot stops or errors" },
              { id: "weekly", label: "Weekly Summary", description: "Weekly performance report" },
            ].map((item) => (
              <div key={item.id} className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{item.label}</p>
                  <p className="text-sm text-muted-foreground">{item.description}</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" defaultChecked className="sr-only peer" />
                  <div className="w-11 h-6 bg-secondary rounded-full peer peer-checked:bg-primary transition-colors after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full"></div>
                </label>
              </div>
            ))}
          </div>
          <Button onClick={handleSave} className="mt-6">
            Save Preferences
          </Button>
        </div>
      )}
    </div>
  );
}
