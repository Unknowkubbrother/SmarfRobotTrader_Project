import { useState, useEffect, useRef } from "react";
import { User, Shield, Bell, Monitor, Globe, Key, Loader2, MessageSquare, Gamepad2, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useSettings, UserProfile, NotificationConfig } from "@/hooks/useSettings";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Settings() {
  const {
    loading,
    profile,
    activityLogs,
    fetchProfile,
    fetchActivityLogs,
    updateProfile,
    updatePassword,
    requestSecurityOtp,
    updateNotifications,
    uploadAvatar
  } = useSettings();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadAvatar(file);
    }
  };

  const [activeTab, setActiveTab] = useState<"profile" | "security" | "notifications">("profile");

  // Local state for forms
  const [profileForm, setProfileForm] = useState({
    username: "",
    email: "",
    recoveryEmail: ""
  });

  const [passwordForm, setPasswordForm] = useState({
    newPassword: "",
    confirmPassword: ""
  });

  const [showOtpInput, setShowOtpInput] = useState(false);
  const [otp, setOtp] = useState("");

  const [tokenForm, setTokenForm] = useState({
    lineNotifyToken: "",
    discordWebhookUrl: ""
  });

  const [avatarUrlInput, setAvatarUrlInput] = useState("");

  const getAvatarSrc = (url: string | null | undefined) => {
    if (!url) return "";
    if (url.startsWith("http") || url.startsWith("data:")) return url;
    if (url.startsWith("/static")) return `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${url}`;
    return url;
  };

  const [thresholds, setThresholds] = useState({
    alertProfitTarget: "100",
    alertLossLimit: "-50",
    alertMarginLevelThreshold: "50"
  });

  // Load data on mount
  useEffect(() => {
    fetchProfile();
    fetchActivityLogs();
  }, [fetchProfile, fetchActivityLogs]);

  // Update form state when profile loads
  useEffect(() => {
    if (profile) {
      setProfileForm({
        username: profile.username,
        email: profile.email,
        recoveryEmail: profile.recoveryEmail || ""
      });

      // Only set input if it's an external URL
      const currentAvatar = profile.avatarUrl || "";
      setAvatarUrlInput((currentAvatar.startsWith('http') || currentAvatar.startsWith('data:')) ? currentAvatar : "");

      if (profile.notificationConfig) {
        setTokenForm({
          lineNotifyToken: profile.notificationConfig.lineNotifyToken || "",
          discordWebhookUrl: profile.notificationConfig.discordWebhookUrl || ""
        });
        setThresholds(prev => ({
          alertProfitTarget: profile.notificationConfig.alertProfitTarget?.toString() ?? prev.alertProfitTarget,
          alertLossLimit: profile.notificationConfig.alertLossLimit?.toString() ?? prev.alertLossLimit,
          alertMarginLevelThreshold: profile.notificationConfig.alertMarginLevelThreshold?.toString() ?? prev.alertMarginLevelThreshold
        }));
      }
    }
  }, [profile]);

  const handleAvatarUrlSave = async () => {
    await updateProfile({
      avatarUrl: avatarUrlInput || null
    });
  };

  const handleProfileSave = async () => {
    await updateProfile({
      username: profileForm.username,
      email: profileForm.email,
      recoveryEmail: profileForm.recoveryEmail
    });
  };

  const handlePasswordSave = async () => {
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      toast.error("New passwords do not match");
      return;
    }
    if (passwordForm.newPassword.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }

    const res = await requestSecurityOtp();
    if (res) {
      setShowOtpInput(true);
    }
  };

  const handleOtpSubmit = async () => {
    if (!otp || otp.length < 6) {
      toast.error("Please enter a valid OTP");
      return;
    }
    const success = await updatePassword(passwordForm.newPassword, otp);
    if (success) {
      setPasswordForm({ newPassword: "", confirmPassword: "" });
      setOtp("");
      setShowOtpInput(false);
    }
  };

  const handleTokenSave = async () => {
    await updateNotifications({
      lineNotifyToken: tokenForm.lineNotifyToken || null,
      discordWebhookUrl: tokenForm.discordWebhookUrl || null
    });
  };

  const handleThresholdChange = (key: string, value: string) => {
    setThresholds(prev => ({ ...prev, [key]: value }));
  };

  const saveThreshold = async (key: keyof NotificationConfig) => {
    const val = parseFloat(thresholds[key as keyof typeof thresholds]);
    if (isNaN(val)) return;
    await updateNotifications({ [key]: val });
  };

  const toggleThreshold = async (key: keyof NotificationConfig) => {
    if (!profile?.notificationConfig) return;
    const currentVal = profile.notificationConfig[key];
    if (currentVal !== null && currentVal !== undefined) {
      await updateNotifications({ [key]: null });
    } else {
      const val = parseFloat(thresholds[key as keyof typeof thresholds]);
      await updateNotifications({ [key]: isNaN(val) ? 0 : val });
    }
  };

  const toggleNotification = async (key: keyof NotificationConfig) => {
    if (!profile?.notificationConfig) return;

    const currentConfig = profile.notificationConfig;
    const isEnabled = !!currentConfig[key];

    // Toggle logic:
    // For booleans, just negate.
    // For numbers (thresholds), if null -> set default, if exists -> set null.

    let newValue: any;

    if (typeof currentConfig[key] === 'boolean') {
      newValue = !currentConfig[key];
    } else {
      // It's a number/decimal field
      if (isEnabled) {
        newValue = null; // Disable
      } else {
        // Enable with defaults
        if (key === 'alertProfitTarget') newValue = 100;
        else if (key === 'alertLossLimit') newValue = -50;
        else if (key === 'alertMarginLevelThreshold') newValue = 50;
        else newValue = 0;
      }
    }

    await updateNotifications({ [key]: newValue });
  };

  if (loading && !profile) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

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
                  <label className="block text-sm text-muted-foreground mb-2">Username</label>
                  <input
                    type="text"
                    value={profileForm.username}
                    onChange={(e) => setProfileForm({ ...profileForm, username: e.target.value })}
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Email</label>
                  <input
                    type="email"
                    value={profileForm.email}
                    onChange={(e) => setProfileForm({ ...profileForm, email: e.target.value })}
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Recovery Email</label>
                  <input
                    type="email"
                    value={profileForm.recoveryEmail}
                    onChange={(e) => setProfileForm({ ...profileForm, recoveryEmail: e.target.value })}
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Language</label>
                  <select disabled className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50 opacity-50 cursor-not-allowed">
                    <option>English</option>
                  </select>
                </div>
              </div>
              <Button onClick={handleProfileSave} className="mt-6">
                {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                Save Changes
              </Button>
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <h3 className="text-lg font-semibold mb-6">Profile Photo</h3>

            <Tabs defaultValue={profile?.avatarUrl?.startsWith('http') ? 'url' : 'upload'} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="upload">Upload Photo</TabsTrigger>
                <TabsTrigger value="url">Image URL</TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="flex flex-col items-center mt-0">
                <div className="relative group cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                  <Avatar className="w-32 h-32 border-4 border-background shadow-xl">
                    <AvatarImage src={getAvatarSrc(profile?.avatarUrl)} className="object-cover" />
                    <AvatarFallback className="text-4xl bg-primary/20 text-primary">
                      {profile?.username?.[0]?.toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  <div className="absolute inset-0 bg-black/50 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <Camera className="w-10 h-10 text-white" />
                  </div>
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                  <div className="absolute bottom-1 right-1 w-8 h-8 bg-green-500 rounded-full border-4 border-background"></div>
                </div>
                <p className="text-sm text-muted-foreground mt-4">
                  Click the avatar to upload a new photo
                </p>
              </TabsContent>

              <TabsContent value="url" className="flex flex-col items-center mt-0">
                <Avatar className="w-32 h-32 border-4 border-background shadow-xl mb-6">
                  <AvatarImage src={getAvatarSrc(profile?.avatarUrl)} className="object-cover" />
                  <AvatarFallback className="text-4xl bg-primary/20 text-primary">
                    {profile?.username?.[0]?.toUpperCase()}
                  </AvatarFallback>
                </Avatar>

                <div className="w-full max-w-sm flex items-center gap-2">
                  <input
                    type="text"
                    value={avatarUrlInput}
                    onChange={(e) => setAvatarUrlInput(e.target.value)}
                    placeholder="Paste image URL (https://...)"
                    className="flex-1 h-9 px-3 rounded-md border border-input bg-transparent text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  />
                  <Button size="sm" onClick={handleAvatarUrlSave} disabled={loading} variant="default">
                    Save
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      )}

      {/* Security Tab */}
      {activeTab === "security" && (
        <div className="space-y-6">
          <div className="glass-card p-6 animate-slide-up">
            <div className="flex items-center gap-2 mb-6">
              <Key className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">
                {profile?.hasPassword ? "Change Password" : "Set Password"}
              </h3>
            </div>

            {!showOtpInput ? (
              <div className="space-y-4 max-w-md">
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">New Password</label>
                  <input
                    type="password"
                    value={passwordForm.newPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                    placeholder="Min 6 characters"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">Confirm New Password</label>
                  <input
                    type="password"
                    value={passwordForm.confirmPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                    placeholder="Re-enter password"
                  />
                </div>
                <Button onClick={handlePasswordSave} disabled={loading}>
                  {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                  {profile?.hasPassword ? "Update Password" : "Set Password"}
                </Button>
              </div>
            ) : (
              <div className="space-y-4 max-w-md animate-fade-in">
                <div className="bg-primary/10 p-3 rounded-lg border border-primary/20 text-sm text-primary mb-4">
                  Please enter the OTP sent to your recovery email to confirm this change.
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">One-Time Password (OTP)</label>
                  <input
                    type="text"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value)}
                    className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50 text-center tracking-widest font-mono"
                    placeholder="000000"
                    maxLength={6}
                  />
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleOtpSubmit} disabled={loading} className="flex-1">
                    {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                    Confirm & Save
                  </Button>
                  <Button variant="outline" onClick={() => setShowOtpInput(false)} disabled={loading}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <div className="flex items-center gap-2 mb-6">
              <Monitor className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Login History</h3>
            </div>
            <div className="space-y-3">
              {activityLogs.length > 0 ? (
                activityLogs.map((login) => (
                  <div key={login.id} className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                    <div>
                      <p className="font-medium">{login.device || "Unknown Device"}</p>
                      <p className="text-sm text-muted-foreground">
                        {login.ip || "Unknown IP"} • {login.topic || "Login"}
                      </p>
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {new Date(login.date).toLocaleString()}
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-muted-foreground">No recent login history found.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Notifications Tab */}
      {activeTab === "notifications" && profile?.notificationConfig && (
        <div className="space-y-6">
          <div className="glass-card p-6 animate-slide-up">
            <h3 className="text-lg font-semibold mb-6">Notification Preferences</h3>
            <div className="space-y-6">
              {/* General Toggles */}
              {[
                {
                  id: "emailNotificationEnable" as keyof NotificationConfig,
                  label: "Email Notifications",
                  description: "Receive updates via email",
                  checked: profile.notificationConfig.emailNotificationEnable
                },
                {
                  id: "enableWeeklySummary" as keyof NotificationConfig,
                  label: "Weekly Summary",
                  description: "Weekly performance report",
                  checked: profile.notificationConfig.enableWeeklySummary
                },
                {
                  id: "enableMonthlySummary" as keyof NotificationConfig,
                  label: "Monthly Summary",
                  description: "Monthly performance report",
                  checked: profile.notificationConfig.enableMonthlySummary
                },
              ].map((item) => (
                <div key={item.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{item.label}</p>
                    <p className="text-sm text-muted-foreground">{item.description}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={item.checked as boolean}
                      onChange={() => toggleNotification(item.id)}
                      className="sr-only peer"
                      disabled={loading}
                    />
                    <div className="w-11 h-6 bg-secondary rounded-full peer peer-checked:bg-primary transition-colors after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full"></div>
                  </label>
                </div>
              ))}

              <div className="border-t border-border my-4 pt-4"></div>

              {/* Threshold Toggles */}
              {[
                { id: "alertProfitTarget", label: "Profit Alerts", desc: "Get notified when daily profit exceeds threshold", unit: "USD" },
                { id: "alertLossLimit", label: "Loss Alerts", desc: "Get notified when losses exceed threshold", unit: "USD" },
                { id: "alertMarginLevelThreshold", label: "Margin Level Alerts", desc: "Get notified when margin level exceeds threshold", unit: "%" },
              ].map(item => {
                const configKey = item.id as keyof NotificationConfig;
                const isEnabled = profile.notificationConfig[configKey] !== null;
                return (
                  <div key={item.id} className="flex items-center justify-between">
                    <div className="flex-1">
                      <p className="font-medium">{item.label}</p>
                      <p className="text-sm text-muted-foreground">{item.desc}</p>
                    </div>

                    <div className="flex items-center gap-4">
                      {isEnabled && (
                        <div className="relative">
                          <input
                            type="number"
                            value={thresholds[item.id as keyof typeof thresholds]}
                            onChange={(e) => handleThresholdChange(item.id, e.target.value)}
                            onBlur={() => saveThreshold(configKey)}
                            className="w-24 h-8 px-2 rounded bg-secondary border border-border text-sm text-right pr-8 focus:outline-none focus:border-primary/50"
                          />
                          <span className="absolute right-2 top-1.5 text-xs text-muted-foreground">{item.unit}</span>
                        </div>
                      )}

                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={isEnabled}
                          onChange={() => toggleThreshold(configKey)}
                          className="sr-only peer"
                          disabled={loading}
                        />
                        <div className="w-11 h-6 bg-secondary rounded-full peer peer-checked:bg-primary transition-colors after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full"></div>
                      </label>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <h3 className="text-lg font-semibold mb-6">Notification Token Preferences</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="w-4 h-4 text-[#00B900]" />
                  <label className="text-sm font-medium">Line Notify Token</label>
                </div>
                <input
                  type="text"
                  value={tokenForm.lineNotifyToken}
                  onChange={(e) => setTokenForm({ ...tokenForm, lineNotifyToken: e.target.value })}
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  placeholder="Enter Line Notify Token"
                />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Gamepad2 className="w-4 h-4 text-[#5865F2]" />
                  <label className="text-sm font-medium">Discord Webhook</label>
                </div>
                <input
                  type="text"
                  value={tokenForm.discordWebhookUrl}
                  onChange={(e) => setTokenForm({ ...tokenForm, discordWebhookUrl: e.target.value })}
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  placeholder="Enter Discord Webhook URL"
                />
              </div>
            </div>
            <Button onClick={handleTokenSave} className="mt-6">
              {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
              Save Preferences
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
