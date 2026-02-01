import { useState } from "react";
import { CreditCard, Receipt, CheckCircle, Clock, Download, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const billingHistory = [
  { id: 1, date: "Jan 8-14, 2024", profit: 845.50, fee: 169.10, status: "paid" },
  { id: 2, date: "Jan 1-7, 2024", profit: -120.30, fee: 0, status: "skipped" },
  { id: 3, date: "Dec 25-31, 2023", profit: 523.20, fee: 104.64, status: "paid" },
  { id: 4, date: "Dec 18-24, 2023", profit: 1102.80, fee: 220.56, status: "paid" },
];

const paymentMethods = [
  { id: 1, type: "visa", last4: "4242", expiry: "12/25", isDefault: true },
  { id: 2, type: "mastercard", last4: "8888", expiry: "03/26", isDefault: false },
];

export default function Subscription() {
  const [activeTab, setActiveTab] = useState<"overview" | "history" | "payment">("overview");

  const weeklyProfit = 567.80;
  const feeRate = 0.20;
  const estimatedFee = weeklyProfit > 0 ? weeklyProfit * feeRate : 0;
  const nextBillingDate = "Jan 21, 2024";

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Subscription & Billing</h1>
        <p className="text-sm text-muted-foreground">Manage your weekly profit-based subscription</p>
      </div>

      {/* Status Card */}
      <div className="glass-card p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-success/10 flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-success" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Active Subscription</h3>
              <p className="text-sm text-muted-foreground">Weekly profit-based billing</p>
            </div>
          </div>
          <span className="px-3 py-1 rounded-full bg-success/10 text-success text-sm font-medium">Active</span>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-4 rounded-lg bg-secondary/50">
            <p className="text-sm text-muted-foreground mb-1">This Week's Profit</p>
            <p className={cn("text-2xl font-bold font-mono", weeklyProfit >= 0 ? "profit-text" : "loss-text")}>
              {weeklyProfit >= 0 ? "+" : ""}${weeklyProfit.toFixed(2)}
            </p>
          </div>
          <div className="p-4 rounded-lg bg-secondary/50">
            <p className="text-sm text-muted-foreground mb-1">Estimated Fee (20%)</p>
            <p className="text-2xl font-bold font-mono">${estimatedFee.toFixed(2)}</p>
          </div>
          <div className="p-4 rounded-lg bg-secondary/50">
            <p className="text-sm text-muted-foreground mb-1">Next Billing</p>
            <p className="text-2xl font-bold font-mono">{nextBillingDate}</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        {[
          { id: "overview", label: "Overview", icon: Receipt },
          { id: "history", label: "Billing History", icon: Clock },
          { id: "payment", label: "Payment Methods", icon: CreditCard },
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

      {/* Tab Content */}
      {activeTab === "overview" && (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="glass-card p-6 animate-slide-up">
            <h3 className="text-lg font-semibold mb-4">How It Works</h3>
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="text-primary font-bold">1</span>
                </div>
                <div>
                  <p className="font-medium">Weekly Profit Calculation</p>
                  <p className="text-sm text-muted-foreground">
                    Your net profit is calculated every Sunday at midnight UTC
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="text-primary font-bold">2</span>
                </div>
                <div>
                  <p className="font-medium">Profit-Based Fee</p>
                  <p className="text-sm text-muted-foreground">
                    20% fee only applies when you make a profit. No profit = no fee
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="text-primary font-bold">3</span>
                </div>
                <div>
                  <p className="font-medium">Automatic Billing</p>
                  <p className="text-sm text-muted-foreground">
                    Fee is automatically charged on Monday. Invoice available immediately
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <h3 className="text-lg font-semibold mb-4">This Week Preview</h3>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Gross Profit</span>
                <span className="font-mono profit-text">+$892.40</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Gross Loss</span>
                <span className="font-mono loss-text">-$324.60</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Net Profit</span>
                <span className="font-mono font-bold profit-text">+$567.80</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Estimated Fee (20%)</span>
                <span className="font-mono font-bold">${estimatedFee.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "history" && (
        <div className="glass-card p-6 animate-slide-up">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left text-xs font-medium text-muted-foreground py-3 px-2">Period</th>
                  <th className="text-right text-xs font-medium text-muted-foreground py-3 px-2">Net Profit</th>
                  <th className="text-right text-xs font-medium text-muted-foreground py-3 px-2">Fee</th>
                  <th className="text-center text-xs font-medium text-muted-foreground py-3 px-2">Status</th>
                  <th className="text-right text-xs font-medium text-muted-foreground py-3 px-2">Invoice</th>
                </tr>
              </thead>
              <tbody>
                {billingHistory.map((bill) => (
                  <tr key={bill.id} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                    <td className="py-4 px-2 font-medium">{bill.date}</td>
                    <td className="py-4 px-2 text-right">
                      <span className={cn("font-mono", bill.profit >= 0 ? "profit-text" : "loss-text")}>
                        {bill.profit >= 0 ? "+" : ""}${bill.profit.toFixed(2)}
                      </span>
                    </td>
                    <td className="py-4 px-2 text-right font-mono">${bill.fee.toFixed(2)}</td>
                    <td className="py-4 px-2 text-center">
                      <span
                        className={cn(
                          "px-2 py-1 rounded-full text-xs font-medium",
                          bill.status === "paid" ? "bg-success/10 text-success" : "bg-muted text-muted-foreground"
                        )}
                      >
                        {bill.status === "paid" ? "Paid" : "Skipped"}
                      </span>
                    </td>
                    <td className="py-4 px-2 text-right">
                      {bill.status === "paid" && (
                        <Button variant="ghost" size="sm">
                          <Download className="w-4 h-4" />
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === "payment" && (
        <div className="space-y-4">
          {paymentMethods.map((method, index) => (
            <div
              key={method.id}
              className={cn(
                "glass-card p-4 flex items-center justify-between animate-slide-up",
                method.isDefault && "ring-1 ring-primary"
              )}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-8 rounded bg-secondary flex items-center justify-center">
                  <CreditCard className="w-5 h-5 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-medium capitalize">
                    {method.type} •••• {method.last4}
                  </p>
                  <p className="text-sm text-muted-foreground">Expires {method.expiry}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {method.isDefault && (
                  <span className="px-2 py-1 rounded bg-primary/10 text-primary text-xs font-medium">Default</span>
                )}
                <Button variant="ghost" size="sm">
                  Edit
                </Button>
              </div>
            </div>
          ))}

          <Button variant="outline" className="w-full">
            <CreditCard className="w-4 h-4" />
            Add Payment Method
          </Button>

          <div className="flex items-center gap-2 justify-center text-sm text-muted-foreground mt-4">
            <Shield className="w-4 h-4" />
            <span>Payments secured by Stripe</span>
          </div>
        </div>
      )}
    </div>
  );
}
