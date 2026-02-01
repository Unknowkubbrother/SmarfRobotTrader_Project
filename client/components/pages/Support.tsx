import { useState } from "react";
import { Book, MessageCircle, Search, ChevronRight, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const faqItems = [
  {
    id: 1,
    question: "How does the AI trading bot work?",
    answer: "Our AI bot uses a combination of Vision-LLM for chart pattern recognition and Reinforcement Learning for trade execution. It analyzes market data in real-time and makes trading decisions based on pre-trained models."
  },
  {
    id: 2,
    question: "What are the risk levels?",
    answer: "Low risk uses conservative position sizing with tight stop losses. Medium balances risk and reward. High risk allows for larger positions but with potentially higher drawdowns."
  },
  {
    id: 3,
    question: "How is the weekly fee calculated?",
    answer: "The fee is 20% of your net weekly profit. If your week ends in a loss or break-even, no fee is charged. Billing occurs every Monday for the previous week."
  },
  {
    id: 4,
    question: "Can I stop the bot anytime?",
    answer: "Yes, you can stop the bot at any time from the Bot Control page. The Panic Button will immediately close all open positions and stop trading."
  },
];

const docCategories = [
  { id: 1, title: "Getting Started", articles: 5, icon: "🚀" },
  { id: 2, title: "Bot Configuration", articles: 8, icon: "⚙️" },
  { id: 3, title: "Trading Strategies", articles: 12, icon: "📈" },
  { id: 4, title: "Billing & Payments", articles: 4, icon: "💳" },
  { id: 5, title: "Troubleshooting", articles: 6, icon: "🔧" },
];

export default function Support() {
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Help & Support</h1>
        <p className="text-sm text-muted-foreground">Find answers and get help with your trading bot</p>
      </div>

      {/* Search */}
      <div className="glass-card p-8 text-center animate-slide-up">
        <h2 className="text-xl font-semibold mb-4">How can we help you?</h2>
        <div className="relative max-w-xl mx-auto">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search for help articles, FAQ, or guides..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full h-12 pl-12 pr-4 rounded-xl bg-secondary border border-border text-sm placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/20"
          />
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Documentation */}
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <div className="flex items-center gap-2 mb-6">
              <Book className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Documentation</h3>
            </div>
            <div className="grid sm:grid-cols-2 gap-4">
              {docCategories.map((cat) => (
                <button
                  key={cat.id}
                  className="flex items-center gap-4 p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors text-left group"
                >
                  <span className="text-2xl">{cat.icon}</span>
                  <div className="flex-1">
                    <p className="font-medium group-hover:text-primary transition-colors">{cat.title}</p>
                    <p className="text-sm text-muted-foreground">{cat.articles} articles</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                </button>
              ))}
            </div>
          </div>

          {/* FAQ */}
          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "150ms" }}>
            <h3 className="text-lg font-semibold mb-6">Frequently Asked Questions</h3>
            <div className="space-y-3">
              {faqItems.map((faq) => (
                <div key={faq.id} className="border border-border rounded-lg overflow-hidden">
                  <button
                    onClick={() => setExpandedFaq(expandedFaq === faq.id ? null : faq.id)}
                    className="w-full flex items-center justify-between p-4 text-left hover:bg-secondary/30 transition-colors"
                  >
                    <span className="font-medium">{faq.question}</span>
                    <ChevronRight
                      className={cn(
                        "w-4 h-4 text-muted-foreground transition-transform",
                        expandedFaq === faq.id && "rotate-90"
                      )}
                    />
                  </button>
                  {expandedFaq === faq.id && (
                    <div className="px-4 pb-4">
                      <p className="text-sm text-muted-foreground">{faq.answer}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Contact */}
        <div className="space-y-6">
          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "200ms" }}>
            <div className="flex items-center gap-2 mb-6">
              <MessageCircle className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Create a Ticket</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Can't find what you're looking for? Submit a support ticket and we'll get back to you within 24 hours.
            </p>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Subject</label>
                <input
                  type="text"
                  placeholder="Brief description of your issue"
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                />
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Category</label>
                <select className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50">
                  <option>Technical Issue</option>
                  <option>Billing Question</option>
                  <option>Feature Request</option>
                  <option>Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Description</label>
                <textarea
                  rows={4}
                  placeholder="Please describe your issue in detail..."
                  className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-sm resize-none focus:outline-none focus:border-primary/50"
                />
              </div>
              <Button className="w-full">Submit Ticket</Button>
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "250ms" }}>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <div className="space-y-2">
              {[
                { label: "Video Tutorials", url: "#" },
                { label: "Community Forum", url: "#" },
                { label: "API Documentation", url: "#" },
                { label: "Status Page", url: "#" },
              ].map((link) => (
                <a
                  key={link.label}
                  href={link.url}
                  className="flex items-center gap-2 p-2 rounded-lg hover:bg-secondary/50 transition-colors text-sm"
                >
                  <ExternalLink className="w-4 h-4 text-muted-foreground" />
                  <span>{link.label}</span>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
