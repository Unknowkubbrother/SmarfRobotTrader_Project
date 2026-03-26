import { memo, useEffect, useRef } from "react";

interface TradingViewWidgetProps {
    symbol: string;
    interval?: string;
    theme?: "light" | "dark";
    autosize?: boolean;
}

const TRADINGVIEW_INTERVALS: Record<string, string> = {
    M1: "1",
    M5: "5",
    M15: "15",
    M30: "30",
    H1: "60",
    H4: "240",
    D1: "1D",
};

function TradingViewWidget({
    symbol,
    interval = "H1",
    theme = "light",
    autosize = true,
}: TradingViewWidgetProps) {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!container.current) return;

        const el = container.current;

        // Clear previous widget if any
        el.innerHTML = '';

        const script = document.createElement('script');
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
        script.type = 'text/javascript';
        script.async = true;

        // TradingView widget configuration
        // Ensure symbol has a valid format. If it's a pair like "XAUUSD", it usually works.
        // We can add a prefix if needed, e.g., "OANDA:".
        // For now, let's try the raw symbol first.
        const widgetSymbol = symbol.includes(":") ? symbol : `OANDA:${symbol}`;
        const widgetInterval = TRADINGVIEW_INTERVALS[String(interval || "H1").toUpperCase()] || "60";

        script.innerHTML = JSON.stringify({
            autosize,
            symbol: widgetSymbol,
            interval: widgetInterval,
            timezone: "Etc/UTC",
            theme,
            style: "1",
            locale: "en",
            enable_publishing: false,
            allow_symbol_change: true,
            calendar: false,
            hide_top_toolbar: false,
            hide_legend: false,
            save_image: false,
            support_host: "https://www.tradingview.com"
        });

        el.appendChild(script);

        return () => {
            // Cleanup is handled by clearing innerHTML on next effect run
            el.innerHTML = "";
        };
    }, [autosize, interval, symbol, theme]);

    return (
        <div className="tradingview-widget-container h-full w-full" ref={container}>
            <div className="tradingview-widget-container__widget" style={{ height: "calc(100% - 32px)", width: "100%" }}></div>
            <div className="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span className="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
        </div>
    );
}

export default memo(TradingViewWidget);
