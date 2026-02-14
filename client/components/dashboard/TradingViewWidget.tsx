import { useEffect, useRef, memo } from 'react';

interface TradingViewWidgetProps {
    symbol: string;
    theme?: 'light' | 'dark';
    autosize?: boolean;
}

function TradingViewWidget({ symbol, theme = 'dark', autosize = true }: TradingViewWidgetProps) {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!container.current) return;

        // Clear previous widget if any
        container.current.innerHTML = '';

        const script = document.createElement('script');
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
        script.type = 'text/javascript';
        script.async = true;

        // TradingView widget configuration
        // Ensure symbol has a valid format. If it's a pair like "XAUUSD", it usually works.
        // We can add a prefix if needed, e.g., "OANDA:".
        // For now, let's try the raw symbol first.
        const widgetSymbol = symbol.includes(':') ? symbol : `OANDA:${symbol}`;

        script.innerHTML = JSON.stringify({
            "autosize": autosize,
            "symbol": widgetSymbol,
            "interval": "H1",
            "timezone": "Etc/UTC",
            "theme": theme,
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "calendar": false,
            "support_host": "https://www.tradingview.com"
        });

        container.current.appendChild(script);

        return () => {
            // Cleanup is handled by clearing innerHTML on next effect run
            if (container.current) {
                container.current.innerHTML = '';
            }
        };
    }, [symbol, theme, autosize]);

    return (
        <div className="tradingview-widget-container" ref={container} style={{ height: "100%", width: "100%" }}>
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
