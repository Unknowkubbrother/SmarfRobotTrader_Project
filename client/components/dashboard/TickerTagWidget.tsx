import { useEffect, memo } from 'react';

declare global {
    namespace JSX {
        interface IntrinsicElements {
            'tv-ticker-tag': any;
        }
    }
}

// Popular Forex pairs, Commodities, Indices, and Crypto
const SYMBOLS = [
    'FX:EURUSD', 'FX:GBPUSD', 'FX:USDJPY', 'FX:USDCHF',
    'FX:AUDUSD', 'FX:USDCAD', 'FX:NZDUSD',
    'OANDA:XAUUSD', 'OANDA:XAGUSD', // Gold, Silver
    'BITSTAMP:BTCUSD', 'BITSTAMP:ETHUSD',
    'FOREXCOM:SPX500', 'FOREXCOM:NAS100', 'FOREXCOM:US30'
];

function TickerTagWidget() {
    useEffect(() => {
        const scriptId = 'tv-ticker-tag-script';
        if (!document.getElementById(scriptId)) {
            const script = document.createElement('script');
            script.id = scriptId;
            script.type = 'module';
            script.src = 'https://widgets.tradingview-widget.com/w/en/tv-ticker-tag.js';
            document.head.appendChild(script);
        }
    }, []);

    useEffect(() => {
        const suppressKnownTradingViewError = (event: ErrorEvent) => {
            const message = String(event.message || "");
            const source = String(event.filename || "");

            if (
                message.includes("Element is not connected") &&
                source.includes("tradingview-widget.com")
            ) {
                event.preventDefault();
                event.stopImmediatePropagation();
            }
        };

        window.addEventListener("error", suppressKnownTradingViewError, true);
        return () => {
            window.removeEventListener("error", suppressKnownTradingViewError, true);
        };
    }, []);

    // Duplicate list for seamless scrolling
    const items = [...SYMBOLS, ...SYMBOLS];

    return (
        <div className="w-full h-full overflow-hidden flex items-center relative mask-gradient pointer-events-none">
            <style>{`
         @keyframes scroll {
           0% { transform: translateX(0); }
           100% { transform: translateX(-50%); }
         }
         .animate-scroll {
           display: flex;
           animation: scroll 60s linear infinite; /* Slow smooth scroll */
           width: max-content;
         }
         .mask-gradient {
            mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
            -webkit-mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
         }
       `}</style>
            <div className="animate-scroll">
                {items.map((symbol, i) => (
                    <div key={`${symbol}-${i}`} className="mx-4 pointer-events-none">
                        <tv-ticker-tag symbol={symbol} theme="light" large></tv-ticker-tag>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default memo(TickerTagWidget);
