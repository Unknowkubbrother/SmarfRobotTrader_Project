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

    // Duplicate list for seamless scrolling
    const items = [...SYMBOLS, ...SYMBOLS];

    return (
        <div className="w-full h-full overflow-hidden flex items-center relative group mask-gradient">
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
         .group:hover .animate-scroll {
           animation-play-state: paused;
         }
         .mask-gradient {
            mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
            -webkit-mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
         }
       `}</style>
            <div className="animate-scroll">
                {items.map((symbol, i) => (
                    <div key={`${symbol}-${i}`} className="mx-4 transform transition-transform hover:scale-105">
                        <tv-ticker-tag symbol={symbol} theme="light" large></tv-ticker-tag>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default memo(TickerTagWidget);
