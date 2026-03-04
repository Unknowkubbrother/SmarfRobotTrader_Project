from __future__ import annotations

from collections import OrderedDict

MT5_BROKER_SERVER_CATALOG: tuple[dict[str, object], ...] = (
    {
        "broker_name": "ABH Forex Ltd",
        "server_names": ("ABHForex-STP",),
    },
    {
        "broker_name": "Admirals Group AS",
        "server_names": ("AdmiralsGroup-Demo", "AdmiralsGroup-Live"),
    },
    {
        "broker_name": "Admirals SC Ltd",
        "server_names": ("AdmiralsSC-Demo", "AdmiralsSC-Live"),
    },
    {
        "broker_name": 'ООО "Альфа-Форекс"',
        "server_names": ("AlfaForexRU-Real",),
    },
    {
        "broker_name": "Ava Trade Markets Ltd.",
        "server_names": ("AvaTradeMarkets-Demo 1-MT5", "AvaTradeMarkets-Real 1-MT5"),
    },
    {
        "broker_name": "B.I.C. Markets Co., Ltd.",
        "server_names": ("BICForex-MT5RUSD", "BICMarkets-Server"),
    },
    {
        "broker_name": 'ООО "БКС-Форекс"',
        "server_names": ("BCSForex-MT5RUSD",),
    },
    {
        "broker_name": "CTForex Ltd.",
        "server_names": ("CTForex-Trade",),
    },
    {
        "broker_name": "Dragonstone Forex Trading Limited",
        "server_names": ("DragonstoneForex-Live",),
    },
    {
        "broker_name": "Exispro Ltd.",
        "server_names": ("Exispro-WForex Server",),
    },
    {
        "broker_name": "FBS Markets Inc.",
        "server_names": ("FBS-Demo", "FBS-Real"),
    },
    {
        "broker_name": "GAIN Capital - FOREX.com Canada Ltd.",
        "server_names": ("FOREX.comCA-Demo 531", "FOREX.comCA-Live 532", "FOREX.comCA-Demo"),
    },
    {
        "broker_name": "Gain Capital Group, LLC",
        "server_names": ("Forex.com-Demo 535", "Forex.com-Live 536", "Forex.com"),
    },
    {
        "broker_name": "Gain Global Markets, Inc. (FOREX.com Global CN)",
        "server_names": ("FOREX.comGlobalCN-Demo 533", "FOREX.comGlobalCN-Live 534"),
    },
    {
        "broker_name": "Gain Global Markets, Inc. (FOREX.com Global)",
        "server_names": ("FOREX.comGlobal-Demo 531", "FOREX.comGlobal-Live 532", "FOREX.comGlobalCN"),
    },
    {
        "broker_name": "StoneX Europe Ltd",
        "server_names": ("FOREX.comEurope-Demo 531", "FOREX.comEurope-Live 532", "FOREX.comEurope"),
    },
    {
        "broker_name": "Forexcitypro for Financial Consulting Limited",
        "server_names": ("Forexcitypro-Live", "Forexcitypro-Trial", "Forexcitypro"),
    },
    {
        "broker_name": "Forex Club International LLC",
        "server_names": (
            "ForexClubInternational-MT5 Market Real 2",
            "ForexClub-MT5 Demo Server",
            "ForexClub-MT5 Instant Real Server",
            "ForexClub-MT5 Real Server",
            "ForexClub",
            "ForexClubInternational",
        ),
    },
    {
        "broker_name": 'ООО "Финансовая Компания "Форекс Клуб"',
        "server_names": (
            "ForexClubBY-MT5 Demo Server",
            "ForexClubBY-MT5 Real Server",
            "ForexClubBy",
        ),
    },
    {
        "broker_name": "Forexer Limited",
        "server_names": ("FOREXer-Server", "FOREXer"),
    },
    {
        "broker_name": "Myrtle Company For Trade Brokerage Ltd",
        "server_names": ("ForexIraqMyrtle-Server", "ForexIraqMyrtle"),
    },
    {
        "broker_name": "Forexpress Limited",
        "server_names": ("Forexpress-Server", "Forexpress"),
    },
    {
        "broker_name": "FXTM",
        "server_names": (
            "ForexTimeFXTM-Demo01",
            "ForexTimeFXTM-Demo02",
            "ForexTimeFXTM-Live01",
            "ForexTimeFXTM-Live02",
            "ForexTimeFXTM",
        ),
    },
    {
        "broker_name": "ForexTime Ltd.",
        "server_names": (
            "ForexTime-Demo01",
            "ForexTime-Demo02",
            "ForexTime-Live01",
            "ForexTime-Live02",
            "ForexTime",
        ),
    },
    {
        "broker_name": "FP Markets LLC",
        "server_names": ("FPMarketsLLC-Demo", "FPMarketsLLC-Live", "FPMarketsLLC"),
    },
    {
        "broker_name": "FP Markets Limited",
        "server_names": ("FPMarketsKE-Demo", "FPMarketsKE-Live", "FPMarketsKE"),
    },
    {
        "broker_name": "FP Markets Ltd.",
        "server_names": ("FPMarkets-Demo2", "FPMarkets-Live2", "FPMarkets"),
    },
    {
        "broker_name": "Riston Capital Ltd.",
        "server_names": ("FreshForex-MT5", "FreshForex"),
    },
    {
        "broker_name": "Global Dynamic Markets Limited",
        "server_names": ("GlobalDynamicMarkets-Demo", "GlobalDynamicMarkets-Real", "GlobalDynamicMarkets"),
    },
    {
        "broker_name": "Icare Forex Limited",
        "server_names": ("IcareForex-Server", "IcareForex"),
    },
    {
        "broker_name": "IC Markets (EU) Ltd",
        "server_names": ("ICMarketsEU-Demo", "ICMarketsEU-MT5-5", "ICMarketsEU-MT5"),
    },
    {
        "broker_name": "IC Markets (KE) Limited",
        "server_names": ("ICMarketsKE-Demo", "ICMarketsKE-MT5-7", "ICMarketsKE-MT5"),
    },
    {
        "broker_name": "IC Markets Group Ltd",
        "server_names": ("ICMarketsGRP-Demo", "ICMarketsGRP-MT5", "ICMarketsGRP"),
    },
    {
        "broker_name": "IC Markets Ltd",
        "server_names": (
            "ICMarketsInternational-Demo",
            "ICMarketsInternational-MT5",
            "ICMarketsInternational-MT5-2",
            "ICMarketsInternational-MT5-4",
            "ICMarketsInternational",
        ),
    },
    {
        "broker_name": "Ikas Forex Ltd.",
        "server_names": ("IkasForex-Trade", "IkasForex"),
    },
    {
        "broker_name": "Instant Trading EU Ltd",
        "server_names": ("InstaForex-Server", "InstaForex"),
    },
    {
        "broker_name": "InterStellar Financial Group Limited",
        "server_names": ("InterStellarFinancial-Demo", "InterStellarFinancial-Server", "InterStellarFinancial"),
    },
    {
        "broker_name": "PT. Jasa Mulia Forexindo",
        "server_names": ("JasaMuliaForexindo-Demo", "JasaMuliaForexindo-Real"),
    },
    {
        "broker_name": "Liteforex (Europe) Ltd",
        "server_names": ("LiteForexEU-MT5-Demo", "LiteForexEU-MT5-Live", "LiteForexEU-MT5"),
    },
    {
        "broker_name": "MetaQuotes Ltd.",
        "server_names": ("MetaQuotes-Demo",),
    },
    {
        "broker_name": "OANDA (Canada) Corporation ULC",
        "server_names": ("OANDA_Canada-Demo-1", "OANDA_Canada-Demo", "OANDA_Canada"),
    },
    {
        "broker_name": "OANDA Asia Pacific Pte. Ltd.",
        "server_names": ("OANDA_SG-Demo-1", "OANDA_SG-Live-1", "OANDA_SG"),
    },
    {
        "broker_name": "OANDA Australia Pty Ltd",
        "server_names": ("OANDA_AU-Demo-1", "OANDA_AU-Live-1", "OANDA_AU"),
    },
    {
        "broker_name": "OANDA Corporation",
        "server_names": (
            "OANDA-Demo-1",
            "OANDA-Live-1",
            "OANDA-Prop Trader",
            "OANDA-Japan MT5 Demo",
            "OANDA-Japan MT5 Live",
            "OANDA",
        ),
    },
    {
        "broker_name": "OANDA Europe Limited",
        "server_names": ("OANDA_UK-Demo-1", "OANDA_UK-Live-1", "OANDA_UK"),
    },
    {
        "broker_name": "OANDA Global Markets Limited",
        "server_names": ("OANDA_Global-Demo-1", "OANDA_Global-Live-1", "OANDA_Global"),
    },
    {
        "broker_name": "OANDA TMS Brokers S.A.",
        "server_names": ("OANDATMS-MT5",),
    },
    {
        "broker_name": "Pepperstone Financial Services L.L.C",
        "server_names": ("PepperstoneFinancialUAE-MT5-Live01",),
    },
    {
        "broker_name": "Pepperstone Group Limited",
        "server_names": ("Pepperstone-Demo", "Pepperstone-MT5-Live01", "Pepperstone-MT5"),
    },
    {
        "broker_name": "Pepperstone Limited",
        "server_names": ("PepperstoneUK-Demo", "PepperstoneUK-Live", "PepperstoneUK"),
    },
    {
        "broker_name": "RamForex Limited",
        "server_names": ("RamForex-Server", "RamForex"),
    },
    {
        "broker_name": "Raw Forex Ltd.",
        "server_names": ("RawForex-Live", "RawForex"),
    },
    {
        "broker_name": "RoboForex Ltd",
        "server_names": ("RoboForex-ECN", "RoboForex-Pro", "RoboForex"),
    },
    {
        "broker_name": "Tickmill Europe Ltd",
        "server_names": ("TickmillEU-Demo", "TickmillEU-Live", "TickmillEU"),
    },
    {
        "broker_name": "Tickmill Ltd",
        "server_names": ("Tickmill-Demo", "Tickmill-Live", "Tickmill"),
    },
    {
        "broker_name": "Tickmill UK Ltd",
        "server_names": ("TickmillUK-Demo", "TickmillUK-Live", "TickmillUK"),
    },
    {
        "broker_name": "W2Forex Ltd.",
        "server_names": ("W2Forex-Server", "W2Forex"),
    },
    {
        "broker_name": "WhiteForex Limited",
        "server_names": ("WhiteForex-Server", "WhiteForex"),
    },
    {
        "broker_name": "xm.com",
        "server_names": ("xm.com",),
    },
    {
        "broker_name": "Z Forex Capital Market LLC",
        "server_names": ("ZForexcapitalmarket-Server", "ZForexcapitalmarket"),
    },
)


def get_mt5_broker_server_catalog() -> list[dict[str, list[str]]]:
    return [
        {
            "broker_name": str(entry["broker_name"]),
            "server_names": [str(v) for v in entry["server_names"]],
        }
        for entry in MT5_BROKER_SERVER_CATALOG
    ]


def get_all_mt5_servers() -> list[str]:
    unique: OrderedDict[str, None] = OrderedDict()
    for entry in MT5_BROKER_SERVER_CATALOG:
        for server_name in entry["server_names"]:
            name = str(server_name).strip()
            if name:
                unique[name] = None
    return list(unique.keys())


def validate_mt5_broker_server_pair(broker_name: str, server_name: str) -> tuple[bool, str | None]:
    broker = str(broker_name or "").strip()
    server = str(server_name or "").strip()

    if not broker:
        return False, "brokerName cannot be empty"
    if not server:
        return False, "serverName cannot be empty"

    catalog = {str(entry["broker_name"]): {str(v) for v in entry["server_names"]} for entry in MT5_BROKER_SERVER_CATALOG}
    if broker not in catalog:
        return False, "Unknown brokerName. Please select from supported broker list."

    allowed_servers = catalog[broker]
    if server not in allowed_servers:
        return False, "serverName is not supported for selected broker. Please select from server list."

    return True, None
