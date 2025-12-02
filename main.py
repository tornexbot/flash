# quick_poster_bot.py
# ORIGINAL WORKING VERSION - Webhook deployment ready

import os, re, requests, logging, json, datetime, asyncio
from typing import Optional, Tuple, List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, constants
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from io import BytesIO

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable not set")

JUP_API_KEY = os.getenv("JUP_API_KEY", "").strip() or ""
TARGET_CHANNELS = ["@CODYWHALESCALLS"]

# ================== NETWORK DETECTION & SYMBOL LOOKUP ==================
SESSION = requests.Session()

# Network configurations with better RPC endpoints
EVM_RPCS = {
    "eth": ["https://cloudflare-eth.com", "https://rpc.ankr.com/eth"],
    "bnb": ["https://bsc-dataseed.bnbchain.org", "https://bsc-dataseed1.ninicoin.io"],
    "base": ["https://mainnet.base.org", "https://base-rpc.publicnode.com", "https://base.llamarpc.com"],
    "matic": ["https://polygon-rpc.com", "https://rpc.ankr.com/polygon"],
    "arb": ["https://arb1.arbitrum.io/rpc", "https://rpc.ankr.com/arbitrum"],
    "op": ["https://mainnet.optimism.io", "https://rpc.ankr.com/optimism"],
    "avax": ["https://api.avax.network/ext/bc/C/rpc", "https://rpc.ankr.com/avalanche"],
    "ftm": ["https://rpc.ftm.tools", "https://rpc.ankr.com/fantom"],
}

# API endpoints
DEX_TOKENS_API = "https://api.dexscreener.com/latest/dex/tokens"
SOLSCAN_PUBLIC_META = "https://public-api.solscan.io/token/meta"
JUP_SEARCH = "https://lite-api.jup.ag/ultra/v1/search"
TRON_TRCSCAN = "https://apilist.tronscanapi.com/api/token_trc20"
TON_API = "https://tonapi.io/v2/jettons/"

# Regex patterns for address detection
ETHLIKE = re.compile(r"^0x[a-fA-F0-9]{40}$")
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
TRON_BASE58CHECK = re.compile(r"^T[1-9A-HJ-NP-Za-km-z]{33}$")
TON_ADDR = re.compile(r"^(?:-?\d:)?[A-Za-z0-9_-]{48,66}$")
APT_ADDR = re.compile(r"^0x[a-fA-F0-9]{64}(::[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*)?$")
SUI_ADDR = re.compile(r"^0x[a-fA-F0-9]{40,64}(::[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*)?$")

# Chain labels (no emojis)
CHAIN_LABELS_ZH = {
    "eth": "ETH", "bnb": "BNB", "base": "BASE", "sol": "SOL",
    "matic": "POLYGON", "arb": "ARBITRUM", "op": "OPTIMISM", 
    "avax": "AVAX", "ftm": "FANTOM", "tron": "TRON", 
    "ton": "TON", "apt": "APTOS", "sui": "SUI"
}

# ================== ROBUST CHAIN DETECTION ==================

def detect_network_via_dexscreener(ca: str) -> Tuple[str, List[str]]:
    """
    Returns (best_chain, candidate_chains). Picks best by liquidity among pairs
    whose baseToken.address == ca (case-insensitive). Falls back to all pairs if needed.
    """
    try:
        url = f"{DEX_TOKENS_API}/{ca}"
        r = SESSION.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code != 200:
            return "unknown", []

        pairs = (r.json() or {}).get("pairs") or []
        if not pairs:
            return "unknown", []

        # Keep only pairs where the *base* token is the CA (most token pages use base side)
        def is_base_match(p):
            base_addr = str((p.get("baseToken") or {}).get("address") or "")
            return base_addr.lower() == ca.lower()

        base_pairs = [p for p in pairs if is_base_match(p)]
        candidates = base_pairs if base_pairs else pairs

        # Build (chain -> best pair) by liquidity
        by_chain = {}
        for p in candidates:
            chain = (p.get("chainId") or "").lower()
            liq = float(((p.get("liquidity") or {}).get("usd") or 0) or 0)
            # keep the max-liquidity pair per chain
            if chain and (chain not in by_chain or liq > by_chain[chain][0]):
                by_chain[chain] = (liq, p)

        chain_map = {
            "ethereum": "eth", "bsc": "bnb", "base": "base", "polygon": "matic",
            "arbitrum": "arb", "optimism": "op", "avalanche": "avax", "fantom": "ftm",
            "tron": "tron", "solana": "sol"
        }
        
        # Candidate chains in our internal naming
        cand_chains = []
        for chain_raw in by_chain.keys():
            internal_chain = chain_map.get(chain_raw, chain_raw)
            if internal_chain not in cand_chains:
                cand_chains.append(internal_chain)

        # Best chain = max liquidity among candidates
        if by_chain:
            best_raw = max(by_chain.items(), key=lambda kv: kv[1][0])[0]
            best = chain_map.get(best_raw, best_raw)
            return best, cand_chains

        return "unknown", cand_chains
    except Exception as e:
        logger.error(f"DexScreener detection error: {e}")
        return "unknown", []

def check_chain_via_rpc(ca: str, chain: str) -> bool:
    """Check if contract exists on specific chain via RPC using eth_getCode"""
    payload = {
        "jsonrpc": "2.0", 
        "id": 1, 
        "method": "eth_getCode",
        "params": [ca, "latest"]
    }
    
    for rpc in EVM_RPCS.get(chain, []):
        try:
            result = _post_json(rpc, payload, timeout=3)
            code = result.get("result", "0x")
            # If contract exists, code will be more than "0x"
            if code and code != "0x" and code != "0x0":
                return True
        except Exception:
            continue
    
    return False

def probe_evm_candidates(ca: str, candidates: List[str]) -> Optional[str]:
    """
    Return the first chain that clearly hosts a contract at `ca`.
    If multiple chains return bytecode, prefer the one where symbol() succeeds.
    """
    good = []
    for chain in candidates:
        if chain not in EVM_RPCS: 
            continue
        if check_chain_via_rpc(ca, chain):
            # Try to pull a symbol to strengthen the signal
            sym = evm_symbol_via_rpc(ca, chain)
            score = 1 + (1 if sym else 0)
            good.append((score, chain))
    if not good:
        return None
    # Prefer symbol() success
    good.sort(reverse=True)
    return good[0][1]

def detect_evm_network(ca: str) -> str:
    """Detect specific EVM network using multiple verification methods"""
    # Step 1: narrow via DexScreener
    best, candidates = detect_network_via_dexscreener(ca)

    # If DexScreener gave one clear answer, still *verify* via RPC
    if best != "unknown":
        if check_chain_via_rpc(ca, best):
            return best
        # If best failed RPC (rate limit / temp outage), try other candidates
        other = [c for c in candidates if c != best]
        verified = probe_evm_candidates(ca, other)
        if verified:
            return verified

    # If ambiguous / unknown, probe a short prioritized set
    prio_candidates = candidates or ["base", "arb", "op", "bnb", "matic", "avax", "ftm", "eth"]
    verified = probe_evm_candidates(ca, prio_candidates)
    return verified or "unknown"  # Never default to eth

def detect_network(ca: str) -> str:
    """Auto-detect network from contract address with robust logic"""
    
    # First, check non-EVM chains
    if 32 <= len(ca) <= 48 and all(c in BASE58 for c in ca):
        return "sol"
    if TRON_BASE58CHECK.match(ca):
        return "tron"
    if TON_ADDR.match(ca):
        return "ton"
    if APT_ADDR.match(ca):
        return "apt"
    if SUI_ADDR.match(ca):
        return "sui"
    
    # For EVM chains (0x addresses), use robust detection
    if ETHLIKE.match(ca):
        chain = detect_evm_network(ca)
        return chain if chain != "unknown" else "unknown"
    
    return "unknown"

# ================== REST OF THE CODE (unchanged) ==================

def _post_json(url: str, payload: dict, timeout: int = 5) -> dict:
    r = SESSION.post(url, json=payload, timeout=timeout, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()

def _decode_erc20_symbol(hexdata: str) -> Optional[str]:
    if not hexdata or not hexdata.startswith("0x"):
        return None
    try:
        data = bytes.fromhex(hexdata[2:])
        if len(data) == 32:
            return data.rstrip(b"\x00").decode("utf-8", "ignore") or None
        if len(data) >= 64:
            strlen = int.from_bytes(data[32:64], "big")
            raw = data[64:64+strlen]
            return raw.decode("utf-8", "ignore") or None
    except Exception:
        pass
    return None

def evm_symbol_via_rpc(contract: str, chain: str) -> Optional[str]:
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": contract, "data": "0x95d89b41"}, "latest"]
    }
    for rpc in EVM_RPCS.get(chain, []):
        try:
            result = _post_json(rpc, payload).get("result")
            sym = _decode_erc20_symbol(result)
            if sym:
                return sym.upper()
        except Exception:
            continue
    return None

def sol_symbol_via_jupiter(mint: str) -> Optional[str]:
    if not JUP_API_KEY:
        return None
    try:
        r = SESSION.get(JUP_SEARCH, params={"query": mint},
                       headers={"x-api-key": JUP_API_KEY, "User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            arr = r.json()
            if isinstance(arr, list) and arr:
                for it in arr:
                    if str(it.get("id", "")).strip() == mint:
                        sym = (it.get("symbol") or "").upper()
                        if sym:
                            return sym
                sym = (arr[0].get("symbol") or "").upper()
                return sym or None
    except Exception:
        pass
    return None

def generic_symbol_via_dexscreener(addr: str) -> Optional[str]:
    try:
        url = f"{DEX_TOKENS_API}/{addr}"
        r = SESSION.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code != 200:
            return None

        pairs = r.json().get("pairs") or []
        if not pairs:
            return None

        def is_same_base(p):
            bt = (p.get("baseToken") or {})
            return str(bt.get("address", "")).lower() == addr.lower()

        filtered = [p for p in pairs if is_same_base(p)]
        candidates = filtered if filtered else pairs

        def score(p):
            liq = ((p.get("liquidity") or {}).get("usd") or 0) or 0
            vol = (p.get("volume24h") or 0) or 0
            return (float(liq), float(vol))

        best = max(candidates, key=score)
        sym = ((best.get("baseToken") or {}).get("symbol") or "").upper()
        return sym or None
    except Exception:
        return None

def get_symbol_from_ca(ca: str, chain: str) -> Optional[str]:
    """Get symbol for contract address"""
    if chain in ["eth", "bnb", "base", "matic", "arb", "op", "avax", "ftm"]:
        return evm_symbol_via_rpc(ca, chain) or generic_symbol_via_dexscreener(ca)
    elif chain == "sol":
        return sol_symbol_via_jupiter(ca) or generic_symbol_via_dexscreener(ca)
    elif chain == "tron":
        try:
            r = SESSION.get(TRON_TRCSCAN, params={"contract": ca}, 
                           headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                data = r.json()
                arr = data.get("trc20_tokens") if isinstance(data, dict) else None
                if isinstance(arr, list) and arr:
                    sym = (arr[0].get("symbol") or "").upper()
                    return sym or None
                sym = (data.get("symbol") or "").upper() if isinstance(data, dict) else None
                return sym or None
        except Exception:
            pass
        return generic_symbol_via_dexscreener(ca)
    else:
        return generic_symbol_via_dexscreener(ca)

def get_dex_url(chain: str, ca: str) -> Optional[str]:
    """Get DexScreener URL for token"""
    try:
        url = f"{DEX_TOKENS_API}/{ca}"
        r = SESSION.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code != 200:
            return None
            
        pairs = r.json().get("pairs") or []
        if not pairs:
            return None
            
        # Find the best pair by liquidity
        def score(p):
            liq = ((p.get("liquidity") or {}).get("usd") or 0) or 0
            return float(liq)
            
        best = max(pairs, key=score)
        return best.get("url")
    except Exception:
        return None

# ================== CHINESE TEMPLATE GENERATION ==================
def generate_chinese_caption(symbol: str, chain: str, ca: str, socials: dict) -> str:
    """Generate Chinese post caption in your exact format (no emojis)"""
    
    chain_label = CHAIN_LABELS_ZH.get(chain, chain.upper())
    
    # Get DexScreener URL
    chart_url = get_dex_url(chain, ca)
    
    # Build main caption in your exact format (no emojis)
    caption = f"${symbol}\n\n"
    caption += "社区看起来坚实而有趣，让我们看看它会走向何方，月亮还是尘埃\n"
    caption += "看你的条目\n\n"
    
    if chain != "xrp":
        caption += f"{chain_label}合约: <code>{ca}</code>\n\n"
    else:
        caption += f"{chain_label}（原生资产）\n\n"
    
    caption += f"CHART: {chart_url or 'N/A'}"
    
    # Add social links if provided
    tail_lines = []
    if socials.get("tg"):  
        tg_text = socials['tg']
        if not tg_text.startswith(('http://', 'https://')):
            tg_text = f"https://t.me/{tg_text.lstrip('@')}"
        tail_lines.append(f"Tg: {tg_text}")
    if socials.get("x"):   
        x_text = socials['x']
        if not x_text.startswith(('http://', 'https://')):
            x_text = f"https://x.com/{x_text.lstrip('@')}"
        tail_lines.append(f"X: {x_text}")
    if socials.get("web"): 
        web_text = socials['web']
        if not web_text.startswith(('http://', 'https://')):
            web_text = f"https://{web_text}"
        tail_lines.append(f"Web: {web_text}")

    if tail_lines:
        caption += "\n\n" + "\n".join(tail_lines)
    
    return caption

# ================== KEYBOARDS ==================
def kb_post_options():
    """Generate post options with social link buttons (no emojis)"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Post to Channel", callback_data="post_confirm")],
        [InlineKeyboardButton("Edit Text", callback_data="edit_text")],
        [
            InlineKeyboardButton("Add TG", callback_data="set_tg"),
            InlineKeyboardButton("Add Twitter", callback_data="set_twitter"),
            InlineKeyboardButton("Add Website", callback_data="set_website")
        ]
    ])

def kb_cancel_social():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Cancel", callback_data="cancel_social")]
    ])

def kb_choose_chain(candidates: List[str]):
    """Keyboard for chain selection when detection is ambiguous"""
    rows = []
    for chain in candidates:
        label = CHAIN_LABELS_ZH.get(chain, chain.upper())
        rows.append([InlineKeyboardButton(label, callback_data=f"choose_chain:{chain}")])
    rows.append([InlineKeyboardButton("Cancel", callback_data="cancel_social")])
    return InlineKeyboardMarkup(rows)

# ================== BOT HANDLERS ==================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Simple start command"""
    welcome_text = (
        "Quick Poster Bot\n\n"
        "Just paste any contract address and I'll generate a professional post!\n\n"
        "Paste a contract address to get started!"
    )
    
    if update.message:
        await update.message.reply_text(welcome_text)
    else:
        await update.callback_query.message.edit_text(welcome_text)

async def process_ca_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process CA input - only when no social state is active"""
    # Check if we're waiting for social input first
    if (context.user_data.get("awaiting_tg") or 
        context.user_data.get("awaiting_twitter") or 
        context.user_data.get("awaiting_website") or
        context.user_data.get("awaiting_text_edit")):
        return  # Let the social/text handlers process this
    
    ca = update.message.text.strip()
    
    # Validate CA format
    if not (ETHLIKE.match(ca) or 
            (32 <= len(ca) <= 48 and all(c in BASE58 for c in ca)) or
            TRON_BASE58CHECK.match(ca) or
            TON_ADDR.match(ca) or
            APT_ADDR.match(ca) or
            SUI_ADDR.match(ca)):
        
        await update.message.reply_text(
            "Invalid contract address format. Please check and try again."
        )
        return
    
    # Show processing message
    processing_msg = await update.message.reply_text("Detecting network and fetching symbol...")
    
    try:
        # Auto-detect network with robust detection
        if ETHLIKE.match(ca):
            # For EVM addresses, use the robust detection system
            best, candidates = detect_network_via_dexscreener(ca)
            chain = detect_evm_network(ca)
            
            # If still ambiguous but we have candidates, ask user to choose
            if chain == "unknown" and candidates:
                context.user_data.update({"pending_ca": ca, "candidates": candidates})
                await processing_msg.edit_text(
                    "I found multiple possible networks for this address. Please choose the chain:",
                    reply_markup=kb_choose_chain(candidates)
                )
                return
        else:
            # For non-EVM chains, use original detection
            chain = detect_network(ca)
        
        if chain == "unknown":
            await processing_msg.edit_text("Could not detect network. Please try a different CA.")
            return
        
        # Get symbol
        symbol = get_symbol_from_ca(ca, chain)
        if not symbol:
            symbol = "TOKEN"
        
        # Store data - start with fresh socials for each token
        context.user_data.clear()
        context.user_data.update({
            "ca": ca,
            "chain": chain,
            "symbol": symbol,
            "socials": {},  # Fresh socials for each token
        })
        
        # Generate Chinese caption with fresh socials
        socials = context.user_data.get("socials", {})
        caption = generate_chinese_caption(symbol, chain, ca, socials)
        context.user_data["caption"] = caption
        
        # Show the actual post preview immediately
        await processing_msg.edit_text(
            caption,
            parse_mode=constants.ParseMode.HTML,
            reply_markup=kb_post_options()
        )
        
    except Exception as e:
        logger.error(f"Error processing CA: {e}")
        await processing_msg.edit_text(
            "Error processing contract address. Please try again."
        )

async def choose_chain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle chain selection when detection is ambiguous"""
    query = update.callback_query
    await query.answer()
    
    _, chosen = query.data.split(":", 1)
    ca = context.user_data.get("pending_ca")
    
    if not ca:
        await query.message.edit_text("Please paste the contract address again.")
        return
    
    # Get symbol for chosen chain
    symbol = get_symbol_from_ca(ca, chosen) or "TOKEN"
    
    # Store data
    context.user_data.clear()
    context.user_data.update({
        "ca": ca,
        "chain": chosen,
        "symbol": symbol,
        "socials": {},
    })
    
    # Generate caption
    socials = context.user_data.get("socials", {})
    caption = generate_chinese_caption(symbol, chosen, ca, socials)
    context.user_data["caption"] = caption
    
    # Show preview
    await query.message.edit_text(
        caption,
        parse_mode=constants.ParseMode.HTML,
        reply_markup=kb_post_options()
    )

async def post_to_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if not all(k in context.user_data for k in ["ca", "chain", "symbol", "caption"]):
        await query.message.edit_text("Missing post data. Please paste a contract address.")
        return
    
    try:
        ca = context.user_data["ca"]
        symbol = context.user_data["symbol"]
        caption = context.user_data["caption"]
        
        # Post to all channels
        success_count = 0
        for channel in TARGET_CHANNELS:
            try:
                await context.bot.send_message(
                    chat_id=channel,
                    text=caption,
                    parse_mode=constants.ParseMode.HTML,
                    disable_web_page_preview=False
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Error posting to {channel}: {e}")
        
        if success_count > 0:
            await query.message.edit_text(
                f"${symbol} posted to {success_count} channel(s)!\n\n"
                f"Contract: <code>{ca}</code>\n\n"
                "Paste another contract address to create a new post.",
                parse_mode=constants.ParseMode.HTML
            )
        else:
            await query.message.edit_text(
                "Failed to post to any channels. Check bot permissions.\n\nPaste a contract address to try again."
            )
            
    except Exception as e:
        logger.error(f"Error in post_to_channel: {e}")
        await query.message.edit_text(
            "Error posting to channel. Please try again.\n\nPaste a contract address to create a new post."
        )

async def edit_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    current_caption = context.user_data.get("caption", "")
    
    await query.message.edit_text(
        f"Edit Post Text\n\nCurrent text:\n\n{current_caption}\n\n"
        "Send your new text:",
        parse_mode=constants.ParseMode.HTML,
        reply_markup=kb_cancel_social()
    )
    
    context.user_data["awaiting_text_edit"] = True

async def process_text_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_text_edit"):
        return
    
    new_text = update.message.text.strip()
    if not new_text:
        await update.message.reply_text("Text cannot be empty. Please try again.")
        return
    
    # Update caption with custom text
    context.user_data["caption"] = new_text
    context.user_data["awaiting_text_edit"] = False
    
    # Show updated preview
    await update.message.reply_text(
        new_text,
        parse_mode=constants.ParseMode.HTML,
        reply_markup=kb_post_options()
    )

async def set_telegram(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.message.edit_text(
        "Set Telegram Link\n\n"
        "Send Telegram username, group link, or channel link:",
        reply_markup=kb_cancel_social()
    )
    
    context.user_data["awaiting_tg"] = True

async def set_twitter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.message.edit_text(
        "Set Twitter Link\n\n"
        "Send Twitter username or profile URL:",
        reply_markup=kb_cancel_social()
    )
    
    context.user_data["awaiting_twitter"] = True

async def set_website(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.message.edit_text(
        "Set Website Link\n\n"
        "Send website URL:",
        reply_markup=kb_cancel_social()
    )
    
    context.user_data["awaiting_website"] = True

async def process_social_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process social link inputs - checks states first"""
    text = update.message.text.strip()
    
    if context.user_data.get("awaiting_tg"):
        context.user_data.setdefault("socials", {})["tg"] = text
        context.user_data["awaiting_tg"] = False
        
        # Regenerate caption with new socials
        symbol = context.user_data.get("symbol")
        chain = context.user_data.get("chain")
        ca = context.user_data.get("ca")
        socials = context.user_data.get("socials", {})
        
        if all([symbol, chain, ca]):
            caption = generate_chinese_caption(symbol, chain, ca, socials)
            context.user_data["caption"] = caption
            
            # Send updated preview immediately
            await update.message.reply_text(
                caption,
                parse_mode=constants.ParseMode.HTML,
                reply_markup=kb_post_options()
            )
        
    elif context.user_data.get("awaiting_twitter"):
        context.user_data.setdefault("socials", {})["x"] = text
        context.user_data["awaiting_twitter"] = False
        
        # Regenerate caption with new socials
        symbol = context.user_data.get("symbol")
        chain = context.user_data.get("chain")
        ca = context.user_data.get("ca")
        socials = context.user_data.get("socials", {})
        
        if all([symbol, chain, ca]):
            caption = generate_chinese_caption(symbol, chain, ca, socials)
            context.user_data["caption"] = caption
            
            # Send updated preview immediately
            await update.message.reply_text(
                caption,
                parse_mode=constants.ParseMode.HTML,
                reply_markup=kb_post_options()
            )
        
    elif context.user_data.get("awaiting_website"):
        context.user_data.setdefault("socials", {})["web"] = text
        context.user_data["awaiting_website"] = False
        
        # Regenerate caption with new socials
        symbol = context.user_data.get("symbol")
        chain = context.user_data.get("chain")
        ca = context.user_data.get("ca")
        socials = context.user_data.get("socials", {})
        
        if all([symbol, chain, ca]):
            caption = generate_chinese_caption(symbol, chain, ca, socials)
            context.user_data["caption"] = caption
            
            # Send updated preview immediately
            await update.message.reply_text(
                caption,
                parse_mode=constants.ParseMode.HTML,
                reply_markup=kb_post_options()
            )
    else:
        # If no social state is active, let CA handler process it
        await process_ca_input(update, context)

async def cancel_social(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Clear any waiting states
    context.user_data["awaiting_tg"] = False
    context.user_data["awaiting_twitter"] = False
    context.user_data["awaiting_website"] = False
    context.user_data["awaiting_text_edit"] = False
    
    # Return to current preview
    symbol = context.user_data.get("symbol")
    chain = context.user_data.get("chain")
    ca = context.user_data.get("ca")
    socials = context.user_data.get("socials", {})
    
    if all([symbol, chain, ca]):
        caption = generate_chinese_caption(symbol, chain, ca, socials)
        await query.message.edit_text(
            caption,
            parse_mode=constants.ParseMode.HTML,
            reply_markup=kb_post_options()
        )

# ================== MAIN APPLICATION (WEBHOOK VERSION) ==================
def main():
    # Build application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", start_cmd))
    
    # Callback query handlers
    app.add_handler(CallbackQueryHandler(post_to_channel, pattern="^post_confirm$"))
    app.add_handler(CallbackQueryHandler(edit_text, pattern="^edit_text$"))
    app.add_handler(CallbackQueryHandler(set_telegram, pattern="^set_tg$"))
    app.add_handler(CallbackQueryHandler(set_twitter, pattern="^set_twitter$"))
    app.add_handler(CallbackQueryHandler(set_website, pattern="^set_website$"))
    app.add_handler(CallbackQueryHandler(cancel_social, pattern="^cancel_social$"))
    app.add_handler(CallbackQueryHandler(choose_chain, pattern=r"^choose_chain:"))
    
    # Message handlers - PROPER ORDER: social inputs first, then CA
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), process_social_input))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), process_text_edit))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), process_ca_input))
    
    logger.info("Quick Poster Bot starting in webhook mode...")

    # Render gives us the port in $PORT
    port = int(os.getenv("PORT", "8000"))

    # Public base URL of your Render service, e.g. "https://quick-poster-bot.onrender.com"
    webhook_base_url = os.getenv("WEBHOOK_URL", "").rstrip("/")
    if not webhook_base_url:
        raise RuntimeError("WEBHOOK_URL environment variable not set")

    # Path part for Telegram webhook (can be anything, using token is common)
    url_path = TELEGRAM_TOKEN

    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,
        webhook_url=f"{webhook_base_url}/{url_path}",
    )


if __name__ == "__main__":
    main()
