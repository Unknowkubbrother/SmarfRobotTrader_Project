# main.py
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from retrieval import (
    upsert_image_dataset,
    upsert_text_dataset,
    upsert_xmodal_image_dataset,
    hybrid_search_image_query,
)

from langchain_ollama import OllamaLLM


# ----------------------------
# PROMPTS
# ----------------------------
PROMPT_DRAFT_FROM_IMAGE = """
สรุปพฤติกรรมราคา “ช่วงท้ายกราฟ” เพื่อใช้ค้นหากราฟทรงคล้ายกัน

ข้อกำหนด:
- ห้ามใส่ตัวเลขราคา / ระดับราคา / timeframe / ชื่อคู่เงิน
- 25–40 คำ ภาษาไทย
- ระบุ bias: BUY หรือ SELL หรือ NEUTRAL แค่ 1 คำ
- ไม่ใช้ bullet ไม่ใช้หัวข้อ
""".strip()

PROMPT_DOMAIN_REWRITE = """
คุณจะได้รับ (1) DRAFT ที่สรุปจากภาพ และ (2) DOMAIN EXAMPLES จากฐานข้อมูลเดิม (เป็นสำนวนที่ระบบใช้จริง)
งาน: rewrite DRAFT ให้ “ใช้คำ/สำนวน/จังหวะการเล่า” ใกล้เคียง DOMAIN EXAMPLES เพื่อให้ค้นหาแม่นขึ้น

กติกา:
- 40–60 คำ ย่อหน้าเดียว
- ห้ามใส่ตัวเลขราคา / ระดับราคา / timeframe / ชื่อคู่เงิน
- ระบุ bias: BUY หรือ SELL หรือ NEUTRAL แค่ 1 คำ
- ห้าม bullet / ห้ามหัวข้อ / ห้าม markdown
- ปิดท้าย 1 บรรทัด: KEYWORDS: <คำหลัก 8–12 คำ> (คำต้องเป็นคำค้น เช่น rejection, long wick, break structure, lower high, sweep, breakout, fakeout, compression, pullback, etc.)
""".strip()

AUTO_TEXT_COMPRESS_NOTE = """
หมายเหตุ: KEYWORDS สำคัญต่อการค้นหา ควรเป็นคำสั้นๆและตรงภาพ
""".strip()


# ----------------------------
# HELPERS
# ----------------------------
def mask_numbers(s: str) -> str:
    # mask number-like patterns to prevent LLM from copying prices
    return re.sub(r"\d[\d,.\s-]*", "<NUM>", s)


def build_query_text_from_auto(auto_text: str) -> str:
    """
    ทำให้ auto_text เหมาะกับ retrieval:
    - เก็บ summary สั้นๆ + บรรทัด KEYWORDS
    - ตัด markdown/bullets
    - กันยาวเกินไป
    """
    if not auto_text:
        return ""

    lines = [l.strip() for l in auto_text.splitlines() if l.strip()]
    if not lines:
        return ""

    kw_line = ""
    for l in reversed(lines):
        if l.lower().startswith("keywords:"):
            kw_line = l.replace("**", "").strip()
            break

    # summary: pick first 1-2 non-bullet lines
    summary_parts = []
    for l in lines:
        low = l.lower()
        if low.startswith("keywords:"):
            continue
        if l.startswith(("-", "*", "•")) or l.startswith("#"):
            continue
        l = l.replace("**", "").strip()
        if not l:
            continue
        summary_parts.append(l)
        if len(summary_parts) >= 2:
            break

    summary = " ".join(summary_parts).strip()
    if not summary:
        summary = lines[0].replace("**", "").strip()

    # hard truncate by words
    words = summary.split()
    if len(words) > 80:
        summary = " ".join(words[:80])

    if kw_line:
        return f"{summary}\n{kw_line}".strip()
    return summary


def print_results(title: str, results):
    print(f"\n🏁 {title}")
    for i, r in enumerate(results, 1):
        snippet = (r.get("data") or "").replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:180] + "..."

        ranks = []
        for k in ["img_rank", "t_rank", "x_rank"]:
            if r.get(k) is not None:
                ranks.append(f"{k}={r[k]}")

        print(f"{i}. {r['image']}")
        print(f"   rrf={r['rrf']:.6f}" + (f" | {' | '.join(ranks)}" if ranks else ""))
        print(f"   {snippet}\n")


def build_rag_context(results, max_chars: int = 1500) -> str:
    chunks = []
    for r in results:
        txt = (r.get("data") or "").strip()
        if txt:
            chunks.append(txt)
    ctx = "\n\n---\n\n".join(chunks)
    return ctx[:max_chars]


# ----------------------------
# MAIN
# ----------------------------
def main():
    DATASET_JSON = "dataset.json"
    QUERY_IMAGE = "datasets1/new_chart3.png"

    # 1) build/update indexes
    chart_db = upsert_image_dataset(DATASET_JSON)          # V4 image->image
    text_db = upsert_text_dataset(DATASET_JSON)            # dataset.data text->text
    xmodal_db = upsert_xmodal_image_dataset(DATASET_JSON)  # auto_text(text)->image

    vision_llm = OllamaLLM(model="ministral-3:14b", temperature=0)

    # 2) PASS A: draft from image
    draft = vision_llm.invoke(PROMPT_DRAFT_FROM_IMAGE, images=[QUERY_IMAGE])
    print("\n📝 Draft (from image):\n", draft)

    # 3) pull domain examples from dataset (TEXT ONLY)
    #    (use draft to retrieve writing style closest to your dataset)
    try:
        ex_docs = text_db.similarity_search(draft, k=6)
    except Exception as e:
        ex_docs = []
        print("\n⚠️ text_db.similarity_search failed, fallback to empty examples:", e)

    domain_examples = "\n\n---\n\n".join(mask_numbers(d.page_content) for d in ex_docs if getattr(d, "page_content", None))
    if not domain_examples:
        domain_examples = "ไม่มีตัวอย่าง (fallback): ให้ใช้สำนวนเทคนิคแบบนักเทรดไทย เน้น PA logic และคำค้นที่ชัดเจน"

    # 4) PASS B: domain rewrite (NO IMAGE)
    rewrite_prompt = f"""
DRAFT:
{draft}

DOMAIN EXAMPLES (จาก dataset เดิม):
{domain_examples}

{PROMPT_DOMAIN_REWRITE}

{AUTO_TEXT_COMPRESS_NOTE}
""".strip()

    auto_text = vision_llm.invoke(rewrite_prompt, images=[QUERY_IMAGE])
    print("\n📝 Auto-text (domain rewritten):\n", auto_text)

    # 5) compress -> query_text (for text_db/xmodal)
    query_text = build_query_text_from_auto(auto_text)
    print("\n🔎 Query text (used for text/xmodal):\n", query_text)

    # 6) hybrid search (IMAGE MODE)
    results = hybrid_search_image_query(
        chart_db=chart_db,
        text_db=text_db,
        dataset_json=DATASET_JSON,
        query_image=QUERY_IMAGE,
        auto_text=query_text,
        xmodal_image_db=xmodal_db,
        k_img=15, k_t=15, k_x=15,
        final_k=5,
        w_img=0.75, w_t=0.18, w_x=0.07,
    )

    print_results("IMAGE → HYBRID (Chart V4 + Text + XModal via auto_text)", results)

    # 7) RAG context from retrieved examples
    rag_context = build_rag_context(results, max_chars=1500)
    print("\n🔎 RAG Context (preview):")
    print(rag_context)

    # 8) FINAL ANALYSIS (image + retrieved domain context)
    final_prompt = f"""
คุณคือนักเทรดผู้เชี่ยวชาญ

วิเคราะห์กราฟปัจจุบันจากภาพ และนำแนวคิดการเข้าออกจากกราฟเก่าที่คล้ายกันมาประยุกต์ใช้ (ยึด logic ไม่ยึดตัวเลขจากกราฟเก่า)

บริบทจากกราฟเก่า (Pattern คล้ายกัน):
{rag_context}

กติกา:
- เขียนเป็นย่อหน้าเดียว
- ไม่เกิน 3 ประโยค
- ไม่ใช้ bullet
- ข้อความล้วน ไม่มี Markdown ไม่มีสัญลักษณ์พิเศษ
- ไม่พูดคำว่า แนวโน้ม โมเมนตัม โครงสร้างราคา
- ห้ามพูดคำว่า KEYWORDS

รูปแบบ:
1) เปิดด้วย "PA อยู่ในช่วง..." หรือ "PA เป็นของฝั่ง..."
2) ระบุแนวรับและแนวต้านจากภาพปัจจุบัน พร้อมบอกว่าใครเสียเปรียบ
3) ปิดด้วยกลยุทธ์สั้น ๆ ว่าควรรอเล่นตรงไหน
""".strip()

    final_answer = vision_llm.invoke(final_prompt, images=[QUERY_IMAGE])

    print("\n🧠 Final Analysis:")
    print(final_answer)


if __name__ == "__main__":
    main()
