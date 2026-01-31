# main.py
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from retrieval import (
    upsert_image_dataset,
    upsert_text_dataset,
    hybrid_search_image_query,
)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        for k in ["img_rank", "t_rank"]:
            if r.get(k) is not None:
                ranks.append(f"{k}={r[k]}")

        extra = []
        if r.get("final_score") is not None:
            extra.append(f"final={float(r['final_score']):.4f}")
        if r.get("rerank_text_score") is not None:
            extra.append(f"rerank={float(r['rerank_text_score']):.4f}")

        print(f"{i}. {r['image']}")
        print(f"   rrf={r['rrf']:.6f}"
              + (f" | {' | '.join(ranks)}" if ranks else "")
              + (f" | {' | '.join(extra)}" if extra else ""))
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
    chart_db = upsert_image_dataset(DATASET_JSON)  # image->image
    text_db = upsert_text_dataset(DATASET_JSON)    # dataset.data text->text

    vision_llm = OllamaLLM(model="ministral-3:14b", temperature=0)

    # 2) PASS A: draft from image
    draft = vision_llm.invoke(PROMPT_DRAFT_FROM_IMAGE, images=[QUERY_IMAGE])
    print("\n📝 Draft (from image):\n", draft)

    # 3) pull domain examples from dataset (TEXT ONLY)
    try:
        ex_docs = text_db.similarity_search(draft, k=6)
    except Exception as e:
        ex_docs = []
        print("\n⚠️ text_db.similarity_search failed, fallback to empty examples:", e)

    domain_examples = "\n\n---\n\n".join(
        mask_numbers(d.page_content) for d in ex_docs if getattr(d, "page_content", None)
    )
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

    # 5) compress -> query_text (for text_db)
    query_text = build_query_text_from_auto(auto_text)
    print("\n🔎 Query text (used for text):\n", query_text)

    # 6) hybrid search (IMAGE + TEXT)
    results = hybrid_search_image_query(
        chart_db=chart_db,
        text_db=text_db,
        dataset_json=DATASET_JSON,
        query_image=QUERY_IMAGE,
        auto_text=query_text,
        k_img=15,
        k_t=15,
        final_k=5,
        w_img=0.85,
        w_t=0.15,

        # ✅ rerank
        rerank=True,
        rerank_top_m=30,
        w_rerank=0.45,
    )

    print_results("IMAGE → HYBRID (Chart + Text via auto_text)", results)

    # 7) RAG context from retrieved examples
    rag_context = build_rag_context(results, max_chars=1500)
    print("\n🔎 RAG Context (preview):")
    print(rag_context)

    # 8) FINAL ANALYSIS (ใช้ RAG Prompt Template)
    # กำหนด RAG Prompt Template
    rag_template = """
คุณคือนักเทรดผู้เชี่ยวชาญ วิเคราะห์กราฟปัจจุบันจากภาพ โดยใช้แนวคิดจากกราฟเก่าที่คล้ายกัน

บริบทจากกราฟเก่า (Pattern คล้ายกัน):
{context}

กติกา:
- เขียนเป็นย่อหน้าเดียว ไม่เกิน 3 ประโยค
- ไม่ใช้ bullet, ไม่มี Markdown, ไม่มีสัญลักษณ์พิเศษ
- ไม่พูดคำว่า แนวโน้ม โมเมนตัม โครงสร้างราคา KEYWORDS

รูปแบบ:
1) เปิดด้วย "PA อยู่ในช่วง..." หรือ "PA เป็นของฝั่ง..."
2) ระบุแนวรับและแนวต้านจากภาพปัจจุบัน พร้อมบอกว่าใครเสียเปรียบ
3) ปิดด้วยกลยุทธ์สั้น ๆ ว่าควรรอเล่นตรงไหน

คำถาม: วิเคราะห์กราฟนี้และให้คำแนะนำการเทรด
"""
    
    # สร้าง ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(rag_template)
    
    # สร้าง RAG Chain
    rag_chain = (
        {"context": lambda x: x["context"]}
        | prompt
        | vision_llm
        | StrOutputParser()
    )
    
    # เรียกใช้ RAG Chain พร้อมรูปภาพ
    # หมายเหตุ: OllamaLLM กับ vision ต้องใช้ invoke โดยตรง
    formatted_prompt = prompt.format(context=rag_context)
    final_answer = vision_llm.invoke(formatted_prompt, images=[QUERY_IMAGE])

    print("\n🧠 Final Analysis:")
    print(final_answer)


if __name__ == "__main__":
    main()


# 1 datasets1/chart7.jpg
# 2 datasets1/chart8.jpg
# 3 datasets1/chart16.png
# 4 datasets1/chart18.png
# 5 datasets1/chart21.png