import os
from dotenv import load_dotenv
import re

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from retrieval import (
    upsert_image_dataset,
    upsert_text_dataset,
    hybrid_search_image_query,
)

# from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import base64

# ----------------------------
# PROMPTS
# ----------------------------
PROMPT_DRAFT_FROM_IMAGE = """
สรุปพฤติกรรมราคา "ช่วงท้ายกราฟ" เพื่อใช้ค้นหากราฟทรงคล้ายกัน

ข้อกำหนด:
- ห้ามใส่ตัวเลขราคา / ระดับราคา / timeframe / ชื่อคู่เงิน
- 25–40 คำ ภาษาไทย
- ระบุ bias: BUY หรือ SELL หรือ NEUTRAL แค่ 1 คำ ต่อท้ายประโยคเดียวกัน
- ไม่ใช้ bullet ไม่ใช้หัวข้อ
- ห้ามเว้นบรรทัด ให้เขียนเป็นย่อหน้าเดียวจบ
""".strip()

PROMPT_DOMAIN_REWRITE = """
คุณจะได้รับ (1) DRAFT ที่สรุปจากภาพ และ (2) DOMAIN EXAMPLES จากฐานข้อมูลเดิม (เป็นสำนวนที่ระบบใช้จริง)
งาน: rewrite DRAFT ให้ "ใช้คำ/สำนวน/จังหวะการเล่า" ใกล้เคียง DOMAIN EXAMPLES เพื่อให้ค้นหาแม่นขึ้น

กติกา:
- 40–60 คำ ย่อหน้าเดียว ห้ามเว้นบรรทัด
- ห้ามใส่ตัวเลขราคา / ระดับราคา / timeframe / ชื่อคู่เงิน
- ระบุ bias: BUY หรือ SELL หรือ NEUTRAL ต่อท้ายข้อความ
- ห้าม bullet / ห้ามหัวข้อ / ห้าม markdown
- ปิดท้ายด้วย KEYWORDS: <คำหลัก 8–12 คำ> (รวมในบรรทัดเดียวกัน)
""".strip()

AUTO_TEXT_COMPRESS_NOTE = """
หมายเหตุ: KEYWORDS สำคัญต่อการค้นหา ควรเป็นคำสั้นๆและตรงภาพ
""".strip()


# ----------------------------
# HELPERS
# ----------------------------
def mask_numbers(s: str) -> str:
    return re.sub(r"\d[\d,.\s-]*", "<NUM>", s)


def strip_markdown(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"_(.+?)_", r"\1", s)
    s = re.sub(r"#+\s*", "", s)
    s = re.sub(r"^[\-\*•]\s*", "", s, flags=re.MULTILINE)
    return s.strip()


def build_query_text_from_auto(auto_text: str) -> str:
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

    words = summary.split()
    if len(words) > 80:
        summary = " ".join(words[:80])

    if kw_line:
        return f"{summary} {kw_line}".strip()
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

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


# ----------------------------
# MAIN
# ----------------------------
def main():
    DATASET_JSON = "dataset.json"
    QUERY_IMAGE = "datasets1/NVDA.png"

    chart_db = upsert_image_dataset(DATASET_JSON)
    text_db = upsert_text_dataset(DATASET_JSON)

    # vision_llm = OllamaLLM(model="ministral-3:14b", temperature=0)
    vision_llm = init_chat_model(
        model="mistralai/ministral-14b-2512",
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    base64_image = encode_image(QUERY_IMAGE)
    
    # 1. Draft from image
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": PROMPT_DRAFT_FROM_IMAGE},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
    ]
    draft = vision_llm.invoke(messages)
    draft_clean = strip_markdown(draft.content)
    print("\n📝 Draft (from image):\n", draft_clean)

    ex_docs = text_db.similarity_search(draft_clean, k=6)

    domain_examples = "\n\n---\n\n".join(
        mask_numbers(d.page_content) for d in ex_docs if getattr(d, "page_content", None)
    )
    
    if not domain_examples:
        domain_examples = "ไม่มีตัวอย่าง (fallback): ให้ใช้สำนวนเทคนิคแบบนักเทรดไทย เน้น PA logic และคำค้นที่ชัดเจน"

    rewrite_prompt = f"""
DRAFT:
{draft.content}

DOMAIN EXAMPLES (จาก dataset เดิม):
{domain_examples}

{PROMPT_DOMAIN_REWRITE}

{AUTO_TEXT_COMPRESS_NOTE}
""".strip()

    # 2. Rewrite prompt with image context
    print(f"Rewrite Prompt Length: {len(rewrite_prompt)}")
    
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": rewrite_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
    ]
    auto_text = vision_llm.invoke(messages)
    auto_text = strip_markdown(auto_text.content)
    print("\n📝 Auto-text (domain rewritten):\n", auto_text)

    query_text = build_query_text_from_auto(auto_text)
    print("\n🔎 Query text (used for text):\n", query_text)

    results = hybrid_search_image_query(
        chart_db=chart_db,
        text_db=text_db,
        dataset_json=DATASET_JSON,
        query_image=QUERY_IMAGE,
        auto_text=query_text,
        k_img=10,
        k_t=10,
        final_k=5,
        w_img=0.85,
        w_t=0.15,
        rerank=True,
        rerank_top_m=20,
        w_rerank=0.45,
    )

    print_results("IMAGE → HYBRID (Chart + Text via auto_text)", results)

    rag_context = build_rag_context(results, max_chars=1500)
    print("\n🔎 RAG Context (preview):")
    print(rag_context)

    rag_template = """
คุณคือนักเทรดผู้เชี่ยวชาญ วิเคราะห์กราฟปัจจุบันจากภาพ โดยใช้แนวคิดจากกราฟเก่าที่คล้ายกัน

บริบทจากกราฟเก่า (Pattern คล้ายกัน):
{context}

กติกา:
- เขียนเป็นย่อหน้าเดียว ห้ามเว้นบรรทัด ไม่เกิน 3 ประโยค
- ไม่ใช้ bullet, ไม่มี Markdown, ไม่มีสัญลักษณ์พิเศษ
- ไม่พูดคำว่า แนวโน้ม โมเมนตัม โครงสร้างราคา KEYWORDS
- เปิดด้วย "PA อยู่ในช่วง..." หรือ "PA เป็นของฝั่ง..."
- ระบุแนวรับและแนวต้านจากภาพปัจจุบัน พร้อมบอกว่าใครเสียเปรียบ
- ปิดด้วยกลยุทธ์สั้นๆ ว่าควรรอเล่นตรงไหน

คำถาม: วิเคราะห์กราฟนี้และให้คำแนะนำการเทรด
"""
    
    # prompt = ChatPromptTemplate.from_template(rag_template)
    
    # rag_chain = (
    #     {"context": lambda x: x["context"]}
    #     | prompt
    #     | vision_llm
    #     | StrOutputParser()
    # )
    
    formatted_prompt = rag_template.format(context=rag_context)
    
    # 3. Final analysis with RAG context and image
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": formatted_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
    ]
    final_answer = vision_llm.invoke(messages)
    final_answer = strip_markdown(final_answer.content)

    print("\n🧠 Final Analysis:")
    print(final_answer)


if __name__ == "__main__":
    main()