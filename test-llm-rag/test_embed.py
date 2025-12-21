from sentence_transformers import SentenceTransformer


final_answer = f"""
PA อยู่ในช่วงขายแรงฝั่งขาลงชัด แนวต้านที่ 76.00 แล้วถูกเบรกลงมา ส่วนแนวรับปัจจุบันที่ 39.95-40.00 ยังคงแข็งแต่ถ้าผ่านไปจะกลายเป็นแนวต้านใหม่ ฝั่งขายมีอำนาจเหนือขาเข้า รอ Sniper Sell โซน 39.95 หากปิดแท่งยืนยันขาลงทันที ไม่ต้องรอใคร มอบตัวถ้าผ่าน 39.95 ไม่ได้.
"""

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

final_vec = model.encode(final_answer, normalize_embeddings=True)

print(final_vec.shape)