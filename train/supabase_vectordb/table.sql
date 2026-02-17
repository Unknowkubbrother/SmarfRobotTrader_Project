-- 0) ลบของเก่า
drop function if exists public.match_documents(extensions.vector, double precision, integer);
drop table if exists public.documents cascade;

-- 1) extension
create extension if not exists vector with schema extensions;

-- 2) สร้างตารางใหม่
create table public.documents (
  id bigserial primary key,
  symbol text,
  symbol_datetime timestamp,          -- รูปแบบ 2009-11-23 22:00:00
  content text not null,
  embedding extensions.vector(1024)
);

-- 3) (แนะนำ) สร้าง index สำหรับเวกเตอร์
create index if not exists documents_embedding_hnsw
on public.documents
using hnsw (embedding vector_cosine_ops);

-- 4) สร้างฟังก์ชันค้นหาใหม่
create or replace function public.match_documents(
  query_embedding extensions.vector(1024),
  match_threshold double precision,
  match_count integer
)
returns table (
  id bigint,
  symbol text,
  symbol_datetime timestamp,
  content text,
  similarity double precision
)
language sql stable
as $$
  select
    d.id,
    d.symbol,
    d.symbol_datetime,
    d.content,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  where d.embedding is not null
    and 1 - (d.embedding <=> query_embedding) > match_threshold
  order by d.embedding <=> query_embedding asc
  limit least(match_count, 200);
$$;

-- 5) สิทธิ์สำหรับ anon/authenticated
grant usage on schema public to anon, authenticated;
grant usage on schema extensions to anon, authenticated;
grant execute on function public.match_documents(extensions.vector, double precision, integer) to anon, authenticated;

-- 6) รีโหลด schema ให้ PostgREST เห็นฟังก์ชัน
notify pgrst, 'reload schema';


alter table public.documents enable row level security;

drop policy if exists "read documents" on public.documents;
create policy "read documents"
on public.documents
for select
to anon, authenticated
using (true);

drop policy if exists "insert documents" on public.documents;
create policy "insert documents"
on public.documents
for insert
to anon, authenticated
with check (true);
