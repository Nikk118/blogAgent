import os, re

d=r'frontend/components/dashboard'
if not os.path.exists(d):
    print(f'not found: {d}, cwd: {os.getcwd()}')

files = [os.path.join(d,f) for f in os.listdir(d) if f.endswith('.tsx')]
replacements = [
    (r'bg-\[\#030407\]', r'bg-slate-50'),
    (r'text-zinc-50\b', r'text-slate-900'),
    (r'text-zinc-100\b', r'text-slate-900'),
    (r'text-zinc-200\b', r'text-slate-800'),
    (r'text-zinc-300\b', r'text-slate-700'),
    (r'text-zinc-400\b', r'text-slate-600'),
    (r'text-zinc-500\b', r'text-slate-500'),
    (r'text-zinc-600\b', r'text-slate-500'),
    (r'bg-zinc-800\b', r'bg-white'),
    (r'bg-zinc-900\b', r'bg-white'),
    (r'bg-zinc-950\b', r'bg-white'),
    (r'border-zinc-800\b', r'border-slate-200'),
    (r'border-zinc-700\b', r'border-slate-200'),
    (r'border-white/10\b', r'border-slate-200'),
    (r'border-white/20\b', r'border-slate-300'),
    (r'bg-white/5\b', r'bg-white'),
    (r'bg-white/10\b', r'bg-slate-100'),
    (r'bg-white/\[0\.02\]', r'bg-white'),
    (r'bg-white/\[0\.04\]', r'bg-white'),
    (r'bg-black/20\b', r'bg-slate-100'),
    (r'bg-black/25\b', r'bg-slate-100'),
    (r'bg-black/50\b', r'bg-slate-200'),
    (r'ring-white/10\b', r'ring-slate-200'),
    (r'ring-white/5\b', r'ring-slate-100'),
    (r'cinematic-light-intense', r'bg-white shadow-[0_4px_24px_rgba(0,0,0,0.05)]'),
    (r'cinematic-light', r'bg-white shadow-sm'),
    (r'text-teal-200\b', r'text-teal-700'),
    (r'text-teal-100\b', r'text-teal-800'),
    (r'text-teal-300\b', r'text-teal-600'),
    (r'bg-teal-300/10\b', r'bg-teal-50'),
    (r'border-teal-300/20\b', r'border-teal-200'),
    (r'text-indigo-200\b', r'text-indigo-700'),
    (r'bg-indigo-300/10\b', r'bg-indigo-50'),
    (r'border-indigo-300/20\b', r'border-indigo-200'),
    (r'text-cyan-200\b', r'text-cyan-700'),
    (r'bg-cyan-300/10\b', r'bg-cyan-50'),
    (r'text-emerald-200\b', r'text-emerald-700'),
    (r'bg-emerald-300/10\b', r'bg-emerald-50'),
    (r'text-amber-200\b', r'text-amber-700'),
    (r'bg-amber-300/10\b', r'bg-amber-50'),
    (r'text-fuchsia-200\b', r'text-fuchsia-700'),
    (r'bg-fuchsia-300/10\b', r'bg-fuchsia-50'),
    (r'text-blue-500\b', r'text-blue-600'),
    (r'text-blue-400\b', r'text-blue-600')
]
for p in files:
    with open(p, 'r', encoding='utf-8') as f: c=f.read()
    for (pat, rep) in replacements: c=re.sub(pat, rep, c)
    with open(p, 'w', encoding='utf-8') as f: f.write(c)
print('Done!')
